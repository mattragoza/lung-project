import numpy as np
import pandas as pd

from ..core import utils


DATA_COLUMNS = ['key', 'val', 'freq']

DENSITY_DATA = [ # kg/m^3
    ('Dense',  500., 0.75),
    ('Porous', 250., 0.25),
]
ELASTIC_DATA = [ # Pa
    ('Hard',   9e3, 0.25),
    ('Medium', 3e3, 0.50),
    ('Soft',   1e3, 0.25),
]
POISSON_RATIO = 0.4


def _make_data_frame(data, prefix):
    return pd.DataFrame(data, columns=[f'{prefix}_{c}' for c in DATA_COLUMNS])


def build_material_catalog(include_background=True):
    dens = _make_data_frame(DENSITY_DATA, prefix='density')
    elas = _make_data_frame(ELASTIC_DATA, prefix='elastic')

    mats = pd.merge(dens, elas, how='cross')
    mats['material_key'] = mats['density_key'] + mats['elastic_key']
    mats['material_freq'] = mats['density_freq'] * mats['elastic_freq']
    mats['material_freq'] /= mats['material_freq'].sum()

    mats['poisson_ratio'] = POISSON_RATIO # constant for now

    if include_background:
        mats.loc[-1, :] = 0.
        mats.loc[-1, 'material_key'] = 'Background'

        # shift index so background is 0
        mats.index += 1

    return mats.sort_index()


def assign_image_parameters(
    mat_df,
    d_ref=1, d_pow=1, # density feature
    e_ref=1, e_pow=1, # elastic feature
    b0=0, b_d=0, b_e=0, b_de=0, # image bias coefs
    r0=0, r_d=0, r_e=0, r_de=0, # image range coefs
    background=0,
    eps=1e-6
):
    df = mat_df.copy()

    def _compute_feature(val, ref, power):
        return np.maximum(val / ref, eps) ** power

    d = _compute_feature(df['density_val'], d_ref, d_pow)
    e = _compute_feature(df['elastic_val'], e_ref, e_pow)

    def _compute_param(x, y, c0, c_x, c_y, c_xy):
        return c0 + c_x * x + c_y * y + c_xy * x * y

    df['density_feat'] = d
    df['elastic_feat'] = e
    df['image_bias']  = _compute_param(d, e, b0, b_d, b_e, b_de)
    df['image_range'] = _compute_param(d, e, r0, r_d, r_e, r_de)

    return df


def sample_region_materials(
    region_mask, prior, vote_rate=1e-3, min_votes=1, seed=0
):
    '''
    Assign materials to a multi-label region mask by sampling
    votes from a prior distribution over a set of materials.

    Iterate over the labeled regions from largest to smallest,
    sampling votes from the prior in proportion to region size.
    Assign the top-ranked material not already assigned to a
    neighboring region.

    Args:
        region_mask: (I, J, K) int array of region labels
        prior: (M,) float array of material probabilities
        vote_rate: number of votes per voxel (default: 1e-3)
        min_votes: minimum votes per region (default: 1)
        seed: random seed
    Returns:
        material_map: (N,) int array mapping region labels
            to material indices, with -1 for not assigned
    '''
    import skimage
    rng = np.random.default_rng(seed)

    # compute region sizes
    regions, sizes = np.unique(region_mask, return_counts=True)

    sel = regions > 0 # exclude background
    regions, sizes = regions[sel], sizes[sel]
    n_votes = np.maximum(vote_rate * sizes, min_votes)

    utils.log(f'Region labels: {regions}')
    utils.log(f'Region sizes:  {sizes}')
    utils.log(f'Region votes: {n_votes}')

    if regions.size == 0:
        return np.zeros(1, dtype=int) # only background

    size_order = np.argsort(-sizes)

    utils.log('Building region adjacency graph')
    g = skimage.graph.rag_boundary(
        region_mask.astype(int, copy=False),
        edge_map=np.ones_like(region_mask, dtype=float),
        connectivity=3
    )
    adjacent = {int(n): {int(nb) for nb in g.neighbors(n)} for n in g.nodes}

    # prior distribution over materials
    prior = np.asarray(prior, dtype=float)
    prior /= prior.sum()
    materials = np.arange(1, prior.size + 1, dtype=int)

    utils.log(f'Material prior: {prior}')

    assigned = {}
    for idx in size_order:
        region, size = int(regions[idx]), int(sizes[idx])

        # sample votes from prior distribution
        n_votes = max(min_votes, int(vote_rate * size))
        votes = rng.multinomial(n_votes, prior)

        # apply slight jitter to break ties
        jitter = rng.uniform(-0.1, 0.1, size=votes.size)
        ranked = materials[np.argsort(-(votes + jitter))]

        neighbors = {assigned.get(nb) for nb in adjacent.get(region, [])}
        neighbors.discard(None)

        choice = None
        for candidate in ranked:
            if candidate not in neighbors:
                choice = candidate
                break

        if choice is None: # cannot avoid a conflict
            utils.warn(f'WARNING: adjacent regions must share a material')
            choice = ranked[0]

        utils.log(f'Region {region} was assigned material {choice} in prior')
        assigned[region] = int(choice)

    max_region = int(regions.max())
    material_map = np.zeros(max_region + 1, dtype=int)
    for r, m in assigned.items():
        material_map[r] = m

    return material_map


def assign_materials_to_regions(region_mask, sample_kws=None):
    region_mask = region_mask.astype(int, copy=False)

    utils.log('Building material catalog')
    mat_df = build_material_catalog(include_background=True)
    utils.log(mat_df)

    utils.log('Sampling materials per region')
    prior = mat_df.loc[1:, 'material_freq'].to_numpy() # exclude background
    material_by_region = sample_region_materials(region_mask, prior, **(sample_kws or {}))

    utils.log(material_by_region)
    return material_by_region


def assign_material_properties(material_labels):
    mat_df = build_material_catalog(include_background=True)

    density_by_material = mat_df['density_val'].to_numpy()
    elastic_by_material = mat_df['elastic_val'].to_numpy()

    if material_labels.min() < 0:
        raise ValueError(material_labels.unique())

    density_values = density_by_material[material_labels]
    elastic_values = elastic_by_material[material_labels]

    return density_values, elastic_values


def compute_region_materials(reg_mask, mat_mask):
    lut = -np.ones(reg_mask.max() + 1, dtype=int)
    for r in np.unique(reg_mask):
        vals = mat_mask[reg_mask == r]
        lut[r] = int(np.bincount(vals).argmax())
    return lut

