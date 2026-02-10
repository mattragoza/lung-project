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


def build_material_catalog(add_background=True):
    dens = _make_data_frame(DENSITY_DATA, prefix='density')
    elas = _make_data_frame(ELASTIC_DATA, prefix='elastic')

    mats = pd.merge(dens, elas, how='cross')
    mats['material_name'] = mats['density_key'] + mats['elastic_key']
    mats['material_freq'] = mats['density_freq'] * mats['elastic_freq']
    mats['material_freq'] /= mats['material_freq'].sum()

    mats['poisson_ratio'] = POISSON_RATIO # constant for now

    if add_background:
        mats.loc[-1, :] = pd.NA
        mats.loc[-1, 'material_key'] = 'Background'
        mats.index += 1 # shift so background index is 0

    return mats.sort_index()


def load_material_catalog(csv_path, add_background=True):
    import pandas as pd
    req_cols = ['material_name', 'density_val', 'elastic_val', 'material_freq']
    mat_df = pd.read_csv(csv_path)
    assert set(mat_df.columns) >= set(req_cols)
    if add_background:
        mat_df.loc[-1, :] = pd.NA
        mat_df.loc[-1, 'material_name'] = 'Background'
        mat_df.index += 1 # shift so background index is 0
    return mat_df.sort_index()


def compute_intensity_model(
    density: np.ndarray,
    elastic: np.ndarray,
    density_scale: float=1.0,
    elastic_scale: float=1.0,
    density_power: float=1.0,
    elastic_power: float=1.0,
    bias_offset: float=0.0,
    bias_from_density: float=0.0,
    bias_from_elastic: float=0.0,
    bias_from_product: float=0.0,
    range_offset: float=0.0,
    range_from_density: float=0.0,
    range_from_elastic: float=0.0,
    range_from_product: float=0.0,
    eps=1e-8
):
    def _compute_feature(val, scale, power):
        return np.maximum(val.astype(float) / float(scale), eps) ** float(power)

    d = _compute_feature(density, density_scale, density_power)
    e = _compute_feature(elastic, elastic_scale, elastic_power)

    def _compute_param(x, y, c0, c_x, c_y, c_xy):
        return c0 + float(c_x) * x + float(c_y) * y + float(c_xy) * x*y

    b = _compute_param(d, e, bias_offset, bias_from_density, bias_from_elastic, bias_from_product)
    r = _compute_param(d, e, range_offset, range_from_density, range_from_elastic, range_from_product)

    return {
        'density_feat': d,
        'elastic_feat': e,
        'intensity_bias': b,
        'intensity_range': r
    }


# ----- assigning materials to regions -----


def sample_region_materials(
    region_mask, prior, sample_rate=0, min_samples=1, random_seed=0
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
        min_samples: minimum samples per region (default: 1)
        sample_rate: number of samples per voxel (default: 1e-3)
            Reduce this parameter to increase the variance.
        random_seed
    Returns:
        material_map: (N,) int array mapping region labels
            to material indices, with -1 for not assigned
    '''
    import skimage
    rng = np.random.default_rng(random_seed)

    # compute region sizes
    regions, sizes = np.unique(region_mask, return_counts=True)

    sel = regions > 0 # exclude background
    regions, sizes = regions[sel], sizes[sel]
    n_samples = np.maximum(min_samples, sizes * sample_rate)

    utils.log(f'Region labels: {regions}')
    utils.log(f'Region sizes:  {sizes}')
    utils.log(f'Region votes:  {n_samples}')

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

        # sample materials from prior distribution
        n_samples = max(min_samples, int(size * sample_rate))
        samples = rng.multinomial(n_samples, prior)

        # apply slight jitter to break ties
        jitter = rng.uniform(-0.1, 0.1, size=samples.size)
        ranked = materials[np.argsort(-(samples + jitter))]

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


def assign_materials_to_regions(region_mask, mat_df, sampling_kws=None, random_seed=0):
    region_mask = region_mask.astype(int, copy=False)

    utils.log('Sampling materials per region')
    prior = mat_df.loc[1:, 'material_freq'].to_numpy() # exclude background
    material_by_region = sample_region_materials(
        region_mask, prior, **(sampling_kws or {}), random_seed=random_seed
    )

    utils.log(material_by_region)
    return material_by_region


def assign_material_properties(material_labels, mat_df):

    density_by_material = mat_df['density_val'].to_numpy()
    elastic_by_material = mat_df['elastic_val'].to_numpy()

    if material_labels.min() < 0:
        raise ValueError(material_labels.unique())

    density_values = density_by_material[material_labels]
    elastic_values = elastic_by_material[material_labels]

    return density_values, elastic_values


def infer_material_by_region(region_labels, material_labels):
    most_common = -np.ones(region_labels.max() + 1, dtype=int)
    for r in np.unique(region_labels):
        m = material_labels[region_labels == r]
        most_common[r] = int(np.bincount(m).argmax())
    return most_common

