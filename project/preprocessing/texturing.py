import random
import imageio
import numpy as np
import scipy.stats

from ..core import utils, fileio, transforms


TEXTURE_TO_MATERIAL = {
    'paper':   'DenseSoft',
    'leather': 'DenseMedium',
    'stone':   'DenseHard',
    'fabric':  'PorousSoft',
    'wood':    'PorousMedium',
    'marble':  'PorousHard'
}


class TextureSampler:

    def __init__(self, texture_dir, seed=0, pos=0, log=None, exts=['.jpg', '.jpeg']):
        self.paths = [p for p in texture_dir.rglob('*') if p.suffix.lower() in exts]
        if not self.paths:
            raise ValueError('No textures found')
        else:
            print(f'{len(self.paths)} textures found')
        rng = np.random.default_rng(seed)
        self.order = rng.permutation(len(self.paths)).tolist()
        self.pos = pos
        self.log = log or []

    def __len__(self):
        return len(self.paths)

    def next(self):
        if self.pos >= len(self.order):
            raise StopIteration
        idx = self.order[self.pos]
        self.pos += 1
        return idx, self.paths[idx]

    def peek(self):
        if self.pos >= len(self.order):
            raise StopIteration
        idx = self.order[self.pos]
        return idx, self.paths[idx]

    def annotate(self, idx, annotation):
        self.log.append((idx, self.paths[idx], str(annotation)))

    def save(self, path):
        import csv
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['idx', 'path', 'annotation'])
            w.writerows(self.log)



def show_image(a, title=None, ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots()
    if img.ndim == 2:
        ret = ax.imshow(a, cmap='gray')
    elif img.ndim == 3:
        ret = ax.imshow(a)
    else:
        raise ValueError(f'cannot show array shape {a.shape} as image')
    if title:
        ax.set_title(title)
    ax.axis('off')
    return ret


def load_annotations(path):
    import pandas as pd
    df = pd.read_csv(path)
    assert set(df.columns) >= {'path', 'annotation', 'material', 'inverted'}
    return df


def load_texture(row):
    img = fileio.load_imageio(row.path, quiet=True)
    return preprocess(img, row.inverted)


def build_texture_cache(path):
    df = load_annotations(path)
    df['image'] = df.apply(load_texture, axis=1)
    return df


def _rgb(a):
    return a.ndim == 3 and a.shape[-1] == 3


def _rgba(a):
    return a.ndim == 3 and a.shape[-1] == 4


def preprocess(img, invert=False):
    import skimage
    x = skimage.util.img_as_float(img)
    if x.ndim != 2 and not (_rgb(x) or _rgba(x)):
        raise ValueError(f'cannot interpret {x.shape} as image')
    if _rgba(x):
        x = skimage.color.rgba2rgb(x)
    if _rgb(x):
        x = skimage.color.rgb2gray(x)
    x = normalize(x, 1.5)
    x = np.clip(x, -1., 1.)
    return 1 - x if invert else x


def normalize(x, iqr_mult=1.5):
    q1, q2, q3 = np.percentile(x, [25, 50, 75])
    iqr = q3 - q1
    hi = q3 + iqr * iqr_mult
    lo = q1 - iqr * iqr_mult
    return (x - q2) / (hi - lo)


def show_textures(df, max_rows=6, max_cols=6):
    from ..visual.matplotlib import subplot_grid

    index_vals = df.index.unique().sort_values().dropna()
    groups = [df.loc[ival] for ival in index_vals]
    n_rows = min(max_rows, len(index_vals))
    n_cols = min(max_cols, max(len(g) for g in groups))
    print(n_rows, n_cols)

    fig, axes = subplot_grid(
        n_rows, n_cols,
        ax_height=1.5,
        ax_width=1.5,
        spacing=(0.5, 0.5), # hw
        padding=(0.75, 0.75, 0.5, 0.25), # lrbt
    )
    for i, ival in enumerate(index_vals):
        axes[i,0].set_ylabel(ival)
        for j, (idx, row) in enumerate(groups[i].iterrows()):
            ax = axes[i,j]
            img = imageio.v2.imread(row.path)
            img = normalize_texture(img)
            if row.inverted:
                img = 1 - img
            H, W = img.shape
            ax.imshow(img, cmap='gray', extent=(0, W - 1, 0, H - 1))

        for j in range(j+1, n_cols):
            axes[i,j].axis('off')

    return fig


def build_affine_matrix_2d(origin, spacing):
    A = np.eye(3, dtype=float)
    A[:2,:2] = np.diag(spacing) @ scipy.stats.ortho_group.rvs(2)
    A[:2,2] = np.array(origin)
    return A


def world_to_pixel_coords(points, affine):
    A = np.asarray(affine)
    assert A.shape == (3, 3)
    H = transforms._homogeneous(points)
    output = np.linalg.solve(A, H.T).T
    return output[:,:-1] / output[:,-1:]


def interpolate_triplanar(textures, points, affine, weights=None):
    if weights is None:
        weights = np.ones(3, dtype=float) / 3

    yz, xz, xy = [1,2], [0,2], [0,1]
    spacing = np.linalg.norm(affine, axis=0)
    origin = affine[:3,3]

    s_yz = spacing[0] #tile_m / (np.max(textures[0].shape) - 1) 
    s_xz = spacing[1] #tile_m / (np.max(textures[1].shape) - 1)
    s_xy = spacing[2] #tile_m / (np.max(textures[2].shape) - 1)

    A_yz = build_affine_matrix_2d(origin[yz], [s_yz, s_yz])
    A_xz = build_affine_matrix_2d(origin[xz], [s_xz, s_xz])
    A_xy = build_affine_matrix_2d(origin[xy], [s_xy, s_xy])
    
    uv_yz = world_to_pixel_coords(points[:,yz], A_yz)
    uv_xz = world_to_pixel_coords(points[:,xz], A_xz)
    uv_xy = world_to_pixel_coords(points[:,xy], A_xy)

    t_yz = scipy.ndimage.map_coordinates(textures[0], uv_yz.T, mode='wrap', order=1, prefilter=False)
    t_xz = scipy.ndimage.map_coordinates(textures[1], uv_xz.T, mode='wrap', order=1, prefilter=False)
    t_xy = scipy.ndimage.map_coordinates(textures[2], uv_xy.T, mode='wrap', order=1, prefilter=False)
    
    return weights[0] * t_yz + weights[1] * t_xz + weights[2] * t_xy


def generate_volumetric_image(mask, affine, tex_cache, mats, weights=[1.,1.,1.], seed=0):
    random.seed(seed)
    I, J, K = mask.shape
    image = np.zeros((I, J, K), dtype=np.float32)

    points = np.stack(np.mgrid[0:I,0:J,0:K], axis=-1).reshape(-1, 3)
    points = transforms.voxel_to_world_coords(points, affine)

    for mask_val in np.unique(mask):
        if mask_val == 0: # skip background
            continue

        region = (mask == mask_val)
        sel_points = points[region.reshape(-1)]

        mat_key = mats.material_key.loc[mask_val]
        mat_tex = tex_cache[tex_cache.material == mat_key].image
        sel_tex = random.choices(list(mat_tex.values), k=3)
        utils.pprint(sel_tex)

        T_interp = interpolate_triplanar(sel_tex, sel_points, affine, weights)
        image[region] = 0.25 + 0.1 * mask_val + 0.5 * T_interp

    return np.clip(image, 0., 1.)

