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


def show_image(array, title=None, ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots()
    if array.ndim == 2:
        ret = ax.imshow(array, cmap='gray')
    elif array.ndim == 3:
        ret = ax.imshow(array)
    else:
        raise ValueError(f'cannot show array shape {array.shape} as image')
    if title:
        ax.set_title(title)
    ax.axis('off')
    return ret


def load_annotations(path):
    import pandas as pd
    df = pd.read_csv(path)
    assert set(df.columns) >= {'path', 'annotation', 'material', 'inverted'}
    return df


def build_texture_cache(path, **kwargs):
    df = load_annotations(path)
    df['image'] = df.apply(load_texture, axis=1)
    df['image'] = df.apply(preprocess, axis=1, **kwargs)
    return df


def load_texture(row):
    return fileio.load_imageio(row.path, quiet=True)


def _rgb(a):
    return a.ndim == 3 and a.shape[-1] == 3


def _rgba(a):
    return a.ndim == 3 and a.shape[-1] == 4


def preprocess(row, iqr_mult=1.5):
    import skimage
    x = skimage.util.img_as_float(row.image)
    if x.ndim != 2 and not (_rgb(x) or _rgba(x)):
        raise ValueError(f'cannot interpret {x.shape} as image')
    if _rgba(x):
        x = skimage.color.rgba2rgb(x)
    if _rgb(x):
        x = skimage.color.rgb2gray(x)
    x = _normalize_iqr(x, iqr_mult=iqr_mult)
    x = np.clip(x, -1., 1.).astype(np.float32)
    return 1. - x if getattr(row, 'inverted') else x


def _normalize_iqr(x, iqr_mult=1.5, eps=1e-6):
    '''
    Center at median, scale by multiple of IQR.
    '''
    x = np.asarray(x, dtype=np.float32)
    q1, q2, q3 = np.percentile(x, [25, 50, 75])
    center = q2

    hi = q2 + iqr_mult * (q3 - q2)
    lo = q2 - iqr_mult * (q2 - q1)
    scale = max([hi - q2, q2 - lo, eps])

    return (x - center) / scale


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
            img = row.image
            H, W = img.shape
            ax.imshow(img, cmap='gray', extent=(0, W - 1, 0, H - 1))

        for j in range(j+1, n_cols):
            axes[i,j].axis('off')

    return fig


def build_affine_matrix_2d(origin, spacing, rotate=False):
    A = np.eye(3, dtype=float)
    A[:2,:2] = np.diag(spacing)
    if rotate:
        A[:2,:2] @= scipy.stats.ortho_group.rvs(2)
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
        weights = np.ones(3, dtype=float)

    weights = np.asarray(weights, dtype=float)
    s = weights.sum()
    if s <= 0:
        raise ValueError('triplanar weights must sum > 0')
    weights = weights / s

    yz, xz, xy = [1,2], [0,2], [0,1]
    spacing = np.linalg.norm(affine[:3,:3], axis=0)
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


def generate_volumetric_image(
    mat_mask,
    affine,
    mat_df,
    tex_cache,
    weights=None,
    seed=0,

    # image parameters
    bias_0:  float=-1000.,
    bias_d:  float=1000.,
    bias_e:  float=0.,
    range_0: float=250.,
    range_d: float=0,
    range_e: float=250.,

    # noise parameters
    t_noise_corr: float=0.,
    t_noise_std:  float=0.,
    b_noise_corr: float=0.,
    b_noise_std:  float=0.,
    s_noise_std:  float=0.,
):
    '''
    Args:
        mat_mask: (I, J, K) voxel mask of material labels
        affine: (4, 4) voxel -> world coordinate mapping
        tex_cache: data frame with material + image columns
        mat_df: material info data frame indexed by label
    Returns:
        (I, J, K) volumetric image with intensity bias and
            texture derived from the material labels
    '''
    rng = np.random.default_rng(seed)

    I, J, K = mat_mask.shape
    shape = np.asarray(mat_mask.shape)
    spacing = np.linalg.norm(affine[:3,:3], axis=0)
    max_spacing = spacing.max()

    points = np.stack(np.mgrid[0:I,0:J,0:K], axis=-1).reshape(-1, 3)
    points = transforms.voxel_to_world_coords(points, affine)

    image = np.zeros((I, J, K), dtype=np.float32)
    
    t_noise = bandlimited_noise(shape, spacing, t_noise_corr * max_spacing, seed)
    b_noise = bandlimited_noise(shape, spacing, b_noise_corr * max_spacing, seed)
    s_noise = np.random.normal(size=shape)

    # for each unique value in material mask,
    for label in np.unique(mat_mask):

        # select region with mask value
        region = (mat_mask == label)
        sel_points = points[region.reshape(-1)]

        # get material info for region
        mat_info = mat_df.loc[label]
        mat_key = mat_info.material_key

        # relationship bt/w material properties and image params
        d = _density_feature(mat_info.density_val)
        e = _elastic_feature(mat_info.elastic_val)
        mat_bias  = bias_0  + bias_d  * d + bias_e  * e
        mat_range = range_0 + range_d * d + range_e * e
        mat_range = 0. if label == 0 else mat_range # background

        utils.log(f'{label:d} {mat_key.rjust(12)} | mat_bias = {mat_bias:.2f} mat_range = {mat_range:.2f}')
 
        # sample texture images associated with material
        mat_tex = tex_cache[tex_cache.material == mat_key]
        if len(mat_tex) == 0:
            T_interp = np.zeros(len(sel_points), dtype=float)
        else:
            textures = np.random.choice(mat_tex.image, size=3, replace=True)

            # interpolate volumetric texture field at selected points
            T_interp = interpolate_triplanar(textures, sel_points, affine, weights)

        # add texture noise
        T_interp += t_noise_std * t_noise[region]

        # assign image values using image params + texture
        image[region] = mat_bias + mat_range * T_interp

    # global multiplicative noise
    if b_noise_std > 0:
        image *= (1.0 + b_noise_std * b_noise)

    # global additive noise
    if s_noise_std > 0:
        image += s_noise_std * s_noise

    return image


def _density_feature(rho, rho_ref=1000.):
    rho_rel = rho / rho_ref
    return np.maximum(rho_rel, 0.)


def _elastic_feature(E, E_ref=1000., power=-1, eps=1e-3):
    E_rel = E / E_ref
    return np.maximum(E_rel, eps)**power



def bandlimited_noise(shape, spacing, corr, seed=None, eps=1e-6):
    rng = np.random.default_rng(seed)

    z = rng.normal(size=shape).astype(np.float32)
    if corr < eps:
        return z

    kx, ky, kz = np.meshgrid(
        np.fft.fftfreq(shape[0], spacing[0]),
        np.fft.fftfreq(shape[1], spacing[1]),
        np.fft.fftfreq(shape[2], spacing[2]),
        indexing='ij'
    )
    kk = kx*kx + ky*ky + kz*kz

    sigma = 1.0 / (2*np.pi*corr)
    H = np.exp(-0.5 * kk / (sigma*sigma))
    Z = np.fft.fftn(z)
    zf = np.fft.ifftn(H * Z).real

    zf -= zf.mean()
    zf /= (zf.std() + eps)
    return zf
