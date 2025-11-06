from functools import lru_cache
import numpy as np
import pandas as pd
import scipy.stats
import scipy.ndimage
import skimage

from ..core import utils, fileio, transforms


TEXTURE_TO_MATERIAL = {
    'paper':   'DenseSoft',
    'leather': 'DenseMedium',
    'stone':   'DenseHard',
    'fabric':  'PorousSoft',
    'wood':    'PorousMedium',
    'marble':  'PorousHard'
}
DEFAULT_IQR_MULT = 4.0
DEFAULT_USE_SOLID = True


def load_texture_annotations(path):
    import pandas as pd
    df = pd.read_csv(path)
    assert set(df.columns) >= {'material', 'path', 'solid_path', 'solid_selected'}
    return df


@lru_cache(maxsize=256)
def load_texture_2d(path, iqr_mult=DEFAULT_IQR_MULT, invert=False):
    a = fileio.load_imageio(path)
    a = skimage.util.img_as_float(a)
    if a.ndim == 3 and a.shape[-1] == 4:
        a = skimage.color.rgba2rgb(a)
    if a.ndim == 3 and a.shape[-1] == 3:
        a = skimage.color.rgb2gray(a)
    if a.ndim != 2:
        raise ValueError(f'invalid 2d texture shape: {a.shape}')
    a = _normalize_median_iqr(a, iqr_mult=iqr_mult)
    a = np.clip(a, -1.0, 1.0)
    return 1. - a if invert else a


@lru_cache(maxsize=32)
def load_texture_3d(path, iqr_mult=DEFAULT_IQR_MULT, invert=False):
    a = fileio.load_nibabel(path).get_fdata()
    a = skimage.util.img_as_float(a)
    if a.ndim == 4 and a.shape[-1] == 4:
        a = skimage.color.rgba2rgb(a)
    if a.ndim == 4 and a.shape[-1] == 3:
        a = skimage.color.rgb2gray(a)
    if a.ndim != 3:
        raise ValueError(f'invalid 3d texture shape: {a.shape}')
    a = _normalize_median_iqr(a, iqr_mult=iqr_mult)
    a = np.clip(a, -1.0, 1.0)
    return 1. - a if invert else a


def _normalize_median_iqr(x, iqr_mult=DEFAULT_IQR_MULT, eps=1e-12):
    x = np.asarray(x, dtype=np.float32)
    q1, q2, q3 = np.percentile(x, [25, 50, 75])
    hi = q2 + iqr_mult * (q3 - q2) / 2
    lo = q2 - iqr_mult * (q2 - q1) / 2
    return (x - q2) / max([hi - q2, q2 - lo, eps])


def _validate_triplanar_weights(weights):
    if weights is None:
        weights = np.ones(3, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if np.any(weights < 0):
        raise ValueError('triplanar weights must be non-negative')
    return weights / weights.sum()


def _apply_random_transform(points, rng):
    points = np.asarray(points, dtype=float)
    center = points.mean(axis=0)
    R = scipy.stats.special_ortho_group.rvs(3, random_state=rng)
    t = rng.normal(scale=128, size=3)
    return (points - center) @ R.T + t + center


def interpolate_triplanar(images, points, weights=None):
    weights = _validate_triplanar_weights(weights)
    t_yz = scipy.ndimage.map_coordinates(images[0], points[:,[1,2]].T, mode='wrap', order=1, prefilter=False)
    t_xz = scipy.ndimage.map_coordinates(images[1], points[:,[0,2]].T, mode='wrap', order=1, prefilter=False)
    t_xy = scipy.ndimage.map_coordinates(images[2], points[:,[0,1]].T, mode='wrap', order=1, prefilter=False)
    return weights[0] * t_yz + weights[1] * t_xz + weights[2] * t_xy


def interpolate_solid(volume, points):
    return scipy.ndimage.map_coordinates(volume, points.T, mode='wrap', order=1, prefilter=False)


def sample_texture_field(
    mat_tex_df,
    points,
    rng,
    iqr_mult=DEFAULT_IQR_MULT,
    use_solid=DEFAULT_USE_SOLID,
    weights=None
):
    assert points.ndim == 2 and points.shape[1] == 3, points.shape
    points = _apply_random_transform(points, rng)
    if use_solid:
        if 'solid_selected' in mat_tex_df:
            selected = mat_tex_df.solid_selected.fillna(False)
            mat_tex_df = mat_tex_df[selected]
        assert len(mat_tex_df) > 0
        sampled = mat_tex_df.sample(1, random_state=rng).iloc[0]
        texture = load_texture_3d(sampled.solid_path, iqr_mult, sampled.inverted)
        return interpolate_solid(texture, points)
    else: # use triplanar
        assert len(mat_tex_df) > 0
        sampled = mat_tex_df.sample(3, replace=True)
        textures = [
            load_texture_2d(s.path, iqr_mult, s.inverted)
                for i, s in sampled.iterrows()
        ]
        return interpolate_triplanar(textures, points, weights)


def bandlimited_noise(shape, corr_len, rng, eps=1e-12):
    z = rng.normal(size=shape).astype(np.float32)
    if corr_len < eps:
        return z

    kx, ky, kz = np.meshgrid(
        np.fft.fftfreq(shape[0]),
        np.fft.fftfreq(shape[1]),
        np.fft.fftfreq(shape[2]),
        indexing='ij'
    )
    kk = kx*kx + ky*ky + kz*kz

    sigma = 1.0 / (2*np.pi*corr_len)
    H = np.exp(-0.5 * kk / (sigma*sigma))
    Z = np.fft.fftn(z)
    zf = np.fft.ifftn(H * Z).real

    zf -= zf.mean()
    zf /= (zf.std() + eps)
    return zf


def generate_volumetric_image(
    mat_mask: np.ndarray,
    affine: np.ndarray,
    mat_df: pd.DataFrame,
    tex_df: pd.DataFrame,

    # texture parameters
    iqr_mult: float=DEFAULT_IQR_MULT,
    use_solid: bool=DEFAULT_USE_SOLID,
    weights=None,
    seed=0,

    # noise parameters
    tex_noise_len: float=0.,
    tex_noise_std: float=0.,
    mul_noise_len: float=0.,
    mul_noise_std: float=0.,
    add_noise_len: float=0.,
    add_noise_std: float=0.,
    mat_sigma: float=1.,
    psf_sigma: float=0.,
):
    '''
    Args:
        mat_mask: (I, J, K) voxel mask of material labels
        affine: (4, 4) voxel -> world coordinate mapping
        mat_df: material catalog df indexed by label
        tex_df: texture annotations (materials and paths)
    Returns:
        (I, J, K) volumetric image with intensity bias and
            texture derived from the material labels
    '''
    rng = np.random.default_rng(seed)

    I, J, K = mat_mask.shape
    points = np.stack(np.mgrid[0:I,0:J,0:K], axis=-1).reshape(-1, 3)
    image  = np.zeros((I, J, K), dtype=np.float32)
    
    tex_noise = bandlimited_noise(mat_mask.shape, tex_noise_len, rng)
    mul_noise = bandlimited_noise(mat_mask.shape, mul_noise_len, rng)
    add_noise = bandlimited_noise(mat_mask.shape, add_noise_len, rng)

    labels = np.unique(mat_mask)

    # compute material blend weights
    blend = np.zeros((len(labels), I, J, K), dtype=float)
    for i, label in enumerate(labels):
        if i == 0:
            blend[i] = (mat_mask == 0)
        else:
            blend[i] = (mat_mask != 0) * scipy.ndimage.gaussian_filter((mat_mask == label).astype(float), sigma=mat_sigma, mode='wrap')

    blend /= (blend.sum(axis=0, keepdims=True) + 1e-12)

    #blend *= scipy.ndimage.gaussian_filter((mat_mask == label).astype(float), sigma=mat_sigma, mode='wrap')

    # for each unique value in material mask,
    for i, label in enumerate(labels):

        # select region with mask value
        sel_region = (mat_mask == label)
        sel_points = points[sel_region.reshape(-1)]

        # get material info for region
        mat_info = mat_df.loc[label]
        mat_name = mat_info.material_key
        img_bias = mat_info.image_bias
        img_range = mat_info.image_range if label > 0 else 0

        utils.log(f'{mat_name} | bias = {img_bias:.4f} range = {img_range:.4f}')

        # sample 3d texture field associated with material
        mat_tex_df = tex_df[(tex_df.material == mat_name)]

        if len(mat_tex_df) > 0:
            t_interp = sample_texture_field(mat_tex_df, points, rng, iqr_mult, use_solid, weights).reshape(I,J,K)
        else:
            t_interp = 0.0

        # add texture noise
        t_interp = t_interp + tex_noise_std * tex_noise

        # assign image values using image params + texture
        image += blend[i] * (img_bias + img_range * t_interp)

    # global multiplicative noise
    if mul_noise_std > 0:
        image *= (1.0 + mul_noise_std * mul_noise)

    image = scipy.ndimage.gaussian_filter(image, sigma=psf_sigma, mode='wrap')

    # global additive noise
    if add_noise_std > 0:
        image += add_noise_std * add_noise

    return image

