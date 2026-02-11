from __future__ import annotations
from typing import Dict
import numpy as np
import scipy.ndimage

from ..core import utils


def generate_simple_image(
    mat_mask: np.ndarray,
    texture_map: Dict[int, np.ndarray],
    texel_scale: float = 1.0,
    trans_sigma: float = 8.0,
    interp_order: int = 1,
    interp_mode: str = 'wrap',
    seed: int = 0,
    rgb = True
):
    mat_mask = np.asarray(mat_mask, dtype=int)
    assert mat_mask.ndim == 3

    I, J, K = mat_mask.shape
    points = grid_coords((I, J, K)).astype(np.float32)

    shape = (I, J, K, 3) if rgb else (I, J, K)
    image = np.zeros(shape, dtype=np.float32)

    rng = np.random.default_rng(seed)

    for label in sorted(np.unique(mat_mask)):
        loc = (mat_mask == label)
        try:
            tex = texture_map(label)
        except KeyError:
            continue

        img_pts = points[loc]
        img_ctr = (np.array([I, J, K], dtype=np.float32) - 1) / 2
        tex_ctr = (np.array(tex.shape[:3]) - 1) / 2
        tex_pts = (img_pts - img_ctr) * texel_scale + tex_ctr

        interp_pts = random_transform(tex_pts, trans_sigma, rng)
        image[loc] = interpolate_volume(tex, interp_pts, interp_order, interp_mode)

    return image


def grid_coords(shape, axis=-1):
    xs = (np.arange(n) for n in shape)
    xs = np.meshgrid(*xs, indexing='ij')
    return np.stack(xs, axis=axis)


def random_transform(points, sigma=0, rng=None):
    import scipy.stats as stats

    points = np.asarray(points)
    assert points.ndim == 2 and points.shape[-1] == 3

    R = stats.special_ortho_group.rvs(3, random_state=rng)
    t = rng.normal(scale=sigma, size=3)

    center = points.mean(axis=0)
    return (points - center) @ R.T + t + center


def interpolate_volume(vol, points, order=1, mode='wrap', background=0):
    import scipy.ndimage
    vol, pts = np.asarray(vol), np.asarray(points)

    assert vol.ndim in {3, 4}, vol.shape
    rgb = (vol.ndim == 4)
    if rgb:
        assert vol.shape[-1] == 3, vol.shape

    assert pts.ndim == 2 and pts.shape[-1] == 3, pts.shape
    
    if not rgb:
        return scipy.ndimage.map_coordinates(
            input=vol,
            coordinates=pts.T,
            order=order,
            prefilter=(order > 1),
            mode=mode,
            cval=background
        )

    out_shape = (pts.shape[0], 3)
    out = np.full(out_shape, background, dtype=vol.dtype)
    
    for c in range(3):
        out[:,c] = scipy.ndimage.map_coordinates(
            input=vol[...,c],
            coordinates=pts.T,
            order=order,
            prefilter=(order > 1),
            mode=mode,
            cval=background
        )
    return out


## DEPRECATED


def generate_volumetric_image(
    mat_mask: np.ndarray,
    affine: np.ndarray,
    mat_df: pd.DataFrame,
    tex_cache: textures.TextureCache,
    # noise settings
    tex_noise_len: float=0.,
    tex_noise_std: float=0.,
    mul_noise_len: float=0.,
    mul_noise_std: float=0.,
    add_noise_len: float=0.,
    add_noise_std: float=0.,
    mat_sigma: float=1.,
    psf_sigma: float=0.,
    random_seed=0,
):
    '''
    Args:
        mat_mask: (I, J, K) voxel mask of material labels
        affine: (4, 4) voxel -> world coordinate mapping
        mat_df: material catalog df indexed by label
        tex_cache: TextureCache
    Returns:
        (I, J, K) volumetric image with intensity bias and
            texture derived from the material labels
    '''
    rng = np.random.default_rng(random_seed)

    I, J, K = mat_mask.shape
    points = np.stack(np.mgrid[0:I,0:J,0:K], axis=-1).reshape(-1, 3)
    image  = np.zeros((I, J, K), dtype=np.float32)
    
    tex_noise = bandlimited_noise(mat_mask.shape, tex_noise_len, rng) * tex_noise_std
    mul_noise = bandlimited_noise(mat_mask.shape, mul_noise_len, rng) * mul_noise_std
    add_noise = bandlimited_noise(mat_mask.shape, add_noise_len, rng) * add_noise_std

    # compute material blend weights
    blend = compute_blend_weights(mat_mask, mat_sigma)

    # for each unique label in material mask,
    for i, label in enumerate(np.unique(mat_mask)):

        # select region with material label
        sel_region = (mat_mask == label)
        sel_points = points[sel_region.reshape(-1)]

        # get material info for region
        mat_info = mat_df.loc[label]
        mat_name = mat_info.material_name
        tex_type = mat_info.texture_class
        img_bias = mat_info.intensity_bias
        img_range = mat_info.intensity_range if label > 0 else 0

        utils.log(f'{mat_name} | bias = {img_bias:.4f} range = {img_range:.4f}')

        if label > 0:
            t_interp = tex_cache.sample_field(tex_type, points, rng).reshape(I, J, K)
        else:
            t_interp = np.zeros((I, J, K), dtype=float)
        image += blend[i] * (img_bias + img_range * (t_interp + tex_noise))

    # global multiplicative noise
    image *= (1.0 + mul_noise)

    if psf_sigma > 0:
        image = scipy.ndimage.gaussian_filter(image, sigma=psf_sigma, mode='wrap')

    # global additive noise
    image += add_noise

    return image


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


def compute_blend_weights(mat_mask, sigma, eps=1e-12):
    I, J, K = mat_mask.shape
    labels = np.unique(mat_mask)
    blend = np.zeros((len(labels), I, J, K), dtype=float)
    for i, label in enumerate(labels):
        if label > 0 and sigma > 0:
            binary = (mat_mask == label).astype(float)
            filtered = scipy.ndimage.gaussian_filter(binary, sigma=sigma, mode='reflect')
            blend[i] = (mat_mask != 0) * filtered
        else:
            blend[i] = (mat_mask == label)
    return blend / (blend.sum(axis=0, keepdims=True) + eps)

