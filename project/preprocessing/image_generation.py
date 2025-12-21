from __future__ import annotations
import numpy as np
import scipy.ndimage

from ..core import utils


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
    seed=0,
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
    rng = np.random.default_rng(seed)

    I, J, K = mat_mask.shape
    points = np.stack(np.mgrid[0:I,0:J,0:K], axis=-1).reshape(-1, 3)
    image  = np.zeros((I, J, K), dtype=np.float32)
    
    tex_noise = bandlimited_noise(mat_mask.shape, tex_noise_len, rng) * tex_noise_std
    mul_noise = bandlimited_noise(mat_mask.shape, mul_noise_len, rng) * mul_noise_std
    add_noise = bandlimited_noise(mat_mask.shape, add_noise_len, rng) * add_noise_std

    labels = np.unique(mat_mask)

    # compute material blend weights
    blend = compute_blend_weights(mat_mask, mat_sigma)

    # for each unique value in material mask,
    for i, label in enumerate(labels):

        # select region with mask value
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


def compute_blend_weights(mat_mask, sigma):
    I, J, K = mat_mask.shape
    labels = np.unique(mat_mask)
    blend = np.zeros((len(labels), I, J, K), dtype=float)
    for i, label in enumerate(labels):
        if i == 0:
            blend[i] = (mat_mask == 0)
        else:
            binary = (mat_mask == label).astype(float)
            filtered = scipy.ndimage.gaussian_filter(binary, sigma=sigma, mode='wrap')
            blend[i] = (mat_mask != 0) * filtered
    return blend / (blend.sum(axis=0, keepdims=True) + 1e-12)

