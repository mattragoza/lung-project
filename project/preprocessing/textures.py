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


class TextureCache:

    def __init__(
        self,
        annotations: str,
        iqr_mult: float=DEFAULT_IQR_MULT,
        use_solid: bool=DEFAULT_USE_SOLID,
        weights=None
    ):
        self.df = load_texture_annotations(annotations)
        self.iqr_mult  = iqr_mult
        self.use_solid = use_solid
        self.weights   = weights

    def sample_field(self, tex_type, points, rng):
        sel = (self.df.texture_class == tex_type)
        if not sel.any():
            print(self.df)
            raise RuntimeError(f'No textures for material name {mat_name!r}')
        return sample_texture_field(
            self.df[sel],
            points=points,
            rng=rng,
            iqr_mult=self.iqr_mult,
            use_solid=self.use_solid,
            weights=self.weights
        )


def load_texture_annotations(path):
    import pandas as pd
    df = pd.read_csv(path)
    print()
    assert set(df.columns) >= {
        'texture_class',
        'image_path',
        'image_valid',
        'solid_path',
        'solid_valid',
        'inverted'
    }, df.columns
    return df


@lru_cache
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


@lru_cache
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
        valid = mat_tex_df.solid_valid.fillna(False)
        mat_tex_df = mat_tex_df[valid]
        assert len(mat_tex_df) > 0
        sampled = mat_tex_df.sample(1, random_state=rng).iloc[0]
        texture = load_texture_3d(sampled.solid_path, iqr_mult, sampled.inverted)
        return interpolate_solid(texture, points)
    else: # use triplanar
        valid = mat_tex_df.image_valid.fillna(False)
        mat_tex_df = mat_tex_df[valid]
        assert len(mat_tex_df) > 0
        sampled = mat_tex_df.sample(3, replace=True)
        textures = [
            load_texture_2d(s.image_path, iqr_mult, s.inverted)
                for i, s in sampled.iterrows()
        ]
        return interpolate_triplanar(textures, points, weights)

