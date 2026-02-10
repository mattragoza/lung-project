from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import numpy as np

from ..core import utils, fileio, transforms


@dataclass(frozen=True)
class PreprocessSpec:
    do_crop: bool = False
    crop_size: int = 256
    do_filter: bool = False
    cutoff_freq: float = 1/32
    grayscale: bool = False
    normalize: bool = False
    norm_mode: str = 'median_iqr'
    per_channel: bool = False


class TextureCache:

    def __init__(self, annotations: pd.DataFrame):
        self.df = annotations
        self._cache = {}

    def __len__(self):
        return len(self.df)

    def get(self, tid: str, use_solid: bool, proc_spec: PreprocessSpec=None):
        row = self.df.loc[tid]

        key = (tid, use_solid, proc_spec)
        if key not in self._cache:

            if proc_spec is None:
                val = self.load(row, use_solid)
            else:
                raw = self.get(tid, use_solid, proc_spec=None)
                val = preprocess_texture(raw, **vars(proc_spec))

            self._cache[key] = val

        return self._cache[key]

    def load(self, row, use_solid):

        if use_solid:
            tex = fileio.load_nibabel(row.solid_path).get_fdata()
            assert tex.ndim == 4 and tex.shape[-1] == 3
    
            tex = tex[...,::-1] # flip RGB order
            if row.name in {'b25c0f2ee0edc563'}:
                tex = tex[::2,::2,::2] # low res
        else:
            tex = fileio.load_imageio(row.image_path)
            assert tex.ndim == 3 and tex.shape[-1] == 3

        tex = tex.astype(np.uint8, casting='same_value')
        if row.inverted or (tex.min() > 0 and tex.max() == 255):
            tex = 255 - tex # invert values

        return tex

    def load_all(self):
        for idx in range(len(self.df)):
            self.__getitem__(idx)

    def clear(self):
        self._cache.clear()


def load_texture_annotations(path: str):
    import pandas as pd

    df = pd.read_csv(path)
    required_cols = {'tid', 'texture_class', 'image_path', 'solid_path'}
    assert set(df.columns) >= required_cols, df.columns

    df['image_valid'] = df['image_valid'].fillna(False)
    df['solid_valid'] = df['solid_valid'].fillna(False)
    df['inverted'] = df['inverted'].fillna(False)

    return df.set_index('tid')


# ----- texture preprocessing -----


def preprocess_texture(
    tex: np.ndarray,
    do_crop: bool = False,
    crop_size: int = 256,
    do_filter: bool = False,
    cutoff_freq: float = 1/32,
    grayscale: bool = False,
    normalize: bool = False,
    norm_mode: str = 'median_iqr',
    per_channel: bool = False
):
    x = np.asarray(tex, dtype=np.float32)
    assert x.ndim in {3, 4} and x.shape[-1] == 3

    spatial_ndim = x.ndim - 1
    spatial_axes = tuple(range(spatial_ndim))
    assert spatial_ndim in {2, 3}

    if do_crop:
        utils.log('Cropping texture')
        if spatial_ndim == 3:
            x = crop_volume(x, crop_size)
        else:
            x = crop_image(x, crop_size)

    if do_filter:
        utils.log('Filtering texture')
        x = highpass_filter(x, cutoff=cutoff_freq)
    
    if grayscale:
        import skimage.color
        utils.log('Converting to grayscale')
        x = skimage.color.rgb2gray(x, channel_axis=-1)

    if normalize:
        utils.log('Normalizing texture')
        norm_axes = spatial_axes if per_channel else None
        x = normalize_texture(x, mode=norm_mode, axes=norm_axes)

    return x


def crop_image(img: np.ndarray, size: int):
    X, Y, C = img.shape
    x0 = (X - size) // 2
    y0 = (Y - size) // 2
    return img[
        x0 : x0 + size,
        y0 : y0 + size,
    ]


def crop_volume(vol: np.ndarray, size: int):
    X, Y, Z, C = vol.shape
    x0 = (X - size) // 2
    y0 = (Y - size) // 2
    z0 = (Z - size) // 2
    return vol[
        x0 : x0 + size,
        y0 : y0 + size,
        z0 : z0 + size,
    ]


def frequency_coords(shape, spacing, axis=-1):
    assert len(shape) == len(spacing)
    ks = (np.fft.fftfreq(n, d) for n, d in zip(shape, spacing))
    ks = np.meshgrid(*ks, indexing='ij')
    return np.stack(ks, axis=axis)


def highpass_filter(tex, cutoff, spacing=None, eps=1e-12):
    x = np.asarray(tex, dtype=np.float64)
    assert x.ndim in {3, 4} and x.shape[-1] == 3

    spatial_ndim = x.ndim - 1
    spatial_axes = tuple(range(spatial_ndim))
    spatial_shape = x.shape[:spatial_ndim]

    if spacing is None:
        spacing = (1,) * spatial_ndim

    mean = x.mean()
    mean_color = x.mean(axis=spatial_axes, keepdims=True)
    F = np.fft.fftn(x - mean_color, axes=spatial_axes)

    ks = frequency_coords(spatial_shape, spacing, axis=0)
    k2 = (ks * ks).sum(axis=0)

    H = 1.0 - np.exp(-0.5 * k2 / max(cutoff * cutoff, eps))
    F = F * H[...,None]

    y = np.fft.ifftn(F, axes=spatial_axes).real
    return y + mean_color - mean


def normalize_texture(tex, mode, axes=None, eps=1e-6):
    assert mode in {
        'mean',
        'median',
        'mean_std',
        'median_iqr',
        'min_max',
        None
    }
    x = np.asarray(tex)

    if mode is None:
        return x

    elif mode == 'mean':
        loc = np.mean(x, axis=axes, keepdims=True)
        scale = 1

    elif mode == 'median':
        loc = np.median(x, axis=axes, keepdims=True)
        scale = 1

    elif mode == 'mean_std':
        loc = np.mean(x, axis=axes, keepdims=True)
        scale = np.std(x, axis=axes, keepdims=True)

    elif mode == 'median_iqr':
        p25, p50, p75 = np.percentile(x, [25, 50, 75], axis=axes, keepdims=True)
        loc, scale = p50, (p75 - p25)

    elif mode == 'min_max':
        a = np.min(x, axis=axes, keepdims=True)
        b = np.max(x, axis=axes, keepdims=True)
        loc, scale  = (a + b) / 2, (b - a) / 2

    return (x - loc) / np.maximum(scale, eps)


def describe_texture(tex: np.ndarray):
    x = np.asarray(tex)

    utils.log(f'shape: {x.shape}')
    utils.log(f'dtype: {x.dtype}')

    x_min, x_max = x.min(), x.max()
    utils.log(f'range: [{x_min:.4f}, {x_max:.4f}]')

    x_mean, x_std = x.mean(), x.std()
    utils.log(f'mean (std):   {x_mean:.4f} ({x_std:.4f})')

    p25, p50, p75 = np.percentile(x, [25, 50, 75])
    utils.log(f'median (IQR): {p50:.4f} ({p75 - p25:.4f})')


# ----- texture vizualization -----


def plot_histogram(tex, bins, alpha=0.2):
    from ..visual import matplotlib as mpl_viz
    import seaborn as sns

    x = np.asarray(tex)
    assert x.ndim in {3, 4}, x.shape

    if x.shape[-1] == 3:
        x = x.reshape(-1, 3)
    else:
        x = x.reshape(-1, 1)

    N, C = x.shape

    fig, axes = mpl_viz.subplot_grid(1, 1, 2, 8, padding=(1.0, 0.5, 0.5, 0.5))
    ax = axes[0,0]

    sns.histplot(
        x,
        stat='count',
        bins=bins,
        palette=list('rgb') if C == 3 else None,
        alpha=alpha,
        ax=ax
    )

    # box plot of intensity values
    yscale = ax.get_yscale()
    yticks = ax.get_yticks()
    yticklabels = ax.get_yticklabels()

    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin
    wbox = 0.1 * yrange
    ybox = ymin - wbox

    ax.boxplot(
        x.flatten(),
        orientation='horizontal',
        positions=[ybox],
        widths=[wbox],
        showfliers=False
    )
    ax.set_yscale(yscale)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
   
    return fig, axes


def power_spectrum(
    tex,
    spacing=(1, 1, 1),
    center=True,
    window=True,
    shift=True
):
    x = np.asarray(tex, dtype=np.float64)
    assert x.ndim == 4 and x.shape[-1] == 3

    if center:
        x = x - x.mean()

    if window:
        w_x = np.hanning(x.shape[0])
        w_y = np.hanning(x.shape[1])
        w_z = np.hanning(x.shape[2])
        w = w_x[:,None,None] * w_y[None,:,None] * w_z[None,None,:]
        if x.ndim == 4:
            x = x * w[...,None]
        else:
            x = x * w

    F = np.fft.fftn(x, axes=(0,1,2))
    P = np.abs(F)**2

    # spatial frequency coordinates
    k = frequency_coords(x.shape[:3], spacing, axis=-1)

    if shift:
        P = np.fft.fftshift(P, axes=(0,1,2))
        k = np.fft.fftshift(k, axes=(0,1,2))

    return P, k


def plot_psd(P, k, bins=None, alpha=0.2, xscale='linear', yscale='linear'):
    from ..visual import matplotlib as mpl_viz
    import pandas as pd
    import seaborn as sns

    P = np.asarray(P)
    assert P.ndim in {3, 4} and P.shape[-1] == 3, P.shape
    spatial_ndim = P.ndim - 1

    k = np.asarray(k)
    assert k.shape == P.shape[:spatial_ndim], k.shape

    df = pd.DataFrame({
        'P': P.reshape(-1),
        'k': np.repeat(k, 3),
        'c': np.tile(np.array(list('rgb')), k.size)
    })

    fig, axes = mpl_viz.subplot_grid(1, 1, 2, 8, padding=(1.0, 0.5, 0.5, 0.5))
    ax = axes[0,0]

    sns.histplot(
        data=df,
        x='k',
        hue='c',
        weights='P',
        bins=list(bins) if hasattr(bins, '__iter__') else bins,
        stat='count',
        palette=list('rgb'),
        alpha=alpha,
        ax=ax
    )
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_ylabel('power')



# DEPRECATED


def sample_field(df, tex_type, points, rng):
    sel = (df.texture_class == tex_type)
    if not sel.any():
        raise RuntimeError(f'No textures for material name {mat_name!r}')
    return sample_texture_field(
        df[sel],
        points=points,
        rng=rng,
        iqr_mult=self.iqr_mult,
        use_solid=self.use_solid,
        use_color=self.use_color,
        weights=self.weights
    )


def sample_texture_field(
    mat_tex_df,
    points,
    rng,
    iqr_mult=4.0,
    use_solid=True,
    use_color=True,
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

