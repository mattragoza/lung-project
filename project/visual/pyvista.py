import nibabel as nib
import pyvista as pv


def get_opacity_values(vmin, vmax, center, width, low=0.0, high=1.0, n=201):
    x = np.linspace(vmin, vmax, n)
    a = low + (high - low) * _sigmoid((x - center) / width)
    return a.tolist()


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _gaussian(x):
    return np.exp(-x**2)


