from typing import Optional, Any, List, Dict, Tuple, Iterable
import numpy as np
import SimpleITK as sitk


def create_reference_grid(
    image: sitk.Image,
    spacing: Tuple[float, float, float],
    anchor: str='center'
):
    assert anchor in {'origin', 'center'}
    
    old_size = np.array(image.GetSize(), dtype=float)
    old_spacing = np.array(image.GetSpacing(), dtype=float)
    old_origin = np.array(image.GetOrigin(), dtype=float)

    new_spacing = np.array(spacing, dtype=float)
    new_size = np.round((old_size - 1.0) * (old_spacing / new_spacing)).astype(int) + 1
    new_size = np.maximum(new_size, 1)

    if anchor == 'origin':
        new_origin = old_origin

    elif anchor == 'center':
        D = np.array(image.GetDirection(), dtype=float).reshape(3, 3)
        center = old_origin + D @ (old_spacing * (old_size - 1) / 2)
        new_origin = center - D @ (new_spacing * (new_size - 1) / 2)

    grid = sitk.Image(tuple(int(n) for n in new_size), image.GetPixelID())
    grid.SetSpacing(tuple(new_spacing))
    grid.SetOrigin(tuple(new_origin))
    grid.SetDirection(image.GetDirection())
    return grid


def resample_image(
    image: sitk.Image,
    grid:  sitk.Image,
    interp: str='linear',
    default: float=0.
):
    assert interp in {'bspline', 'linear', 'nearest'}

    if interp == 'bspline':
        interp = sitk.sitkBSpline
    elif interp == 'linear':
        interp = sitk.sitkLinear
    elif interp == 'nearest':
        interp = sitk.sitkNearestNeighbor
    
    return sitk.Resample(
        image, grid, sitk.Transform(), interp, default, image.GetPixelID()
    )


