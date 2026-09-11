# preprocessing/image_resampling.py

from typing import Tuple

import numpy as np
import SimpleITK as sitk


def _resolve_interpolator(name: str):
    if name == 'bspline':
        return sitk.sitkBSpline
    elif name == 'linear':
        return sitk.sitkLinear
    elif name == 'nearest':
        return sitk.sitkNearestNeighbor
    raise ValueError(f'Invalid interpolator: {name!r}')


def resample_image_spacing(
    src_image: sitk.Image,
    ref_image: sitk.Image,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    interpolator: str = 'linear',
    default_value: float = 0.0
):
    '''
    Resample the source image to a standardized voxel spacing
    while preserving the physical FOV of the reference image.
    '''
    old_size    = np.array(ref_image.GetSize(),    dtype=float)
    old_spacing = np.array(ref_image.GetSpacing(), dtype=float)
    old_origin  = np.array(ref_image.GetOrigin(),  dtype=float)

    new_spacing = np.array(spacing, dtype=float)
    new_size    = np.round((old_size - 1.0) * (old_spacing / new_spacing)).astype(int) + 1
    new_size    = np.maximum(new_size, 1)

    # anchor physical FOV at center instead of origin
    D = np.array(ref_image.GetDirection(), dtype=float).reshape(3, 3)
    center = old_origin + D @ (old_spacing * (old_size - 1) / 2)
    new_origin = center - D @ (new_spacing * (new_size - 1) / 2)

    # ref: https://simpleitk.org/doxygen/latest/html/namespaceitk_1_1simple.html
    return sitk.Resample(
        image1=src_image,
        size=tuple(int(d) for d in new_size),
        transform=sitk.Transform(),
        interpolator=_resolve_interpolator(interpolator),
        outputOrigin=tuple(new_origin),
        outputSpacing=tuple(new_spacing),
        outputDirection=ref_image.GetDirection(),
        defaultPixelValue=float(default_value),
        outputPixelType=src_image.GetPixelID()
    )

