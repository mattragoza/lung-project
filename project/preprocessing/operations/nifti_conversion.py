# preprocessing/nifti_conversion.py

from typing import Tuple

import nibabel as nib
import numpy as np

from ...common import utils, transforms


def convert_array_to_nifti(
    array: np.ndarray,
    spacing: Tuple[float, float, float],
    axcodes: str,
    slope: float = 1.0,
    intercept: float = 0.0
):
    affine = build_nifti_affine(spacing, axcodes)
    array = array.astype(np.float32) * slope + intercept
    return nib.nifti1.Nifti1Image(array, affine)


def build_nifti_affine(
    spacing: Tuple[float, float, float],
    axcodes: str
) -> np.ndarray:

    if len(spacing) != 3:
        raise ValueError(spacing)

    signs = interpret_axcodes(axcodes)

    return np.diag([
        signs[0] * spacing[0],
        signs[1] * spacing[1],
        signs[2] * spacing[2],
        1.0
    ])



def interpret_axcodes(axcodes: str) -> Tuple[int, int, int]:
    cx, cy, cz = axcodes.upper()

    if cx not in 'LR': raise ValueError(cx)
    if cy not in 'PA': raise ValueError(cy)
    if cz not in 'IS': raise ValueError(cz)

    return  (
        1 if cx == 'R' else -1,
        1 if cy == 'A' else -1,
        1 if cz == 'S' else -1,
    )


def convert_binvox_to_nifti(binvox, points, **preprocess_kws):
    from . import mask_processing

    affine = infer_binvox_affine(binvox, points)
    mask, affine = mask_processing.preprocess_binary_mask(
        binvox.numpy(), affine, **preprocess_kws,
    )
    return nib.nifti1.Nifti1Image(mask.astype(np.uint8), affine)


def infer_binvox_affine(binvox, points):

    shape = np.asarray(binvox.dims)
    translate = np.asarray(binvox.translate)
    scale = float(binvox.scale)

    bbox_min, bbox_extent = transforms.compute_bbox(points)

    sign  = _infer_binvox_sign(translate, bbox_min)
    power = _infer_binvox_power(scale, bbox_extent.max())

    utils.log((sign, power))

    spacing = scale ** power / shape
    origin = translate * sign + spacing / 2
    affine = transforms.to_affine_matrix(origin, spacing)

    utils.log(affine)
    
    return affine


def _infer_binvox_sign(a, b, tol=1e-3):
    e_pos = _relative_error(a, +b)
    e_neg = _relative_error(a, -b)
    if min(e_pos, e_neg) < tol:
        return -1 if e_neg < e_pos else +1


def _infer_binvox_power(a, b, tol=1e-3):
    e_eq  = _relative_error(a, b)
    e_inv = _relative_error(a * b, 1)
    if min(e_eq, e_inv) < tol:
        return -1 if e_inv < e_eq else +1


def _relative_error(a, b, eps=1e-12):
    from numpy.linalg import norm
    return norm(a - b) / (norm(b) + eps)

