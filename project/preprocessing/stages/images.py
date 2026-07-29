# preprocessing/stages/images.py

from typing import Tuple
from pathlib import Path
import numpy as np

from ...core import utils, fileio


def _interpret_axcodes(codes):
    cx, cy, cz = codes.upper()
    assert cx in 'LR', cx
    assert cy in 'PA', cy
    assert cz in 'IS', cz
    return  (
        1 if cx == 'R' else -1,
        1 if cy == 'A' else -1,
        1 if cz == 'S' else -1
    )


def convert_image_to_nifti(
    input_path: Path,
    output_path: Path,
    shape: Tuple[int, int, int],
    dtype: str,
    axcodes: str,
    spacing: Tuple[float, float, float],
    slope: float = 1.0,
    intercept: float = 0.0
):
    signs = _interpret_axcodes(axcodes)

    array = fileio.load_binary_image(input_path, shape, dtype)
    array = array.astype(np.float32) * slope + intercept

    affine = np.diag([
        signs[0] * spacing[0],
        signs[1] * spacing[1],
        signs[2] * spacing[2],
        1.0
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_nibabel(output_path, array, affine)


def resample_image_spacing(input_path, output_path, ref_path, config):
    utils.check_keys(
        config,
        valid={'spacing', 'interpolation', 'default_value'},
        where='image_resampling'
    )
    from .. import image_resampling

    src_image = fileio.load_simpleitk(input_path)
    ref_image = fileio.load_simpleitk(ref_path)

    utils.log('Resampling image using reference grid')
    output_image = image_resampling.resample_image(src_image, ref_image, **config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_simpleitk(output_path, output_image)

