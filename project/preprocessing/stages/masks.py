# preprocessing/stages/masks.py

from typing import List, Dict, Tuple, Any
from pathlib import Path
import numpy as np

from ...core import utils, fileio


def create_segmentation_masks(
    image_path: Path,
    segment_dir: Path, # individual masks for each class
    output_path: Path, # combined mask across all classes
    config: Dict[str, Any]
):
    utils.check_keys(
        config,
        valid={'tasks'},
        where='image_segmentation'
    )
    from .. import segmentation

    utils.log('Running segmentation tasks')
    segment_dir.mkdir(parents=True, exist_ok=True)

    for task_config in config.get('tasks', []):
        segmentation.run_segmentation_task(
            image_path=image_path,
            output_dir=segment_dir,
            config=task_config
        )

    utils.log('Combining segmentation masks')
    nifti = segmentation.combine_segmentation_masks(segment_dir, class_type='lung')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_nibabel(output_path, nifti.get_fdata(), nifti.affine)


def convert_binvox_mask(
    mask_path: Path,
    mesh_path: Path,
    output_path: Path,
    config: Dict[str, Any]
):
    utils.check_keys(
        config,
        valid={'foreground_filter', 'background_filter', 'center_mask', 'pad_amount'},
        where='binary_mask'
    )
    from .. import binvox_affine, mask_cleanup

    mesh   = fileio.load_meshio(mesh_path)
    binvox = fileio.load_binvox(mask_path)
    affine = binvox_affine.infer_binvox_affine(binvox, mesh.points)

    foreground_kws = config.get('foreground_filter', {})
    background_kws = config.get('background_filter', {})
    mask = mask_cleanup.filter_binary_mask(binvox.numpy(), foreground_kws, background_kws)

    center = config.get('center_mask')
    if center:
        mask, affine = mask_cleanup.center_array_and_affine(mask, affine)

    pad = config.get('pad_amount', 0)
    if pad > 0:
        mask, affine = mask_cleanup.pad_array_and_affine(mask, affine, pad)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_nibabel(output_path, mask.astype(np.uint8), affine)

