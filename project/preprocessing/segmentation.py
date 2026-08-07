# preprocessing/segmentation.py

from typing import List, Dict, Any, Optional
from pathlib import Path

from ..core import utils, fileio

VALID_METHODS = ['totalsegmentator', 'visionfeature', 'hu_threshold']
DEFAULT_METHOD = 'totalsegmentator'
DEFAULT_TS_TASK = 'total'

TS_LABELS_BY_TASK = {
    'total': [
        'lung_upper_lobe_right',
        'lung_middle_lobe_right',
        'lung_lower_lobe_right',
        'lung_upper_lobe_left',
        'lung_lower_lobe_left'
    ],
    'lung_vessels': [
        'lung_airways',
        'lung_airways_wall',
        'lung_arteries',
        'lung_veins'
    ],
    'lung_vessels_LEGACY': ['lung_trachea_bronchia', 'lung_vessels'],
    'body': ['body', 'body_trunc', 'body_extremeties', 'skin'],
    'lung_nodules': ['lung', 'lung_nodules']
}

VF_LABELS = [
    "nodule",
    "ggo",
    "consolidation",
    "emphysema",
    "honeycombing",
    "pleural_effusion"
]


def run_segmentation_task(
    image_path: Path, output_dir: Path, config: Dict[str, Any]
):
    utils.check_keys(
        config,
        valid={'method', 'kwargs'},
        where='image_segmentation.tasks[]'
    )

    method = config.get('method', DEFAULT_METHOD).lower()
    kwargs = config.get('kwargs', {})

    if method == 'totalsegmentator':
        return run_totalsegmentator_task(image_path, output_dir, **kwargs)

    elif method == 'visionfeature':
        return run_visionfeature_segmentation(image_path, output_dir, **kwargs)

    elif method == 'hu_threshold':
        return run_threshold_segmentation(image_path, output_dir, **kwargs)

    raise ValueError(f'Invalid segmentation method: {method!r}')


def run_totalsegmentator_task(
    image_path: Path,
    output_dir: Path,
    task: str = DEFAULT_TS_TASK,
    **kwargs
):
    utils.log(f'Running TotalSegmentator task: {task!r}')
    from totalsegmentator import python_api

    return python_api.totalsegmentator(
        input=image_path, output=output_dir, task=task, **kwargs
    )


def run_visionfeature_segmentation(
    image_path: Path, output_dir: Path, **kwargs
):
    import os
    utils.log('Running VisionFeature segmentation')

    # save and restore nnUNet environment
    nnunet_raw = os.environ.pop('nnUNet_raw')
    nnunet_pre = os.environ.pop('nnUNet_preprocessed')
    nnunet_res = os.environ.pop('nnUNet_results')

    try:
        from VisionFeature import segmentation_api
        
        return segmentation_api.segment_case(
            image_path=image_path, output_dir=output_dir, **kwargs
        )

    finally:
        if nnunet_raw: os.environ['nnUNet_raw'] = nnunet_raw
        if nnunet_pre: os.environ['nnUNet_preprocessed'] = nnunet_pre
        if nnunet_res: os.environ['nnUNet_results'] = nnunet_res


def run_threshold_segmentation(
    image_path: Path,
    output_dir: Path,
    thresholds: Dict[str, Dict[str, Any]]
):
    utils.log('Running threshold-based segmentation')
    import numpy as np

    nifti = fileio.load_nibabel(image_path)
    image = nifti.get_fdata()

    for label, config in thresholds.items():
        utils.check_keys(
            config,
            valid={'value', 'type', 'operator'},
            where=f'thresholds[{label}]'
        )
        value = float(config['value']) # required

        ttype = config.get('type', 'absolute')
        if ttype == 'absolute':
            threshold = value
        elif ttype == 'percentile':
            threshold = np.percentile(image, value)
        else:
            raise ValueError(f'Invalid threshold type: {ttype!r}')

        operator = config.get('operator', '<')
        if operator == '<':
            mask = (image < threshold)
        elif operator == '>':
            mask = (image > threshold)
        elif operator == '<=':
            mask = (image <= threshold)
        elif operator == '>=':
            mask = (image >= threshold)
        else:
            raise ValueError(f'Invalid threshold operator: {operator!r}')

        mask_path = output_dir / f'{label}.nii.gz'
        fileio.save_nibabel(mask_path, mask.astype(np.int16), nifti.affine)


def combine_segmentation_masks(mask_dir: Path, class_type: str = 'lung'):
    from totalsegmentator import libs
    return libs.combine_masks(mask_dir, class_type)

