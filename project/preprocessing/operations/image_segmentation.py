# preprocessing/operations/image_segmentation.py

from typing import Any, Dict, Optional

from pathlib import Path
import numpy as np

from ...common import fileio, utils


VALID_METHODS = ['totalsegmentator', 'visionfeature', 'hu_threshold']
DEFAULT_METHOD = 'totalsegmentator'
DEFAULT_TS_TASK = 'total'

TS_LABELS_BY_TASK = {
    'total': [
        'lung_upper_lobe_right',
        'lung_middle_lobe_right',
        'lung_lower_lobe_right',
        'lung_upper_lobe_left',
        'lung_lower_lobe_left',
    ],
    'lung_vessels': [
        'lung_airways',
        'lung_airways_wall',
        'lung_arteries',
        'lung_veins',
    ],
    'lung_vessels_LEGACY': ['lung_trachea_bronchia', 'lung_vessels'],
    'body': ['body', 'body_trunc', 'body_extremeties', 'skin'],
    'lung_nodules': ['lung', 'lung_nodules'],
}

VF_LABELS = [
    'nodule',
    'ggo',
    'consolidation',
    'emphysema',
    'honeycombing',
    'pleural_effusion',
]


def run_segmentation_task(
    image_path: Path,
    output_dir: Path,
    method: str = DEFAULT_METHOD,
    kwargs: Optional[Dict[str, Any]] = None
):
    key = method.lower()
    kwargs = kwargs or {}

    if key == 'totalsegmentator':
        return run_totalsegmentator_task(image_path, output_dir, **kwargs)

    elif key == 'visionfeature':
        return run_visionfeature_segmentation(image_path, output_dir, **kwargs)

    elif key == 'hu_threshold':
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
        input=image_path,
        output=output_dir,
        task=task,
        **kwargs
    )


def run_visionfeature_segmentation(
    image_path: Path,
    output_dir: Path,
    **kwargs
):
    utils.log('Running VisionFeature segmentation')

    # Save and restore nnUNet environment variables
    # VisionFeature sets its own nnUNet environment
    import os

    names = ['nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results']
    saved = {name: os.environ.pop(name, None) for name in names}

    try:
        from VisionFeature import segmentation_api

        return segmentation_api.segment_case(
            image_path=image_path,
            output_dir=output_dir,
            **kwargs
        )

    finally:
        for name, value in saved.items():
            if value is not None:
                os.environ[name] = value


def run_threshold_segmentation(
    image_path: Path,
    output_dir: Path,
    thresholds: Dict[str, Dict[str, Any]]
):
    utils.log('Running threshold-based segmentation')

    nifti = fileio.load_nibabel(image_path)
    image = nifti.get_fdata()
    affine = nifti.affine

    for label, config in thresholds.items():
        utils.check_keys(
            config,
            valid={'value', 'type', 'operator'},
            where=f'thresholds[{label}]'
        )
        value = float(config['value']) # required

        type_ = config.get('type', 'absolute')
        if type_ == 'absolute':
            threshold = value
        elif type_ == 'percentile':
            threshold = np.percentile(image, value)
        else:
            raise ValueError(f'Invalid threshold type: {type_!r}')

        operator = config.get('operator', '<')
        if operator == '<':
            mask = image < threshold
        elif operator == '>':
            mask = image > threshold
        elif operator == '<=':
            mask = image <= threshold
        elif operator == '>=':
            mask = image >= threshold
        else:
            raise ValueError(f'Invalid threshold operator: {operator!r}')

        fileio.save_nibabel(
            output_dir / f'{label}.nii.gz',
            mask.astype(np.int16),
            nifti.affine
        )


def combine_segmentation_masks(mask_dir: Path, class_type: str = 'lung'):
    from totalsegmentator import libs
    return libs.combine_masks(mask_dir, class_type)

