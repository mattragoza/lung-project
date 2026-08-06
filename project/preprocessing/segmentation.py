# preprocessing/segmentation.py

from typing import List, Dict, Any, Optional
from pathlib import Path

from ..core import utils

VALID_METHODS = ['totalsegmentator', 'visionfeature']
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
    'lung_vessels_LEGACY': [
        'lung_trachea_bronchia',
        'lung_vessels',
    ],
    'body': [
        'body',
        'body_trunc',
        'body_extremeties',
        'skin'
    ],
    'lung_nodules': [
        'lung',
        'lung_nodules'
    ]
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
    image_path: Path,
    output_dir: Path,
    config: Dict[str, Any]
):
    utils.check_keys(
        config,
        valid={'method', 'kwargs'},
        where='image_segmentation.task'
    )
    method = config.get('method', DEFAULT_METHOD).lower()
    kwargs = config.get('kwargs', {})

    if method == 'totalsegmentator':
        from totalsegmentator import python_api as ts_api
        task = kwargs.pop('task', DEFAULT_TS_TASK)
        utils.log(f'Running TotalSegmentator task: {task}')

        return ts_api.totalsegmentator(
            input=image_path,
            output=output_dir,
            task=task,
            **kwargs
        )

    elif method == 'visionfeature':
        import os
        os.environ.pop('nnUNet_raw')
        os.environ.pop('nnUNet_preprocessed')
        os.environ.pop('nnUNet_results')

        from VisionFeature import segmentation_api as vf_api
        utils.log('Running VisionFeature segmentation')
    
        return vf_api.segment_case(
            image_path=image_path,
            output_dir=output_dir,
            **kwargs
        )

    raise ValueError(f'Invalid segmentation method: {method!r}')


def combine_segmentation_masks(mask_dir: Path, class_type: str = 'lung'):
    from totalsegmentator import libs
    return libs.combine_masks(mask_dir, class_type)

