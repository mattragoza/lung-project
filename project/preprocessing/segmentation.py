from typing import List, Optional

TASK_ROIS = {
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


def run_segmentation_task(
    image_path,
    output_dir,
    task_name: str,
    roi_subset: Optional[List[str]] = None
):
	from totalsegmentator import python_api
	return python_api.totalsegmentator(
		image_path, output_dir, task=task_name, roi_subset=roi_subset
	)


def combine_segmentation_masks(mask_dir, class_type: str = 'lung'):
	from totalsegmentator import libs
	return libs.combine_masks(mask_dir, class_type)

