
TOTAL_TASK_ROIS = [
    'lung_upper_lobe_right',
    'lung_middle_lobe_right',
    'lung_lower_lobe_right',
    'lung_upper_lobe_left',
    'lung_lower_lobe_left',
]
VESSEL_TASK_ROIS = [
    'lung_trachea_bronchia',
    'lung_vessels'
]
ALL_TASK_ROIS = TOTAL_TASK_ROIS + VESSEL_TASK_ROIS


BACKGROUND_LABEL = 0
LOBE_UPPER_RIGHT_LABEL = 1
LOBE_MIDDLE_RIGHT_LABEL = 2
LOBE_LOWER_RIGHT_LABEL = 3
LOBE_UPPER_LEFT_LABEL = 4
LOBE_LOWER_LEFT_LABEL = 5
AIRWAYS_LABEL = 6
VESSELS_LABEL = 7


def run_total_segmentator(image_path, output_dir, roi_subset=None):
	from totalsegmentator import python_api
	roi_subset = roi_subset or TOTAL_TASK_ROIS
	return python_api.totalsegmentator(image_path, output_dir, task='total', roi_subset=roi_subset)


def run_vessel_segmentation(image_path, output_dir):
	from totalsegmentator import python_api
	return python_api.totalsegmentator(image_path, output_dir, task='lung_vessels')


def combine_segmentation_masks(mask_dir, class_type='lung'):
	from totalsegmentator import libs
	return libs.combine_masks(mask_dir, class_type)

