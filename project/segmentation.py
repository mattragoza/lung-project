import numpy as np
import nibabel as nib
import skimage

TOTAL_TASK_ROIS = [
    'lung_upper_lobe_right',
    'lung_middle_lobe_right',
    'lung_lower_lobe_right',
    'lung_upper_lobe_left',
    'lung_lower_lobe_left',
]
VESSELS_TASK_ROIS = [
    'lung_trachea_bronchia',
    'lung_vessels'
]
ALL_TASK_ROIS = TOTAL_TASK_ROIS + VESSELS_TASK_ROIS

BACKGROUND_LABEL = 0
LOBE_UPPER_RIGHT_LABEL = 1
LOBE_MIDDLE_RIGHT_LABEL = 2
LOBE_LOWER_RIGHT_LABEL = 3
LOBE_UPPER_LEFT_LABEL = 4
LOBE_LOWER_LEFT_LABEL = 5
AIRWAYS_LABEL = 6
VESSELS_LABEL = 7


def run_segmentation_tasks(visit, variant, image_name):
	from totalsegmentator.python_api import totalsegmentator
	from totalsegmentator.libs import combine_masks

	image_path = visit.image_file(variant, image_name)
	mask_path = visit.mask_file(variant, image_name, mask_name='lung_combined_mask')
	mask_dir = visit.mask_dir(variant, image_name)

	mask_dir.mkdir(parents=True, exist_ok=True)

	print(f'Running total segmentation task on {image_path}')
	totalsegmentator(
		input=image_path,
		output=mask_dir,
		task='total',
		roi_subset=TOTAL_TASK_ROIS
	)

	print(f'Running lung_vessels segmentation task on {image_path}')
	totalsegmentator(
		input=image_path,
		output=mask_dir,
		task='lung_vessels'
	)

	print('Combining segmentation masks')
	combined_mask = combine_masks(
		mask_dir=mask_dir,
		class_type='lung'
	)
	nib.save(combined_mask, mask_path)

	print(f'Combined mask saved to {mask_path}')


def filter_connected_components(
    mask,
    min_count=10,
    min_percent=0,
    keep_largest=True,
    connectivity=None,
    max_components=None,
    verbose=True
):
    assert np.issubdtype(mask.dtype, np.integer), mask.dtype

    # label connected regions and measure their size
    labeled_mask, num_labels = skimage.measure.label(
        mask,
        background=0,
        return_num=True,
        connectivity=connectivity
    )
    labels, counts = np.unique(labeled_mask, return_counts=True)
    if verbose:
        print(f'  # {connectivity}-connected inputs components: {num_labels}')
    
    total_voxels = counts[1:].sum() # exclude background

    filtered_mask = np.zeros_like(mask, dtype=mask.dtype)
    total_components = 0
    total_dropped = 0

    for idx in np.argsort(-counts[1:]):
        label = labels[idx+1]
        count = counts[idx+1]
        percent = count / total_voxels * 100
        if verbose > 1:
            print(f'    Component {label} has {count} voxels ({percent:.4f}%): ', end='')
        component_check = (max_components and total_components >= max_components)
        if count >= min_count and percent >= min_percent and not component_check:
            filtered_mask[labeled_mask == label] = 1
            total_components += 1
        else:
            total_dropped += count

    if keep_largest and filtered_mask.sum() == 0 and num_labels > 0:
        idx = np.argmax(counts[1:])
        label = labels[idx+1]
        count = counts[idx+1]
        percent = count / total_voxels * 100
        filtered_mask = (labeled_mask == label)
        total_dropped -= count

    if verbose:
        percent = total_dropped / total_voxels * 100
        num_final = count_connected_components(filtered_mask, connectivity)
        print(f'    {total_dropped} voxels were dropped ({percent:.4f}%)')
        print(f'  # {connectivity}-connected output components: {num_final}')

    return filtered_mask


def count_connected_components(mask, connectivity=None):
    return skimage.measure.label(
        mask,
        background=0,
        return_num=True,
        connectivity=connectivity
    )[1]


