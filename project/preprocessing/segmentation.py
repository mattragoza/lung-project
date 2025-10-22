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
    input_mask,
    min_count=10,
    min_percent=0,
    max_components=None,
    keep_largest=True,
    connectivity=1,
    verbose=True
):
    # label connected regions and measure their size
    labeled, input_components = skimage.measure.label(
        (input_mask != 0), connectivity=connectivity, return_num=True
    )
    if verbose:
        print(f'Input {connectivity}-connected components: {input_components}')

    labels, counts = np.unique(labeled[labeled > 0], return_counts=True)

    total = counts.sum()
    if total == 0:
        return np.zeros_like(input_mask, dtype=bool)

    percents = counts / total * 100.
    size_order = np.argsort(-counts) # largest to smallest

    if verbose:
        print(f'  Voxel counts:   {counts[size_order]} {total}')
        #print(f'  Voxel percents: {percents[size_order]}')

    output_labels = []
    output_components = 0
    voxels_dropped = 0

    for rank, i in enumerate(size_order):
        l, c, p = int(labels[i]), int(counts[i]), float(percents[i])

        size_ok = (c >= min_count) and (p >= min_percent)
        hit_cap = max_components and (output_components >= max_components)
        keep = (size_ok and not hit_cap) or (keep_largest and rank == 0)

        if keep:
            output_labels.append(l)
            output_components += 1
        else:
            voxels_dropped += c

    output_mask = np.isin(labeled, output_labels)

    if verbose:
        pct_dropped = voxels_dropped / total * 100.
        print(f'Output {connectivity}-connected components: {output_components}')
        print(f'  Voxels dropped: {voxels_dropped} ({pct_dropped:.4f}%)')

    return output_mask


def count_connected_components(mask, connectivity=None):
    return skimage.measure.label(
        mask,
        background=0,
        return_num=True,
        connectivity=connectivity
    )[1]


