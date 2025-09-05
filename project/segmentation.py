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
LUNGS_LABEL = 1
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


def create_lung_regions_mask(
    lungs_mask,
    airways_mask,
    vessels_mask,
    lungs_kws,
    airways_kws,
    vessels_kws,
    verbose=True
):
    print('Filtering connected components: lungs')
    lungs_mask = filter_connected_components(
        lungs_mask, verbose=verbose, **lungs_kws
    )

    print('Filtering connected components: airways')
    airways_mask = filter_connected_components(
        airways_mask, verbose=verbose, **airways_kws
    )

    print('Filtering connected components: vessels')
    vessels_mask = filter_connected_components(
        vessels_mask, verbose=verbose, **vessels_kws
    )

    print('Combining into lung regions mask')
    regions_mask = np.zeros(lungs_mask.shape, dtype=np.uint16)
    regions_mask[lungs_mask] = LUNGS_LABEL
    regions_mask[airways_mask] = AIRWAYS_LABEL
    regions_mask[vessels_mask] = VESSELS_LABEL

    for i in [1, 2, 3]:
        num_components = count_connected_components(regions_mask > 0, connectivity=i)
        print(f'Mask has {num_components} components ({i}-connectivity)')

    return regions_mask


def filter_connected_components(
    mask,
    min_count=30,
    min_percent=0,
    keep_largest=True,
    connectivity=None,
    max_components=None,
    verbose=True
):
    # label connected regions in the mask and measure their size
    labeled_mask, num_labels = skimage.measure.label(
        mask, background=0, return_num=True, connectivity=connectivity
    )
    labels, voxel_counts = np.unique(labeled_mask, return_counts=True)
    if verbose:
        print(f'  Input mask has {num_labels} components')
    
    total_voxels = voxel_counts[1:].sum() # exclude background
    filtered_mask = np.zeros_like(mask, dtype=bool)

    total_dropped = 0
    total_components = 0
    for idx in np.argsort(-voxel_counts[1:]):
        label = labels[idx+1]
        count = voxel_counts[idx+1]
        percent = count / total_voxels * 100
        if verbose > 1:
            print(f'    Component {label} has {count} voxels ({percent:.4f}%): ', end='')
        if count >= min_count and percent >= min_percent and not (max_components and total_components >= max_components):
            filtered_mask[labeled_mask == label] = True
            if verbose > 1:
                print('keep')
            total_components += 1
        else:
            total_dropped += count
            if verbose > 1:
                print('drop')

    if keep_largest and filtered_mask.sum() == 0 and num_labels > 0:
        idx = np.argmax(voxel_counts[1:])
        label = labels[idx+1]
        count = voxel_counts[idx+1]
        precent = count / total_voxels * 100
        filtered_mask = (labeled_mask == label)
        total_dropped -= count

    if verbose:
        percent = total_dropped / total_voxels * 100
        print(f'    {total_dropped} voxels were dropped ({percent:.4f}%)')

        num_final = count_connected_components(filtered_mask, connectivity)
        print(f'  Output mask has {num_final} components')

    return filtered_mask


def count_connected_components(mask, connectivity=None):
    return skimage.measure.label(
        mask, background=0, return_num=True, connectivity=connectivity
    )[1]


