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


def create_lung_regions_mask(
    lungs_mask,
    airways_mask,
    vessels_mask,
    connectivity=1,
    verbose=True
):
    print('Filtering connected components: lungs')
    lungs_mask = filter_connected_components(
        lungs_mask,
        connectivity=connectivity,
        max_components=2,
        verbose=verbose
    )

    print('Filtering connected components: airways')
    airways_mask = filter_connected_components(
        airways_mask,
        connectivity=connectivity,
        verbose=verbose
    )

    print('Filtering connected components: vessels')
    vessels_mask = filter_connected_components(
        vessels_mask,
        connectivity=connectivity,
        verbose=verbose
    )

    print('Combining into lung regions mask')
    regions_mask = np.zeros(lungs_mask.shape, dtype=np.uint16)
    regions_mask[lungs_mask] = LUNGS_LABEL
    regions_mask[airways_mask] = AIRWAYS_LABEL
    regions_mask[vessels_mask] = VESSELS_LABEL

    for i in [1, 2, 3]:
        num_components = count_connected_components(regions_mask > 0, connectivity=i)
        print(f'Output mask has {num_components} {i}-connected component(s)')

    return regions_mask


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

    # label connected regions in the mask and measure their size
    labeled_mask, num_labels = skimage.measure.label(
        mask, background=0, return_num=True, connectivity=connectivity
    )
    labels, voxel_counts = np.unique(labeled_mask, return_counts=True)
    if verbose:
        print(f'  # {connectivity}-connected inputs components: {num_labels}')
    
    total_voxels = voxel_counts[1:].sum() # exclude background
    filtered_mask = np.zeros_like(mask, dtype=mask.dtype)

    total_dropped = 0
    total_components = 0
    for idx in np.argsort(-voxel_counts[1:]):
        label = labels[idx+1]
        count = voxel_counts[idx+1]
        percent = count / total_voxels * 100
        if verbose > 1:
            print(f'    Component {label} has {count} voxels ({percent:.4f}%): ', end='')
        if count >= min_count and percent >= min_percent and not (max_components and total_components >= max_components):
            filtered_mask[labeled_mask == label] = 1
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
        print(f'  # {connectivity}-connected output components: {num_final}')

    return filtered_mask


def count_connected_components(mask, connectivity=None):
    return skimage.measure.label(
        mask, background=0, return_num=True, connectivity=connectivity
    )[1]


def compute_density_map(
    ct_array,
    m_atten_ratio=1.0,
    density_water=1000.0,
    density_air=1.0,
):
    # HU = 1000 (mu_x - mu_water) / mu_water
    # HU / 1000 = mu_x / mu_water - 1
    # mu_x = (HU / 1000 + 1) * mu_water

    # m_atten_x = mu_x / rho_x
    # rho_x = mu_x / m_atten_x
    # mu_x = rho_x * m_atten_x

    # rho_x = (HU / 1000 + 1) * mu_water / m_atten_x
    # rho_x = (HU / 1000 + 1) * rho_water * m_atten_water / m_atten_x
    # rho_x = (HU / 1000 + 1) * rho_water / m_atten_ratio

    # where m_atten_ratio = m_atten_x / m_atten_water
    assert m_atten_ratio > 0
    density_array = (ct_array / 1000 + 1) * density_water / m_atten_ratio
    return np.maximum(density_array, density_air) # kg/m^3

