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
VESSEL_TASK_ROIS = [
    'lung_trachea_bronchia',
    'lung_vessels'
]

LUNG_LABEL = 1
AIRWAY_LABEL = 6
VESSEL_LABEL = 7


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

 
def count_connected_components(mask):
    return skimage.measure.label(mask, background=0, return_num=True)[1]


def filter_connected_components(mask, min_count=30, min_percent=0, keep_largest=True, verbose=True, label_name=None):

    # label connected regions in the mask and measure their size
    labeled_mask, n_labels = skimage.measure.label(mask, background=0, return_num=True)
    labels, voxel_counts = np.unique(labeled_mask, return_counts=True)
    print(f'[{label_name}] Mask has {n_labels} connected components')
    
    total_voxels = voxel_counts[1:].sum() # exclude background
    filtered_mask = np.zeros_like(mask, dtype=bool)

    total_dropped = 0
    for idx in np.argsort(-voxel_counts[1:]):
        label = labels[idx+1]
        count = voxel_counts[idx+1]
        percent = count / total_voxels * 100
        if count >= min_count and percent >= min_percent:
            filtered_mask[labeled_mask == label] = True
            print(f'[{label_name}] Component {label} has {count} voxels ({percent:.4f}%)')
        else:
            total_dropped += count

    if keep_largest and filtered_mask.sum() == 0 and n_labels > 0:
        idx = np.argmax(voxel_counts[1:])
        label = labels[idx+1]
        count = voxel_counts[idx+1]
        precent = count / total_voxels * 100
        filtered_mask = (labeled_mask == label)
        total_dropped -= count

    percent = total_dropped / total_voxels * 100
    print(f'[{label_name}] {total_dropped} voxels were dropped ({percent:.4f}%)')

    return filtered_mask


def create_lung_region_mask(visit, variant, image_name, mask_name='lung_regions'):

    lung_file = visit.mask_file(variant, image_name, mask_name='lung_combined_mask')
    print(f'Loading lung mask from {lung_file}')

    lung_nifti = nib.load(lung_file)
    lung_mask = lung_nifti.get_fdata().astype(bool)
    assert lung_mask.sum() > 0

    print('Filtering lung mask')
    lung_mask = filter_connected_components(lung_mask, min_count=5000, min_percent=1, label_name='lung')

    airway_file = visit.mask_file(variant, image_name, mask_name='lung_trachea_bronchia')
    print(f'Loading airway mask from {airway_file}')

    airway_mask = nib.load(airway_file).get_fdata().astype(bool)
    assert airway_mask.sum() > 0

    print('Filtering airway mask')
    airway_mask = filter_connected_components(airway_mask, min_count=100, min_percent=0, label_name='airways')
    
    vessel_file = visit.mask_file(variant, image_name, mask_name='lung_vessels')
    print(f'Loading vessel mask from {vessel_file}')

    vessel_mask = nib.load(vessel_file).get_fdata().astype(bool)
    assert vessel_mask.sum() > 0

    print('Filtering vessel mask')
    vessel_mask = filter_connected_components(vessel_mask, min_count=100, min_percent=1, label_name='vessels')
    vessel_mask &= (lung_mask | airway_mask)

    print(f'Combining lung region masks')
    regions = [
    	lung_mask   * LUNG_LABEL,
    	airway_mask * AIRWAY_LABEL,
    	vessel_mask * VESSEL_LABEL,
    ]
    combined = np.stack(regions).max(axis=0).astype(np.uint16)

    num_components = count_connected_components(combined)
    assert num_components > 0
    if num_components > 1:
        print(f'WARNING: mask has {num_components} components')

    regions_file = visit.mask_file(variant, image_name, mask_name)
    print(f'Saving lung region mask to {regions_file}')

    regions_nifti = nib.nifti1.Nifti1Image(combined, lung_nifti.affine)
    nib.save(regions_nifti, regions_file)

