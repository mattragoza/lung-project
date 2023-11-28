data_name=Emory-4DCT
case_id=1
phase=50

data_root=data/$data_name

case_name=Case${case_id}Pack
case_dir=$data_root/$case_name

image_dir=$case_dir/NIFTI
mask_dir=$case_dir/TotalSegment

image_name=case${case_id}_T${phase}

image_file=$image_dir/$image_name.nii.gz
output_dir=$mask_dir/$image_name

TotalSegmentator -i $image_file -o $output_dir -ta total --roi_subset lung_upper_lobe_right lung_upper_lobe_left lung_middle_lobe_right lung_lower_lobe_right lung_lower_lobe_left
TotalSegmentator -i $image_file -o $output_dir -ta lung_vessels
totalseg_combine_masks -i $output_dir -o $output_dir/lung_combined_mask.nii.gz -m lung

echo Done
