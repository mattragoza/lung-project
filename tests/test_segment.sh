data_root=data/COPDGene
subject_id=10009Y
visit_name=Phase-1
image_state=EXP
image_filter=STD
image_code=NJC
image_post=COPD

image_root=${data_root}/Images
subject_dir=${image_root}/${subject_id}
visit_dir=${subject_dir}/${visit_name}

image_dir=${visit_dir}/Resized
mask_dir=${visit_dir}/TotalSegment

image_name=${subject_id}_${image_state}_${image_filter}_${image_code}_${image_post}

image_file=${image_dir}/${image_name}.nii.gz
output_dir=${mask_dir}/${image_name}

TotalSegmentator -i $image_file -o $output_dir -ta total --roi_subset lung_upper_lobe_right lung_upper_lobe_left lung_middle_lobe_right lung_lower_lobe_right lung_lower_lobe_left
TotalSegmentator -i $image_file -o $output_dir -ta lung_vessels
totalseg_combine_masks -i $output_dir -o $output_dir/lung_combined_mask.nii.gz -m lung

echo Done
