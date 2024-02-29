data_name=Emory-4DCT
case_id=1
phase=00

data_root=data/$data_name
data_root=$(realpath $data_root)

case_name=Case${case_id}Pack
case_dir=$data_root/$case_name

image_dir=$case_dir/NIFTI
mask_dir=$case_dir/TotalSegment
output_dir=$case_dir/CorrField

mkdir -p $output_dir

fixed_phase=50
fixed_name=case${case_id}_T${fixed_phase}
fixed_image=$image_dir/$fixed_name.nii.gz
fixed_mask=$mask_dir/$fixed_name/lung_combined_mask.nii.gz

moving_phase=00
moving_name=case${case_id}_T${moving_phase}
moving_image=$image_dir/$moving_name.nii.gz

reg_name=case${case_id}_T${moving_phase}_T${fixed_phase}
output_path=$output_dir/$reg_name

cd ../Lung250M-4B
python -m corrfield -F $fixed_image -M $moving_image -m $fixed_mask -O $output_path

echo Done
