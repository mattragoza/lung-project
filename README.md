# Lung biomechanical modeling project

## Emory 4DCT data download

1. Go to [download page](https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/downloads-and-reference-data/4dct.html)
2. Submit [access request form](https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/access-request-form.html)
	- Landing page contains dropbox password
3. For each case packet i=1..10,
	- Follow the download link (DropBox)
	- Enter password and download `Case${i}Pack.zip`
	- Move .zip file to `lung-project/data/download`
4. Unzip case packets into `lung-project/data/Emory-4DCT`
	- Use the commands below:

```bash
cd lung-project
for i in {1..10};
	do unzip data/download/Case${i}Pack.zip -d data/Emory-4DCT;
done
```
## Conda environment setep

Run the following to create the conda environment and register it as jupyter notebook kernel:

```bash
mamba env create --file=environment.yml
mamba activate 4DCT
python -m ipykernel install --user --name=4DCT
```

## TotalSegmentator

```bash
#pip install TotalSegmentator
TotalSegmentator -i $input_image -o $output_dir --device gpu --preview --statistics -ta total --roi_subset lung_upper_lobe_right lung_upper_lobe_left lung_middle_lobe_right lung_lower_lobe_right lung_lower_lobe_left
TotalSegmentator -i $input_image -o $output_dir --device gpu --preview --statistics -ta lung_vessels
totalseg_combine_masks -i $output_dir -o $output_dir/lung_combined_mask.nii.gz -m lung
```

## CorrField

```bash
git clone git@github.com/multimodallearning/Lung250M-4B.git
cd Lung250M-4B/corrfield
python corrfield.py -F {fixed_image} -M {moving_image} -m {fixed_mask} -o {output_path}
```

