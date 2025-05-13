# Lung biomechanical deep learning project

## Summary of procedure

1. Setup conda environment
2. Download and preprocess images
4. Create image segmentation masks
5. Create image registration fields
6. Train deep learning model

## Conda environment

Run the following to create the conda environment and register it as jupyter notebook kernel:

```bash
mamba env create --file=environment.yml
mamba activate lung-project
python -m ipykernel install --user --name=lung-project
```

## Emory 4DCT image set

### Download the images

1. Go to [download page](https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/downloads-and-reference-data/4dct.html)
2. Submit [access request form](https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/access-request-form.html)
	- Landing page contains dropbox password
3. For each case packet i=1..10,
	- Follow the download link to DropBox
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

### Convert images to NIFTI format

We need to convert the Emory 4DCT images to NIFTI format before running segmentation and registration steps. To do so, run the jupyter notebook `notebooks/Emory-4DCT-preprocessing.ipynb`.
## TotalSegmentator

### Install

```bash
#pip install TotalSegmentator
TotalSegmentator -i $input_image -o $output_dir --device gpu --preview --statistics -ta total --roi_subset lung_upper_lobe_right lung_upper_lobe_left lung_middle_lobe_right lung_lower_lobe_right lung_lower_lobe_left
TotalSegmentator -i $input_image -o $output_dir --device gpu --preview --statistics -ta lung_vessels
totalseg_combine_masks -i $output_dir -o $output_dir/lung_combined_mask.nii.gz -m lung
```

### Usage

TODO

## CorrField registration

### Install

```bash
git clone git@github.com/multimodallearning/Lung250M-4B.git
cd Lung250M-4B/corrfield
python corrfield.py -F {fixed_image} -M {moving_image} -m {fixed_mask} -o {output_path}
```

### Usage

TODO

## Train deep learning model

TODO


