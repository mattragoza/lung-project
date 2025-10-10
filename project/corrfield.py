import sys, os
import numpy as np
import nibabel as nib
import torch

import corrfield


def create_deformation_field(
	visit, variant, source_name, target_name, mask_name='lung_combined_mask'
):
	source_path = visit.image_file(variant, source_name)
	source_nifti = nib.load(source_path)
	source_array = source_nifti.get_fdata()
	print(source_array.shape)

	target_path = visit.image_file(variant, target_name)
	target_nifti = nib.load(target_path)
	target_array = target_nifti.get_fdata()
	print(target_array.shape)

	mask_path = visit.mask_file(variant, target_name, mask_name)
	mask_nifti = nib.load(mask_path)
	mask_array = mask_nifti.get_fdata()

	disp_path = visit.disp_file(variant, source_name, target_name)
	disp = corrfield.corrfield.corrfield(
		img_mov=torch.as_tensor(source_array, dtype=torch.float)[None,None,...].cuda(),
		img_fix=torch.as_tensor(target_array, dtype=torch.float)[None,None,...].cuda(),
		mask_fix=torch.as_tensor(mask_array, dtype=torch.float)[None,None,...].cuda()
	)[0][0].detach().cpu().numpy()

	

	disp_path.parent.mkdir(exist_ok=True, parents=True)
	disp_nifti = nib.nifti1.Nifti1Image(disp, target_nifti.affine)
	nib.save(disp_nifti, disp_path)

