import numpy as np
import torch
import nibabel as nib
import fenics as fe
from mpi4py import MPI

from . import imaging, utils, meshing
from . import pde as pde_module


class Dataset(torch.utils.data.Dataset):

    def __init__(self, examples, dtype=torch.float32, device='cpu'):
        super().__init__()

        self.examples = examples
        self.dtype = dtype
        self.device = device

        self.cache = [None] * len(examples)

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        if self.cache[idx] is None:
            self.cache[idx] = self.load_example(idx)
        return self.cache[idx]
    
    def load_example(self, idx):
        example = self.examples[idx]    
        example_name = example['name']
        
        # load images from NIFTI files
        a_image = load_nii_file(example['anat_file'])
        u_image = load_nii_file(example['disp_file'])
        mask = load_nii_file(example['mask_file'])  

        # get image spatial resolution
        resolution = a_image.header.get_zooms()
        
        # convert arrays to tensors with shape (c,x,y,z)
        kwargs = dict(dtype=self.dtype, device=self.device)
        a_image = torch.as_tensor(a_image.get_fdata(), **kwargs).unsqueeze(0)
        u_image = torch.as_tensor(u_image.get_fdata(), **kwargs).permute(3,0,1,2)
        mask = torch.as_tensor(mask.get_fdata(), **kwargs).unsqueeze(0)

        # load mesh from xdmf file
        mesh, cell_labels = meshing.load_mesh_fenics(example['mesh_file'])

        # initialize biomechanical model
        pde = pde_module.FiniteElementModel(mesh, resolution, cell_labels)

        if 'elast_file' in example: # has ground truth
            e_image = load_nii_file(example['elast_file'])
            e_image = torch.as_tensor(e_image.get_fdata(), **kwargs).unsqueeze(0)
        else:
            e_image = torch.zeros_like(a_image)

        if 'mask_file1' in example: # has disease masks
            mask1 = load_nii_file(example['mask_file1'])
            mask1 = torch.as_tensor(mask1.get_fdata(), **kwargs).unsqueeze(0)

            mask2 = load_nii_file(example['mask_file2'])
            mask2 = torch.as_tensor(mask2.get_fdata(), **kwargs).unsqueeze(0)

            mask3 = load_nii_file(example['mask_file3'])
            mask3 = torch.as_tensor(mask3.get_fdata(), **kwargs).unsqueeze(0)

            disease_mask = torch.cat([mask1, mask2, mask3], dim=0)
        else:
            zeros = torch.zeros_like(mask)
            disease_mask = torch.cat([zeros, zeros, zeros], dim=0)

        return a_image, e_image, u_image, mask, disease_mask, resolution, pde, example_name


def load_nii_file(nii_file):
    print(f'Loading {nii_file}... ', end='')
    nifti = nib.load(nii_file)
    print(nifti.header.get_data_shape())
    return nifti
