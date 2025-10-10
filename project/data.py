from functools import lru_cache
import numpy as np
import torch
import nibabel as nib

from . import imaging, utils, meshing

from . import pde as pde_module


class Dataset(torch.utils.data.Dataset):

    def __init__(
        self,
        examples,
        dtype=torch.float32,
        device='cpu',
        image_scale=[1,1,1],
        cache_size=None
    ):
        super().__init__()

        self.examples = examples
        self.dtype = dtype
        self.device = device
        self.image_scale = image_scale
        self.cache_size = cache_size

        self.get_example = lru_cache(cache_size)(self.load_example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.get_example(idx)
    
    def load_example(self, idx):
        example = self.examples[idx]    
        name = example['name']
        
        # load images from NIFTI files
        a_image = load_nii_file(example['anat_file'])
        u_image = load_nii_file(example['disp_file'])
        mask = load_nii_file(example['mask_file'])  

        # get image spatial resolution
        resolution = a_image.header.get_zooms() # tuple
        
        # convert arrays to tensors with shape (c,x,y,z)
        kwargs = dict(dtype=self.dtype, device=self.device)
        a_image = torch.as_tensor(a_image.get_fdata(), **kwargs).unsqueeze(0)
        u_image = torch.as_tensor(u_image.get_fdata(), **kwargs).permute(3,0,1,2)
        mask = torch.as_tensor(mask.get_fdata(), **kwargs).unsqueeze(0)

        # load mesh from xdmf file
        mesh, cell_labels = meshing.load_mesh_fenics(
            example['mesh_file'], example['has_labels']
        )

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

            dis_mask = torch.cat([mask1, mask2, mask3], dim=0)
        else:
            zeros = torch.zeros_like(mask)
            dis_mask = torch.cat([zeros, zeros, zeros], dim=0)

        # downsample if necessary
        x_scale, y_scale, z_scale = self.image_scale
        if x_scale > 1 or y_scale > 1 or z_scale > 1:
            a_image = a_image[:,::x_scale,::y_scale,::z_scale]
            e_image = e_image[:,::x_scale,::y_scale,::z_scale]
            u_image = u_image[:,::x_scale,::y_scale,::z_scale]
            mask = mask[:,::x_scale,::y_scale,::z_scale]
            dis_mask = dis_mask[:,::x_scale,::y_scale,::z_scale]
            resolution = tuple(r*s for r,s in zip(resolution, self.image_scale))

        # initialize biomechanical model
        pde = pde_module.FiniteElementModel(mesh, resolution, cell_labels)

        return (
            a_image, e_image, u_image, mask, dis_mask, resolution, pde, name
        )


def load_nii_file(nii_file, verbose=False):
    if verbose:
        print(f'Loading {nii_file}... ', end='')
    nifti = nib.load(nii_file)
    if verbose:
        print(nifti.header.get_data_shape())
    return nifti
