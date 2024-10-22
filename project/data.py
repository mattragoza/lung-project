import torch
import nibabel as nib
import fenics as fe
from mpi4py import MPI

from . import imaging
from . import utils


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
        mesh_radius = example['mesh_radius']
        
        # load images from NIFTI files
        anat = load_nii_file(example['anat_file'])
        disp = load_nii_file(example['disp_file'])
        mask = load_nii_file(example['mask_file'])        
        
        # get image spatial resolution
        resolution = anat.header.get_zooms()

        # load mesh from xdmf file
        mesh = load_mesh_file(example['mesh_file'])

        # convert arrays to tensors with shape (c,x,y,z)
        kwargs = dict(dtype=self.dtype, device=self.device)
        anat = torch.as_tensor(anat.get_fdata(), **kwargs).unsqueeze(0)
        disp = torch.as_tensor(disp.get_fdata(), **kwargs).permute(3,0,1,2)
        mask = torch.as_tensor(mask.get_fdata(), **kwargs).unsqueeze(0)

        if 'elast_file' in example: # has ground truth
            elast = load_nii_file(example['elast_file'])
            elast = torch.as_tensor(elast.get_fdata(), **kwargs).unsqueeze(0)
        else:
            elast = torch.zeros_like(anat)

        return anat, elast, disp, mask, resolution, mesh, mesh_radius, example_name


def load_nii_file(nii_file):
    print(f'Loading {nii_file}... ', end='')
    nifti = nib.load(nii_file)
    print(nifti.header.get_data_shape())
    return nifti


def load_mesh_file(mesh_file):
    print(f'Loading {mesh_file}... ', end='')
    mesh = fe.Mesh()
    with fe.XDMFFile(MPI.COMM_WORLD, str(mesh_file)) as f:
        f.read(mesh)
    print(mesh.num_vertices())
    return mesh
