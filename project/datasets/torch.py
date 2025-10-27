from typing import List, Dict, Any
import numpy as np
import torch

from ..core import fileio
from ..base import Example


class TorchDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        examples: List[Example],
        dtype=torch.float32,
        device='cuda'
    ):
        self.examples = examples
        self.dtype = dtype
        self.device = device

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        image_nifti = fileio.load_nibabel(ex.paths['input_image'])
        mask_nifti  = fileio.load_nibabel(ex.paths['region_mask'])
        disp_nifti  = fileio.load_nibabel(ex.paths['disp_field'])

        affine = np.array(image_nifti.affine)

        def _as_tensor(a):
            return torch.as_tensor(a, dtype=self.dtype, device=self.device)

        image = _as_tensor(image_nifti.get_fdata()).unsqueeze(0)
        mask  = _as_tensor(mask_nifti.get_fdata()).unsqueeze(0)
        disp  = _as_tensor(disp_nifti.get_fdata()).permute(3,0,1,2)

        elast = None
        if 'elast_field' in ex.paths:
            elast_nifti = fileio.load_nibabel(ex.paths['elast_field'])
            elast = _as_tensor(elast_nifti).unsqueeze(0)

        mesh = fileio.load_meshio(ex.paths['volume_mesh'])

        return {
            'affine':  affine,
            'image':   image,
            'mask':    mask,
            'disp':    disp,
            'elast':   elast,
            'mesh':    mesh,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    output = {}
    for key in batch[0].keys():
        output[key] = _stack_or_list([ex[key] for ex in batch])
    return output


def _stack_or_list(items: List[Any]) -> torch.Tensor | List[Any]:
    if all(torch.is_tensor(x) for x in items):
        return torch.stack(items, dim=0)
    return list(items)

