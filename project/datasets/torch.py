from typing import List, Dict, Any
import numpy as np
import torch

from .base import Example
from ..core import utils, fileio


class TorchDataset(torch.utils.data.Dataset):

    def __init__(self, examples: List[Example], dtype=torch.float32):
        self.examples = examples
        self.dtype = dtype
        self.cache = {}

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]

        ex = self.examples[idx]
        image = fileio.load_nibabel(ex.paths['input_image'])
        mask  = fileio.load_nibabel(ex.paths['material_mask'])
        mesh  = fileio.load_meshio(ex.paths['sim_fields'])

        def _as_tensor(a):
            return torch.as_tensor(a, dtype=self.dtype, device='cpu')

        output = {
            'example': ex,
            'affine': _as_tensor(image.affine),
            'image':  _as_tensor(image.get_fdata()).unsqueeze(0),
            'mask':   _as_tensor(mask.get_fdata()).unsqueeze(0),
            'mesh':   mesh,
        }
        if 'elast_field' in ex.paths:
            elast = fileio.load_nibabel(ex.paths['elast_field'])
            output['elast'] = _as_tensor(elast.get_fdata()).unsqueeze(0)

        self.cache[idx] = output
        return output


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    output = {}
    for key in batch[0]:
        vals = [ex[key] for ex in batch]
        if all(torch.is_tensor(v) for v in vals):
            output[key] = torch.stack(vals, dim=0)
        elif all(v is None for v in vals):
            output[key] = None
        else:
            output[key] = vals
    return output

