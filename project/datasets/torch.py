from typing import List, Dict, Any
import numpy as np
import torch

from .base import Example
from ..core import utils, fileio


class TorchDataset(torch.utils.data.Dataset):

    def __init__(self, examples: List[Example], dtype=torch.float32):
        self.examples = examples
        self.dtype = dtype
        self._cache = {}

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if idx not in self._cache:
            self._cache[idx] = self.load_example(idx)        
        return self._cache[idx]

    def load_example(self, idx):
        ex = self.examples[idx]

        image = fileio.load_nibabel(ex.paths['input_image'])
        mask  = fileio.load_nibabel(ex.paths['material_mask'])
        mesh  = fileio.load_meshio(ex.paths['image_mesh'])

        def _as_cpu_tensor(a, dtype=None):
            return torch.as_tensor(a, dtype=dtype or self.dtype, device='cpu')

        output = {
            'example': ex,
            'affine': _as_cpu_tensor(image.affine), # (4, 4)
            'image':  _as_cpu_tensor(image.get_fdata()).unsqueeze(0), # (1, I, J, K)
            'mask':   _as_cpu_tensor(mask.get_fdata(), dtype=torch.int).unsqueeze(0),
            'mesh':   mesh,
        }
        if 'elastic_field' in ex.paths:
            elast = fileio.load_nibabel(ex.paths['elastic_field'])
            output['elast'] = _as_cpu_tensor(elast.get_fdata()).unsqueeze(0)

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

