from typing import List, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F

from .base import Example
from ..core import utils, fileio


class TorchDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        examples: List[Example],
        normalize=False,
        image_mean=0.0,
        image_std=1.0,
        apply_mask=False,
        use_cache=False
    ):
        self.examples = examples

        # transform parameters
        self.normalize  = normalize
        self.image_mean = image_mean
        self.image_std  = image_std
        self.apply_mask = apply_mask

        self.use_cache = use_cache
        self._cache = {}

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        if self.use_cache:
            key = id(ex)
            if key not in self._cache:
                self._cache[key] = self.load_example(ex)
            return self._cache[key]

        return self.load_example(ex)

    def clear_cache(self):
        self._cache.clear()

    def load_example(self, ex):
        image    = fileio.load_nibabel(ex.paths['input_image'])
        material = fileio.load_nibabel(ex.paths['material_mask'])
        mesh     = fileio.load_meshio(ex.paths['interp_mesh'])

        def _as_cpu_tensor(a, dtype):
            return torch.as_tensor(a, dtype=dtype, device='cpu')

        affine_t   = _as_cpu_tensor(image.affine, dtype=torch.float) # (4, 4)
        image_t    = _as_cpu_tensor(image.get_fdata(), dtype=torch.float).unsqueeze(0)   # (1, I, J, K)
        material_t = _as_cpu_tensor(material.get_fdata(), dtype=torch.long).unsqueeze(0) # (1, I, J, K)

        mask_t = (material_t > 0)

        # (1, I, J, K) -> (I, J, K) -> (I, J, K, C) -> (C, I, J, K)
        mat_onehot_t = F.one_hot(material_t.squeeze(0), num_classes=6).permute(3,0,1,2)

        if self.normalize:
            image_t = (image_t - self.image_mean) / self.image_std

        if self.apply_mask:
            image_t = image_t * mask_t

        output = {
            'example':    ex,
            'affine':     affine_t,
            'img_true':   image_t, 
            'mat_true':   material_t,
            'mat_onehot': mat_onehot_t,
            'mask':       mask_t,
            'mesh':       mesh
        }

        if 'elastic_field' in ex.paths:
            elast = fileio.load_nibabel(ex.paths['elastic_field'])
            elast_t = _as_cpu_tensor(elast.get_fdata(), dtype=torch.float).unsqueeze(0) # (1, I, J, K)
            log_e_t = torch.log10(elast_t.clamp_min(1e-12))
            output['E_true']    = elast_t
            output['logE_true'] = log_e_t

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

