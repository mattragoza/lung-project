from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any, Dict, List, Tuple, Iterable
from itertools import permutations

import nibabel as nib
import numpy as np
import torch
import meshio


@dataclass
class Example:

    dataset:  str
    subject:  str
    variant:  Optional[str] = None
    visit:    Optional[str] = None
    fixed_state:  Optional[str] = None
    moving_state: Optional[str] = None
    paths:    Dict[str, Path] = None
    metadata: Dict[str, Any]  = None


class BaseDataset:

    def subjects(self) -> List[str]:
        raise NotImplementedError

    def visits(self, subject: str) -> List[str]:
        raise NotImplementedError

    def variants(self, subject: str, visit: str) -> List[str]:
        raise NotImplementedError

    def states(self, subject: str, visit: str) -> List[str]:
        raise NotImplementedError

    def state_pairs(self, subject: str, visit: str) -> List[Tuple[str, str]]:
        states = self.states(subject, visit)
        return list(permutations(states, 2))

    def get_path(
        self,
        subject: str,
        variant: str,
        visit: str,
        state: str,
        asset_type: str,
        **selectors
    ) -> Path:
        raise NotImplementedError

    def examples(self, *args, **kwargs) -> Iterable[Example]:
        raise NotImplementedError


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

        image_nifti = nib.load(ex.paths['fixed_image'])
        mask_nifti  = nib.load(ex.paths['fixed_mask'])
        disp_nifti  = nib.load(ex.paths['disp_field'])

        affine = np.array(image_nifti.affine)

        def as_tensor(a):
            return torch.as_tensor(a, dtype=self.dtype, device=self.device)

        image = as_tensor(image_nifti.get_fdata()).unsqueeze(0)
        mask  = as_tensor(mask_nifti.get_fdata()).unsqueeze(0)
        disp  = as_tensor(disp_nifti.get_fdata()).permute(3,0,1,2)

        elast = None
        if 'elast_field' in ex.paths:
            elast_nifti = load_nifti(ex.paths['elast_field'])
            elast = as_tensor(elast_nifti).unsqueeze(0)

        mesh = meshio.read(ex.paths['fixed_mesh'])

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

