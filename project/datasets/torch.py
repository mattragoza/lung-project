# datasets/torch.py

from typing import List, Dict, Optional, Any

import numpy as np
import torch

from ..common import fileio
from .base import Example


def _cpu_tensor(a, dtype):
    return torch.as_tensor(a, dtype=dtype, device='cpu')


def _load_tensor(path, dtype=torch.float):
    '''Load nifti as (C, I, J, K) cpu tensor.'''
    array = fileio.load_nibabel(path).get_fdata()
    return _cpu_tensor(array, dtype).unsqueeze(0)


class TorchDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        examples: List[Example],
        normalize: bool = False,
        image_mean: float = 0.0,
        image_std:  float = 1.0,
        apply_mask: bool = False,
        do_augment: bool = False,
        rand_rotate:  bool = False,
        rand_reflect: bool = False,
        rand_translate: float = 0.0,
        use_cache: bool = False
    ):
        self.examples = examples

        # image preprocessing
        self.normalize  = normalize
        self.image_mean = image_mean
        self.image_std  = image_std
        self.apply_mask = apply_mask

        # rigid data augmentation
        self.do_augment     = do_augment
        self.rand_rotate    = rand_rotate
        self.rand_reflect   = rand_reflect
        self.rand_translate = rand_translate

        self.use_cache = use_cache
        self._cache    = {}

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        if self.use_cache:
            if idx not in self._cache:
                self._cache[idx] = self.load_example(ex)
            sample = self._cache[idx]
        else:
            sample = self.load_example(ex)

        if self.do_augment:
            sample = apply_data_augmentation(
                sample,
                do_rotate=self.rand_rotate,
                do_reflect=self.rand_reflect,
                sigma_translate=self.rand_translate,
                rng=0
            )

        return sample

    def clear_cache(self):
        self._cache.clear()

    def load_example(self, ex: Example) -> Dict[str, Any]:

        # load required assets
        image = fileio.load_nibabel(ex.paths['input_image'])
        mask = fileio.load_nibabel(ex.paths['domain_mask'])
        mesh = fileio.load_meshio(ex.paths['target_mesh'])

        # ----- shape validation -----

        affine = image.affine
        image = image.get_fdata()
        mask = mask.get_fdata()

        if affine.shape != (4, 4):
            raise ValueError(f'Invalid affine shape: {affine.shape} vs. (4, 4)')

        if image.ndim != 3 or not all(d > 1 for d in image.shape):
            raise ValueError(f'Invalid 3D image shape: {image.shape} vs. (I, J, K)')

        if mask.shape != image.shape:
            raise ValueError(f'Mask shape mismatch: {mask.shape} vs. {image.shape}')

        # ----- conversion to tensors -----

        affine = _cpu_tensor(affine, dtype=torch.float)             # (4, 4)
        image = _cpu_tensor(image, dtype=torch.float).unsqueeze(0)  # (C, I, J, K)
        mask = _cpu_tensor(mask, dtype=torch.int).unsqueeze(0) > 0  # (C, I, J, K)

        # ----- image preprocessing -----

        if self.normalize:
            image = (image - self.image_mean) / self.image_std

        if self.apply_mask:
            image = image * mask

        # ----- output packaging -----

        sample = {
            'example': ex,
            'affine': affine,
            'image': image,
            'mask': mask,
            'mesh': mesh
        }

        # ----- load optional assets -----

        if 'elastic_field' in ex.paths:
            sample['E'] = _load_tensor(ex.paths['elastic_field'])

        if 'poisson_field' in ex.paths:
            sample['nu'] = _load_tensor(ex.paths['poisson_field'])

        if 'density_field' in ex.paths:
            sample['rho'] = _load_tensor(ex.paths['density_field'])

        if 'material_map' in ex.paths:
            sample['material'] = _load_tensor(ex.paths['material_map'], torch.long)

        if 'anatomical_map' in ex.paths:
            sample['anatomy'] = _load_tensor(ex.paths['anatomical_map'], torch.long)

        return sample


def collate_fn(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    output = {}
    for key in samples[0]:
        values = [sample[key] for sample in samples]
        try:
            output[key] = torch.stack(values, dim=0)
        except TypeError:
            output[key] = values
    return output


@torch.no_grad()
def apply_data_augmentation(
    sample: Dict[str, Any],
    do_rotate: bool = False,
    do_reflect: bool = False,
    sigma_trans: float = 0.0, # in voxels
    device: str = 'cuda',
    rng: Optional[int] = None
):
    from ..core import transforms, interpolation

    sample = sample.copy()
    if not (do_rotate or do_reflect) and np.isclose(sigma_trans, 0):
        return sample

    # get voxel grid indices
    mask = sample['mask'][0] # (I, J, K)
    grid_ijk = transforms.grid_coords(mask.shape, device=device, dtype=torch.float)
    grid_ijk = grid_ijk.reshape(-1, 3) # (N, 3)

    # convert voxel grid to world coordinates
    A = sample['affine'].to(device=device, dtype=torch.float) # (4, 4)
    grid_xyz = transforms.voxel_to_world_coords(grid_ijk, A)  # (N, 3)

    # get the object center (world coords) and max voxel size
    center_xyz = grid_xyz[mask.ravel(),:].mean(0).cpu().numpy() # (3,)
    voxel_size = transforms.get_affine_spacing(A).max().item()

    # randomly sample a rigid transformation
    T = torch.as_tensor(transforms.sample_rigid_transform(
        do_rotate=do_rotate,
        do_reflect=do_reflect,
        sigma_trans=sigma_trans * voxel_size,
        center=center_xyz,
        rng=rng
    ), dtype=torch.float, device=device) # (4, 4)

    # apply transformation to voxel grid in world space
    B = T @ A
    grid_xyz_T = transforms.voxel_to_world_coords(grid_ijk, B)
    grid_ijk_T = transforms.world_to_voxel_coords(grid_xyz_T, A)

    # resample volumes on transformed grid
    def resample_volume(t, mode):
        return interpolation.interpolate_image(
            t.to(device, dtype=torch.float),
            points=grid_ijk_T,
            mode=mode,
            reshape=False
        ).reshape(t.shape).to(t.device, dtype=t.dtype)

    sample['image'] = resample_volume(sample['image'], mode='linear')
    sample['mask'] = resample_volume(sample['mask'], mode='nearest')

    if 'mat_label' in sample:
        sample['mat_label'] = resample_volume(sample['mat_label'], mode='nearest')

    for key in ['E', 'nu', 'rho']:
        if key in sample:
            sample[key] = resample_volume(sample[key], mode='linear')

    # update affine to reflect new voxel -> world mapping
    sample['affine'] = B

    return sample


def add_derived_keys(sample, n_mat_labels, eps=1e-12):
    import torch.nn.functional as F
    sample = sample.copy()

    if 'mat_label' in sample:
        mat_label = sample['mat_label'][0].long() # (1, I, J, K) -> (I, J, K)
        mat_onehot = F.one_hot(mat_label, n_mat_labels + 1) # (C, I, J, K)
        sample['mat_onehot'] = mat_onehot.permute(3,0,1,2).float()

    for key in ['E', 'rho']:
        if key in sample:
            sample[f'log{key}'] = torch.log10(sample[key].clamp_min(eps))

    return sample


def accumulate_stats(loader, keys, use_mask=True):
    from collections import defaultdict

    # initialize
    stats = defaultdict(lambda: {'count': 0, 'sum': 0.0, 'sumsq': 0.0, 'min': np.inf, 'max': -np.inf})

    # accumulate moments
    for batch in loader:
        if use_mask:
            mask = (batch['mask'] > 0)
            assert mask.sum() > 0, 'no foreground voxels'
    
        for k in keys:
            if k.startswith('log_'):
                x = torch.log10(batch[k[4:]].float())
            else:
                x = batch[k].float()
            if use_mask:
                x = x[mask]

            stats[k]['count'] += x.numel()
            stats[k]['sum']   += x.sum().item()
            stats[k]['sumsq'] += (x*x).sum().item()
            stats[k]['min'] = min(stats[k]['min'], x.min().item())
            stats[k]['max'] = max(stats[k]['max'], x.max().item())

    # compute derived stats
    for k in keys:
        count = stats[k]['count']
        sum_  = stats[k]['sum']
        sumsq = stats[k]['sumsq']

        if count == 0:
            raise ValueError(f'count is zero for {k}')
            
        mean = sum_ / count
        var = (sumsq / count) - (mean * mean)
        std = float(np.sqrt(var))

        stats[k]['mean'] = mean
        stats[k]['var'] = var
        stats[k]['std'] = std
    
    return stats

