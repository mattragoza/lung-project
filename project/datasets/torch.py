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
        normalize: bool = False,
        image_mean: float = 0.0,
        image_std:  float = 1.0,
        apply_mask: bool = False,
        do_augment: bool = False,
        rand_rotate:  bool = False,
        rand_reflect: bool = False,
        sigma_trans: float = 0.0,
        use_cache:  bool = False,
        n_mat_labels: int = 5,
        rgb: bool=False
    ):
        self.examples = examples

        # transform parameters
        self.normalize  = normalize
        self.image_mean = image_mean
        self.image_std  = image_std
        self.apply_mask = apply_mask

        # data augmentation settings
        self.do_augment   = do_augment
        self.rand_rotate  = rand_rotate
        self.rand_reflect = rand_reflect
        self.sigma_trans  = sigma_trans

        # expected data shapes
        self.n_mat_labels = n_mat_labels
        self.rgb = rgb

        self.use_cache = use_cache
        self._cache = {}

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        if self.use_cache:
            if idx not in self._cache:
                self._cache[idx] = self.load_example(ex)
            out = self._cache[idx]
        else:
            out = self.load_example(ex)
        if self.do_augment:
            out = self.augment_sample(out)
        return self.add_derived_keys(out)

    def clear_cache(self):
        self._cache.clear()

    def load_example(self, ex):
        image = fileio.load_nibabel(ex.paths['input_image'])
        matrl = fileio.load_nibabel(ex.paths['material_mask'])
        mesh  = fileio.load_meshio(ex.paths['interp_mesh'])

        image_a = image.get_fdata()
        if self.rgb:
            assert image_a.ndim == 4 and image_a.shape[-1] == 3
        else:
            assert image_a.ndim == 3
            image_a = image_a[...,None] # add channel dim

        matrl_a = matrl.get_fdata()
        assert matrl_a.ndim == 3

        def _as_cpu_tensor(a, dtype):
            return torch.as_tensor(a, dtype=dtype, device='cpu')

        affine_t = _as_cpu_tensor(image.affine, dtype=torch.float)
        image_t  = _as_cpu_tensor(image_a, dtype=torch.float).permute(3,0,1,2)
        matrl_t  = _as_cpu_tensor(matrl_a, dtype=torch.long).unsqueeze(0)
        mask_t   = (matrl_t > 0)

        if self.normalize:
            image_t = (image_t - self.image_mean) / self.image_std

        if self.apply_mask:
            image_t = image_t * mask_t

        sample = {
            'example':    ex,
            'affine':     affine_t,
            'img_true':   image_t, 
            'mat_true':   matrl_t,
            'mask':       mask_t,
            'mesh':       mesh
        }

        if 'elastic_field' in ex.paths:
            elast_a = fileio.load_nibabel(ex.paths['elastic_field']).get_fdata()
            elast_t = _as_cpu_tensor(elast_a, dtype=torch.float).unsqueeze(0)
            sample['E_true'] = elast_t

        return sample

    def augment_sample(self, sample, rng=None):
        return apply_data_augmentation(
            sample,
            do_rotate=self.rand_rotate,
            do_reflect=self.rand_reflect,
            sigma_trans=self.sigma_trans,
            rng=rng
        )

    def add_derived_keys(self, sample, eps=1e-12):
        sample = sample.copy()

        mat_label = sample['mat_true'][0].long() # (1, I, J, K) -> (I, J, K)
        mat_onehot = F.one_hot(mat_label, self.n_mat_labels + 1) # (C, I, J, K)
        sample['mat_onehot'] = mat_onehot.permute(3,0,1,2).float()

        if 'E_true' in sample:
            sample['logE_true'] = torch.log10(sample['E_true'].clamp_min(eps))

        return sample


@torch.no_grad()
def apply_data_augmentation(
    sample: Dict[str, Any],
    do_rotate: bool=False,
    do_reflect: bool=False,
    sigma_trans: float=0.0, # in voxels
    device: str='cuda',
    rng=None
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

    sample['img_true'] = resample_volume(sample['img_true'], mode='linear')
    sample['mat_true'] = resample_volume(sample['mat_true'], mode='nearest')
    sample['mask'] = resample_volume(sample['mask'], mode='nearest')
    if 'E_true' in sample:
        sample['E_true'] = resample_volume(sample['E_true'], mode='linear')

    # update affine to reflect new voxel -> world mapping
    sample['affine'] = B

    return sample


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

