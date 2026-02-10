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
        do_augment=False,
        use_cache=False,
        rgb=False
    ):
        self.examples = examples

        # transform parameters
        self.normalize  = normalize
        self.image_mean = image_mean
        self.image_std  = image_std
        self.apply_mask = apply_mask
        self.do_augment = do_augment

        self.rgb = rgb
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

        if self.rgb:
            assert image.ndim == 4 and image.shape[-1] == 3
        else:
            assert image.ndim == 3
            image = image[...,None]

        def _as_cpu_tensor(a, dtype):
            return torch.as_tensor(a, dtype=dtype, device='cpu')

        affine_t   = _as_cpu_tensor(image.affine, dtype=torch.float) # (4, 4)
        image_t    = _as_cpu_tensor(image.get_fdata(), dtype=torch.float).permute(3,0,1,2) # (C, I, J, K)
        material_t = _as_cpu_tensor(material.get_fdata(), dtype=torch.long).unsqueeze(0) # (1, I, J, K)

        mask_t = (material_t > 0)

        # (1, I, J, K) -> (I, J, K) -> (I, J, K, C) -> (C, I, J, K)
        mat_onehot_t = F.one_hot(material_t.squeeze(0), num_classes=6).permute(3,0,1,2)

        if self.normalize:
            image_t = (image_t - self.image_mean) / self.image_std

        if self.apply_mask:
            image_t = image_t * mask_t

        sample = {
            'example':    ex,
            'affine':     affine_t,
            'img_true':   image_t, 
            'mat_true':   material_t,
            'mat_onehot': mat_onehot_t,
            'mask':       mask_t,
            'mesh':       mesh
        }
        aligned_keys = ['img_true', 'mat_true', 'mat_onehot', 'mask']

        if 'elastic_field' in ex.paths:
            elast = fileio.load_nibabel(ex.paths['elastic_field'])
            elast_t = _as_cpu_tensor(elast.get_fdata(), dtype=torch.float).unsqueeze(0) # (1, I, J, K)
            log_e_t = torch.log10(elast_t.clamp_min(1e-12))
            sample['E_true']    = elast_t
            sample['logE_true'] = log_e_t
            aligned_keys.extend(['E_true', 'logE_true'])

        if self.do_augment:
            sample = augment_sample(sample, aligned_keys, max_shift=8, do_flip=True)

        return sample


def augment_sample(sample, aligned_keys, max_shift=0, do_flip=False):
    import random
    sample = sample.copy()

    if max_shift > 0:
        shift = (
            random.randint(-max_shift, max_shift + 1),
            random.randint(-max_shift, max_shift + 1),
            random.randint(-max_shift, max_shift + 1)
        )
    else:
        shift = (0, 0, 0)

    if do_flip:
        flip = (
            random.randint(0, 1),
            random.randint(0, 1),
            random.randint(0, 1)
        )
    else:
        flip = (0, 0, 0)

    for k in aligned_keys:
        t = sample[k]

        if shift != (0, 0, 0):
            t = translate(t, shift)

        if flip != (0, 0, 0):
            flip_dims = tuple(np.nonzero(flip)[0] + 1)
            t = torch.flip(t, dims=flip_dims)

        sample[k] = t

    return sample


def translate(src, shift, pad_value=0):
    C, X, Y, Z = src.shape
    dx, dy, dz = shift
    dst = torch.full(src.shape, pad_value, dtype=src.dtype)

    x0_src = max(0, -dx); x1_src = min(X, X - dx)
    y0_src = max(0, -dy); y1_src = min(Y, Y - dy)
    z0_src = max(0, -dz); z1_src = min(Z, Z - dz)

    x0_dst = max(0, dx); x1_dst = x0_dst + (x1_src - x0_src)
    y0_dst = max(0, dy); y1_dst = y0_dst + (y1_src - y0_src)
    z0_dst = max(0, dz); z1_dst = z0_dst + (z1_src - z0_src)

    dst[:,x0_dst:x1_dst,y0_dst:y1_dst,z0_dst:z1_dst] = \
        src[:,x0_src:x1_src,y0_src:y1_src,z0_src:z1_src]

    return dst


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


@torch.no_grad()
def apply_random_transform(
    inputs,
    rotate: bool=False,
    mirror: bool=False,
    translate: float=0,
    crop_size: int=196,
    device='cuda',
    rng=None
):
    from ..core import transforms, interpolation
    ex     = inputs['example']
    image  = inputs['image'].to(device)  # (1, I, J, K)
    mask   = inputs['mask'].to(device)   # (1, I, J, K)
    elast  = inputs.get('elast')
    elast  = elast.to(device) if elast is not None else None
    affine = inputs['affine'].to(device) # (4, 4), world coords
    mesh   = inputs['mesh'] # world coords

    _, I, J, K = image.shape

    if crop_size is not None: # get center crop indices
        assert 0 < crop_size <= min(I, J, K)
        i0 = (I - crop_size) // 2
        j0 = (J - crop_size) // 2
        k0 = (K - crop_size) // 2
        i1 = i0 + crop_size
        j1 = j0 + crop_size
        k1 = k0 + crop_size
    else:
        i0, i1 = (0, I)
        j0, j1 = (0, J)
        k0, k1 = (0, K)

    # construct voxel grid (possibly center-cropped)
    ii, jj, kk = torch.meshgrid(
        torch.arange(i0, i1, dtype=torch.float, device=device),
        torch.arange(j0, j1, dtype=torch.float, device=device),
        torch.arange(k0, k1, dtype=torch.float, device=device),
        indexing='ij'
    )
    pts_voxel = torch.stack([ii, jj, kk], dim=-1) # (I, J, K, 3)
    pts_voxel = pts_voxel.reshape(-1, 3)

    # sample random rigid transform (only t is in voxel coords)
    R, t = sample_rigid_transform(rotate, mirror, translate)
    R = torch.from_numpy(R).to(dtype=torch.float, device=device)
    t = torch.from_numpy(t).to(dtype=torch.float, device=device)
    t_world = affine[:3,:3] @ t

    # use PRE-crop center for rotation
    ctr_voxel = torch.as_tensor(
        [[(I-1)/2, (J-1)/2, (K-1)/2]], # (1, 3)
        dtype=torch.float, device=device
    )
    ctr_world = transforms.voxel_to_world_coords(ctr_voxel, affine)

    # apply the transform to grid points (in world coords)
    pts_world = transforms.voxel_to_world_coords(pts_voxel, affine)
    pts_w_rot = (pts_world - ctr_world) @ R.T + t_world + ctr_world
    pts_v_rot = transforms.world_to_voxel_coords(pts_w_rot, affine)

    # interpolate 3D image volumes on transformed grid points
    image_out = interpolation.interpolate_image(image, pts_v_rot)
    image_out = image_out.reshape(1, i1 - i0, j1 - j0, k1 - k0)
    if elast is not None:
        elast_out = interpolation.interpolate_image(elast, pts_v_rot)
        elast_out = elast_out.reshape(1, i1 - i0, j1 - j0, k1 - k0)

    mask_out = interpolation.interpolate_image(mask.float(), pts_v_rot, mode='nearest')
    mask_out = mask_out.reshape(1, i1 - i0, j1 - j0, k1 - k0).int()

    # also transform the mesh vertices (already in world coords)
    mesh_out = mesh.copy()
    vts_world = torch.as_tensor(mesh.points, dtype=torch.float, device=device)
    vts_w_rot = (vts_world - ctr_world) @ R.T + t_world + ctr_world
    mesh_out.points = vts_w_rot.detach().cpu().numpy()

    output = {
        'example': ex,
        'affine': affine.cpu(),
        'image': image_out.cpu(),
        'mask':  mask_out.cpu(),
        'mesh':  mesh_out,
    }
    if elast is not None:
        output['elast'] = elast.cpu()

    return output


def sample_rigid_transform(rotate: bool, mirror: bool, translate: float, rng=None):
    from scipy.stats import ortho_group, special_ortho_group
    rng = np.random.default_rng(rng)

    if rotate and mirror:
        R = ortho_group.rvs(3, random_state=rng)
    elif rotate:
        R = special_ortho_group.rvs(3, random_state=rng)
    elif mirror:
        R = np.diag(rng.choice([-1., 1.], size=3))
    else:
        R = np.eye(3)

    t = rng.normal(0., translate, size=3) if translate else np.zeros(3)
    return R, t

