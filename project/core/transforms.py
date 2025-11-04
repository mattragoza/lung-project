import numpy as np
import torch


def _as_tensor(a, b):
    return torch.as_tensor(a, dtype=b.dtype, device=b.device)


def _homogeneous(points):
    '''
    Args:
        points: (..., D) input coords
    Returns:
        (..., D+1) homogeneous coords
    '''
    if isinstance(points, torch.Tensor):
        ones = torch.ones_like(points[...,:1])
        return torch.cat([points, ones], dim=-1)
    else:
        points = np.asarray(points)
        ones = np.ones_like(points[...,:1])
        return np.concatenate([points, ones], axis=-1)


def build_affine_matrix(origin, spacing):
    ox, oy, oz = origin
    sx, sy, sz = spacing
    return np.array([
        [sx, 0., 0., ox],
        [0., sy, 0., oy],
        [0., 0., sz, oz],
        [0., 0., 0., 1.],
    ], dtype=float)


def get_affine_origin(affine):
    return affine[:3,3]


def get_affine_spacing(affine):
    return np.linalg.norm(affine[:3,:3], axis=0)


def voxel_to_world_coords(points, affine):
    '''
    Args:
        points: (N, 3) voxel coordinates
        affine: (4, 4) voxel -> world mapping
    Returns:
        (N, 3) world coordinates
    '''
    assert affine.shape == (4, 4)
    if isinstance(points, torch.Tensor):
        A = _as_tensor(affine, points)
    else:
        A = np.asarray(affine)
    output = (A @ _homogeneous(points).T).T
    return output[:,:3] / output[:,3:4]


def world_to_voxel_coords(points, affine):
    '''
    Args:
        points: (N, 3) world coordinates
        affine: (4, 4) voxel -> world mapping
    Returns:
        (N, 3) voxel coordinates
    '''
    assert affine.shape == (4, 4)
    if isinstance(points, torch.Tensor):
        A = _as_tensor(affine, points)
        H = _homogeneous(points)
        output = torch.linalg.solve(A, H.T).T
    else:
        A = np.asarray(affine)
        H = _homogeneous(points)
        output = np.linalg.solve(A, H.T).T
    return output[:,:3] / output[:,3:4]


def normalize_voxel_coords(points, shape, align_corners=True, flip_order=False):
    '''
    Args:
        points: (N, D) voxel coordinates
        shape: length D tuple of ints
    Returns:
        (N, D) coords normalized to [-1, 1]
    '''
    if isinstance(points, torch.Tensor):
        S = _as_tensor(shape, points)
    else:
        points = np.asarray(points)
        S = np.asarray(shape)

    if align_corners:
        output = (points / (S - 1)) * 2. - 1.
    else:
        output = ((points + 0.5) / S) * 2. - 1.

    if isinstance(points, torch.Tensor) and flip_order:
        return output.flip(-1)

    elif flip_order:
        return output[...,::-1]

    return output


def get_grid_bounds(origin, spacing, shape, align_corners=True):
    
    origin  = np.asarray(origin)
    spacing = np.asarray(spacing)
    shape   = np.asarray(shape)

    if align_corners:
        lo = origin
        hi = origin + (shape - 1.0) * spacing
    else:
        lo = origin - 0.5 * spacing
        hi = origin + (shape - 0.5) * spacing
    
    return lo, hi


def compute_lame_parameters(E, nu=0.4):
    assert 0 < nu < 0.5, nu
    mu  = E / (2*(1 + nu))
    lam = E * nu / ((1 + nu)*(1 - 2*nu))
    return mu, lam


def compute_youngs_modulus(mu, nu=0.4):
    assert 0 < nu < 0.5, nu
    return 2*(1 + nu)*mu


def parameterize_youngs_modulus(theta_global, theta_local):
    return torch.pow(10, theta_global + theta_local - theta_local.mean())


def compute_density_from_ct(ct, m_atten_ratio=1., density_water=1000.):

    # HU = 1000 (mu_x - mu_water) / mu_water
    # HU / 1000 = mu_x / mu_water - 1
    # mu_x = (HU / 1000 + 1) * mu_water

    # m_atten_x = mu_x / rho_x
    # rho_x = mu_x / m_atten_x
    # mu_x = rho_x * m_atten_x

    # rho_x = (HU / 1000 + 1) * mu_water / m_atten_x
    # rho_x = (HU / 1000 + 1) * rho_water * m_atten_water / m_atten_x
    # rho_x = (HU / 1000 + 1) * rho_water / m_atten_ratio

    # where m_atten_ratio = m_atten_x / m_atten_water
    assert m_atten_ratio > 0
    return (ct / 1000 + 1) * density_water / m_atten_ratio


def compute_emphysema_from_ct(ct, threshold=-950):
    return (ct <= threshold)

