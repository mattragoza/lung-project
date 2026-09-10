# common/transforms.py

import numpy as np
import torch


def as_tensor(a, b):
    return torch.as_tensor(a, dtype=b.dtype, device=b.device)


def grid_coords(shape, axis=-1, **kwargs):
    '''
    Args:
        shape: length D tuple of ints
    Returns:
        (..., D) array of grid indices
    '''
    module = torch if 'device' in kwargs else np
    coords = (module.arange(n, **kwargs) for n in shape)
    coords = module.meshgrid(*coords, indexing='ij')
    return module.stack(coords, axis)


# ----- affine / grid geometry -----


def to_homo_coords(points):
    '''
    Args:
        (..., D) coordinate array
    Returns:
        (..., D+1) homogeneous coords
    '''
    module = torch if torch.is_tensor(points) else np
    ones = module.ones_like(points[...,:1])
    return module.concatenate([points, ones], axis=-1)


def compute_bbox(points):
    '''
    Args:
        points: (N, D) array of points
    Returns:
        bbox_min, bbox_extent
    '''
    assert points.ndim == 2
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    return bbox_min, (bbox_max - bbox_min)


def to_affine_matrix(origin, spacing):
    '''
    Args:
        origin:  (x0, y0, z0) point
        spacing: (dx, dy, dz) vector
    Returns:
        (4, 4) voxel -> world matrix
    '''
    ox, oy, oz = origin
    sx, sy, sz = spacing
    return np.array([
        [sx, 0., 0., ox],
        [0., sy, 0., oy],
        [0., 0., sz, oz],
        [0., 0., 0., 1.],
    ], dtype=np.float32)


def get_affine_origin(affine):
    '''
    Args:
        affine: (4, 4) voxel -> world mapping
    Returns:
        (x0, y0, z0) origin point
    '''
    assert affine.shape == (4, 4)
    return affine[:3,3]


def get_affine_spacing(affine):
    '''
    Args:
        affine: (4, 4) voxel -> world mapping
    Returns:
        (dx, dy, dz) spacing vector
    '''
    assert affine.shape == (4, 4)
    module = torch if torch.is_tensor(affine) else np
    return module.linalg.norm(affine[:3,:3], axis=0)


def voxel_to_world_coords(points, affine):
    '''
    Args:
        points: (N, 3) voxel coordinates
        affine: (4, 4) voxel -> world mapping
    Returns:
        (N, 3) world coordinates
    '''
    assert affine.shape == (4, 4)
    if torch.is_tensor(points):
        A = as_tensor(affine, points)
    else:
        A = np.asarray(affine)
    output = (A @ to_homo_coords(points).T).T
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
    if torch.is_tensor(points):
        A = as_tensor(affine, points)
        H = to_homo_coords(points)
        output = torch.linalg.solve(A, H.T).T
    else:
        A = np.asarray(affine)
        H = to_homo_coords(points)
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
    if torch.is_tensor(points):
        S = as_tensor(shape, points)
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


def get_grid_bounds(shape, affine, align_corners=True):
    '''
    Compute the grid bounds in world coordinates.
    '''
    use_torch = torch.is_tensor(affine)

    I, J, K = shape

    if align_corners:
        lo = [0, 0, 0]
        hi = [I - 1, J - 1, K - 1]
    else:
        lo = [-0.5, -0.5, -0.5]
        hi = [I - 0.5, J - 0.5, K - 0.5]

    corners_data = [
        [lo[0], lo[1], lo[2]],
        [lo[0], lo[1], hi[2]],
        [lo[0], hi[1], lo[2]],
        [lo[0], hi[1], hi[2]],
        [hi[0], lo[1], lo[2]],
        [hi[0], lo[1], hi[2]],
        [hi[0], hi[1], lo[2]],
        [hi[0], hi[1], hi[2]],
    ]

    if use_torch:
        corners = torch.tensor(
            corners_data, dtype=affine.dtype, device=affine.device
        )
        ones = torch.ones((8, 1), dtype=affine.dtype, device=affine.device)
    else:
        corners = np.asarray(corners_data, dtype=float)
        ones = np.ones((8, 1), dtype=float)

    if use_torch:
        corners_h = torch.cat([corners, ones], dim=1)
    else:
        corners_h = np.concatenate([corners, ones], axis=1)

    world = (affine @ corners_h.T).T[:, :3]

    if use_torch:
        return world.min(axis=0).values, world.max(axis=0).values
    else:
        return world.min(axis=0), world.max(axis=0)


# ----- tetrahedral mesh geometry -----


def compute_cell_volume(verts, cells):
    '''
    Compute volume of tetrahedral cells.

    Args:
        verts: (N, 3) float array
        cells: (M, 4) int array
    Returns:
        volume: (M,) float array
    '''
    a = verts[cells[:,0]] # (M, 3)
    b = verts[cells[:,1]] # (M, 3)
    c = verts[cells[:,2]] # (M, 3)
    d = verts[cells[:,3]] # (M, 3)

    M = np.stack([
        b - a,
        c - a,
        d - a,
    ], axis=-1) # (M, 3, 3)

    return np.abs(np.linalg.det(M)) / 6


def compute_incidence_matrix(verts, cells):
    '''
    Construct sparse node-to-cell incidence matrix.

    Args:
        verts: (N, 3) float array or tensor
        cells: (M, 4) int array or tensor
    Returns:
        A: (N, M) sparse matrix where A[i,j] = 1
            iff node i is a vertex of cell j.
    '''
    shape = (len(verts), len(cells)) # (N, M)

    if torch.is_tensor(cells):
        node_inds = cells.long().reshape(-1)

        cell_inds = torch.arange(shape[1], device=cells.device)
        cell_inds = cell_inds.repeat_interleave(cells.shape[1])

        indices = torch.stack([node_inds, cell_inds]) # (2, 4M)
        values = torch.ones(
            indices.shape[1], dtype=verts.dtype, device=verts.device
        )

        return torch.sparse_coo_tensor(
            indices, values, shape, device=verts.device
        ).coalesce()

    import scipy.sparse

    node_inds = cells.reshape(-1)
    cell_inds = np.repeat(np.arange(shape[1]), cells.shape[1])

    values = np.ones(len(node_inds), dtype=verts.dtype)

    return scipy.sparse.coo_array(
        (values, (node_inds, cell_inds)), shape=shape
    )


def node_to_cell_values(node_values, incidence):
    '''
    Args:
        node_values: (N, C) or (N,)
        incidence:   (N, M)
    Returns:
        cell_values: (M, C) or (M,)
    '''
    N = incidence.shape[0]

    if torch.is_tensor(node_values):
        ones = torch.ones(N, dtype=node_values.dtype, device=node_values.device)
    else:
        ones = np.ones(N, dtype=node_values.dtype)

    counts = incidence.T @ ones

    if node_values.ndim == 1:
        return (incidence.T @ node_values) / counts

    return (incidence.T @ node_values) / counts[:,None]


def cell_to_node_values(cell_values, cell_volume, incidence):
    '''
    Args:
        cell_values: (M, C) or (M,)
        cell_volume: (M,)
        incidence:   (N, M)
    Returns:
        node_values: (N, C) or (N,)
    '''
    denom = incidence @ cell_volume

    if cell_values.ndim == 1:
        numer = incidence @ (cell_values * cell_volume)
        return numer / denom

    numer = incidence @ (cell_values * cell_volume[:,None])
    return numer / denom[:,None]


# ----- physical parameters -----


def mu_lam_from_E_nu(E, nu):
    mu = E / (2*(1 + nu))
    lam = E * nu / ((1 + nu)*(1 - 2*nu))
    return mu, lam


def E_nu_from_mu_lam(mu, lam):
    E = mu * (3*lam + 2*mu) / (lam + mu)
    nu = lam / (2*(lam + mu))
    return E, nu


def density_from_HU(hu, m_atten_ratio=1.0, rho_water=1000):
    return (hu / 1000 + 1) * rho_water / m_atten_ratio


def emphysema_from_HU(hu, threshold=-950):
    return (hu <= threshold)


# ----- rigid transformations -----


def sample_rigid_transform(
    do_rotate: bool,
    do_reflect: bool,
    sigma_trans: float,
    center: tuple=None,
    rng=None
):
    import scipy.stats as stats
    rng = np.random.default_rng(rng)

    if do_rotate and do_reflect:
        R = stats.ortho_group.rvs(3, random_state=rng)
    elif do_rotate:
        R = stats.special_ortho_group.rvs(3, random_state=rng)
    elif do_reflect:
        R = np.diag(rng.choice([-1, 1], size=3))
    else:
        R = np.eye(3)

    if not np.isclose(sigma_trans, 0):
        t = rng.normal(0, sigma_trans, size=3)
    else:
        t = np.zeros(3)

    R = R.astype(np.float32, copy=False)
    t = t.astype(np.float32, copy=False)

    if center is not None:
        # p' = R @ (p - c) + c + t
        c = np.asarray(center, dtype=np.float32)
        t = t + c - R @ c

    T = np.eye(4, dtype=np.float32)
    T[:3,:3] = R
    T[:3,3] = t
    return T


def apply_rigid_transform(points, transform):
    assert transform.shape == (4, 4)
    if torch.is_tensor(points):
        T = as_tensor(transform, points)
    else:
        T = np.asarray(transform)
    output = to_homo_coords(points) @ T.T
    return output[:,:3] / output[:,3:4]


def random_rigid_transform(points, sigma=0, rng=None):
    import scipy.stats as stats

    points = np.asarray(points)
    assert points.ndim == 2 and points.shape[-1] == 3

    R = stats.special_ortho_group.rvs(3, random_state=rng)
    t = rng.normal(scale=sigma, size=3)

    center = points.mean(axis=0)
    return (points - center) @ R.T + t + center


def gaussian_filter(array, mask, affine, sigma, eps=1e-8):
    import scipy.ndimage as ndi

    array = np.asarray(array, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)

    if sigma <= 0:
        return array * mask

    sigma_v = sigma / get_affine_spacing(affine)

    array_f = ndi.gaussian_filter(array * mask, sigma=sigma_v)
    mask_f = ndi.gaussian_filter(mask, sigma=sigma_v)

    return array_f / np.maximum(mask_f, eps) * mask

