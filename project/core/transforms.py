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


def get_grid_bounds(shape, affine, unit_m, align_corners=True):
    '''
    Return the upper and lower bounding corners of a grid
    in world coordinates (in *meters*, NOT world units) using
    the provided world affine matrix, 3D shape, and world unit.
    '''
    origin  = get_affine_origin(affine)
    spacing = get_affine_spacing(affine)

    if torch.is_tensor(affine):
        shape = torch.as_tensor(shape)
    else:
        shape = np.asarray(shape)

    if align_corners:
        lo = origin
        hi = origin + (shape - 1.0) * spacing
    else:
        lo = origin - 0.5 * spacing
        hi = origin + (shape - 0.5) * spacing
    
    return lo * unit_m, hi * unit_m


def compute_cell_volume(verts, cells):
    a = verts[cells[:,0]]
    b = verts[cells[:,1]]
    c = verts[cells[:,2]]
    d = verts[cells[:,3]]
    M = np.stack([b - a, c - a, d - a], axis=-1) # (M,3,3)
    return np.abs(np.linalg.det(M)) / 6


def compute_node_adjacency(verts, cells, volume):
    inds, vals = [], []
    for c, vert_inds in enumerate(cells):
        for v in vert_inds:
            inds.append([int(v), int(c)])
            vals.append(float(volume[c]))

    inds = np.array(inds).T
    shape = len(verts), len(cells)
    if torch.is_tensor(verts):
        return torch.sparse_coo_tensor(inds, vals, shape)
    else:
        import scipy.sparse
        return scipy.sparse.coo_array((vals, inds), shape)


def node_to_cell_values(cells, node_vals):
    assert cells.ndim == 2 and cells.shape[-1] == 4, cells.shape
    assert node_vals.ndim in {1, 2}, node_vals.shape
    return node_vals[cells].mean(axis=1)


def cell_to_node_values(cells_to_nodes, cell_vals, eps=1e-12):
    assert cell_vals.ndim in {1, 2}

    if cell_vals.ndim == 1:
        cell_vals = cell_vals[:,None]
        do_squeeze = True
    else:
        do_squeeze = False

    num = cells_to_nodes @ cell_vals # (N, C) x (C, D) -> (N, D)
    den = cells_to_nodes.sum(axis=1) # (N, C) -> (N,)
    out = num / np.maximum(den, eps)[:,None]

    return out[:,0] if do_squeeze else out


def cell_to_node_labels(verts, cells, cell_labels):
    vol = compute_cell_volume(verts, cells)
    num_verts = len(verts)
    max_label = int(cell_labels.max())
    acc = np.zeros((num_verts, max_label + 1), dtype=float)
    for i, vert_inds in enumerate(cells):
        for j in vert_inds:
            acc[j, cell_labels[i]] += vol[i]
    return np.argmax(acc, axis=1)


def smooth_mesh_values(verts, cells, node_vals, cell_vals, degree):
    assert degree in {0, 1}
    if degree == 0:
        out_vals = (cell_vals + node_to_cell_values(cells, node_vals)) / 2
    elif degree == 1:
        out_vals = (node_vals + cell_to_node_values(verts, cells, cell_vals)) / 2
    return out_vals


def compute_lame_parameters(E, nu=0.4):
    assert 0 < nu < 0.5, nu
    mu  = E / (2*(1 + nu))
    lam = E * nu / ((1 + nu)*(1 - 2*nu))
    return mu, lam


def compute_youngs_modulus(mu, nu=0.4):
    assert 0 < nu < 0.5, nu
    return 2*(1 + nu)*mu


def compute_density_from_CT(ct, m_atten_ratio=1., density_water=1000.):

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


def compute_emphysema_from_CT(ct, threshold=-950):
    return (ct <= threshold)


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

