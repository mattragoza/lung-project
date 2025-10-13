import numpy as np


def voxel_to_world_coords(points, affine):
    assert points.ndim == 2 and points.shape[1] == 3
    assert affine.shape == (4, 4) and np.allclose(affine[3], [0,0,0,1])
    points_h = np.c_[points, np.ones(points.shape[0])]
    return (affine @ points_h.T).T[:,:3]


def world_to_voxel_coords(points, affine):
    assert points.ndim == 2 and points.shape[1] == 3
    assert affine.shape == (4, 4) and np.allclose(affine[3], [0,0,0,1])
    points_h = np.c_[points, np.ones(points.shape[0])]
    return np.linalg.solve(affine, points_h.T).T[:,:3]


def compute_lame_parameters(E, nu):
    assert 0 < nu < 0.5
    mu  = E / (2*(1 + nu))
    lam = E * nu / ((1 + nu)*(1 - 2*nu))
    return mu, lam


def compute_density_from_ct(ct, m_atten_ratio=1.0, density_water=1000.0):

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

