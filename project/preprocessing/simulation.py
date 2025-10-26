from __future__ import annotations
import numpy as np
import torch

from ..core import utils, transforms, interpolation


def simulate_forward(
    mesh: meshio.Mesh,     # world units
    affine: np.ndarray,    # world units
    rho_field: np.ndarray, # kg/m^3
    E_field: np.ndarray,   # Pa
    nu_value: float = 0.4, # unitless
    unit_m: float = 1e-3,  # meters per world unit
    solver_kws=None,
    dtype=torch.float32,
    device='cuda'
):
    '''
    Args:
        mesh: Tetrahedral volume mesh in world coordinates
        affine: (4, 4) array mapping voxel -> world coordinates
        rho_field: (I, J, K) mass density array (kg/m^3)
        E_field: (I, J, K) Young's modulus array (Pa)
        nu_value: Poisson ratio constant (default: 0.4)
        unit_m: World space unit, in meters (default=1e-3)
        solver_kws: Optional params passed to PDE solver
    Returns:
        u_field: (I, J, K, 3) simulated displacement field
            in world space units, rasterized on input grid
        node_values: Dict[str, np.ndarray] mapping of keys to
            len N arrays of field values on the N mesh nodes
    '''
    from .. import solvers

    # map mesh nodes from world to voxel coords to sample volumes
    pts_voxel = transforms.world_to_voxel_coords(mesh.points, affine)

    def _as_tensor(a):
        return torch.as_tensor(a, dtype=dtype, device=device)

    pts_tensor = _as_tensor(pts_voxel)              # (N, 3)
    rho_tensor = _as_tensor(rho_field).unsqueeze(0) # (1, I, J, K)
    E_tensor   = _as_tensor(E_field).unsqueeze(0)   # (1, I, J, K)

    utils.log('Interpolating material fields onto mesh nodes')
    rho_nodes = interpolation.interpolate_image(rho_tensor, pts_tensor) # (N, 1)
    E_nodes = interpolation.interpolate_image(E_tensor, pts_tensor)     # (N, 1)

    mu_nodes, lam_nodes = transforms.compute_lame_parameters(E_nodes, nu_value)

    bc_nodes = torch.zeros_like(pts_tensor, dtype=dtype, device=device) # (N, 3)

    utils.log('Initializing finite element solver')
    solver = solvers.warp.WarpFEMSolver(mesh, unit_m=unit_m, **(solver_kws or {}))
    solver.set_observed(bc_nodes)
    solver.set_density(rho_nodes)
    solver.set_elasticity(mu_nodes, lam_nodes)

    utils.log('Assembling load vector')
    solver.assemble_forcing()

    utils.log('Assembling stiffness matrix')
    solver.assemble_stiffness()

    utils.log('Applying boundary conditions')
    solver.assemble_projector()
    solver.assemble_lifting()
    solver.assemble_rhs_shift()

    utils.log('Solving forward system')
    u_nodes = solver.solve_forward(ret_tensor=True)

    utils.log('Rasterizing output field')
    u_field = solvers.warp.rasterize_field(solver.u_sim_field, E_field.shape, affine)

    # convert displacements from meters to world units
    u_nodes = u_nodes / unit_m
    u_field = u_field / unit_m

    node_values = {
        'E': E_nodes.detach().cpu().numpy(),
        'mu': mu_nodes.detach().cpu().numpy(),
        'lam': lam_nodes.detach().cpu().numpy(),
        'rho': rho_nodes.detach().cpu().numpy(),
        'bc': bc_nodes.detach().cpu().numpy(),
        'u': u_nodes.detach().cpu().numpy()
    }
    return u_field.detach().cpu().numpy(), node_values

