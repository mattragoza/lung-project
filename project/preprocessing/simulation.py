from __future__ import annotations
import numpy as np
import torch

from ..core import utils, transforms


def optimize_elasticity(
    verts: np.ndarray,        # world units
    cells: np.ndarray,        # tetra
    rho_values: np.ndarray,   # kg/m^3
    u_obs_values: np.ndarray, # world units
    nu_value: float=0.4,      # unitless
    unit_m: float=1e-3,       # meters per world unit
    scalar_degree: int=1,
    vector_degree: int=1,
    solver_kws=None,
    theta_init: float=3.,    # log Pa (base 10)
    global_kws=None,
    local_kws=None,
    dtype=torch.float32,
    device='cuda'
):
    from .. import solvers

    verts = torch.as_tensor(verts * unit_m, dtype=dtype, device=device) # to meters
    cells = torch.as_tensor(cells, dtype=torch.int, device=device)

    rho_values = torch.as_tensor(rho_values, dtype=dtype, device=device)
    u_obs_values = torch.as_tensor(u_obs_values * unit_m, dtype=dtype, device=device) # to meters

    solver = solvers.warp.WarpFEMSolver(
        scalar_degree=scalar_degree,
        vector_degree=vector_degree,
        device=device,
        **(solver_kws or {})
    )
    module = solvers.base.PDESolverModule(solver, verts, cells, rho_values, u_obs_values)

    theta_global = torch.full((1,), theta_init, requires_grad=True, dtype=dtype, device=device)
    theta_local = torch.zeros(rho_values.shape, requires_grad=True, dtype=dtype, device=device)

    def f(theta_g, theta_l):
        E_vals = transforms.parameterize_youngs_modulus(theta_g, theta_l)
        mu_vals, lam_vals = transforms.compute_lame_parameters(E_vals, nu_value)
        loss, res, u_sim = module.forward(mu_vals, lam_vals)
        return loss, res, u_sim

    utils.log('Optimizing global parameter(s)')
    optimize_lbfgs(lambda x: f(x, theta_local)[0], theta_global, **(global_kws or {}))

    utils.log('Optimizing local parameter(s)')
    optimize_lbfgs(lambda x: f(theta_global, x)[0], theta_local, **(local_kws or {}))

    # final forward solve
    E_opt_values = transforms.parameterize_youngs_modulus(theta_global, theta_local)
    mu_values, lam_values = transforms.compute_lame_parameters(E_opt_values, nu_value)
    u_opt_values, res_values, loss = solver.adjoint_forward(mu_values, lam_values)

    return {
        'E_opt': E_opt_values.detach().cpu().numpy(),
        'u_opt': u_opt_values.detach().cpu().numpy() / unit_m, # to world units
        'res_opt': res_values.detach().cpu().numpy()
    }


