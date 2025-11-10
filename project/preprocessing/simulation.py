from __future__ import annotations
import numpy as np
import torch

from ..core import utils, transforms


def simulate_displacement(
    verts: np.ndarray,      # world coords
    cells: np.ndarray,      # tetra cells
    rho_values: np.ndarray, # kg/m^3
    E_values: np.ndarray,   # Pa
    nu_value: float=0.4,    # unitless
    unit_m: float=1e-3,     # meters per world unit
    scalar_degree: int=0,
    vector_degree: int=1,
    solver_kws=None,
    dtype=torch.float32,
    device='cuda'
):
    '''
    Args:
        verts: (N, 3) mesh vertices in world coordinates
        cells: (M, 4) tetrahedral mesh cells (vertex indices)
        rho_values: (N,) mass density array (kg/m^3)
        E_values: (N,) Young's modulus array (Pa)
        nu_value: Poisson ratio constant (default: 0.4)
        unit_m: World space unit, in meters (default=1e-3)
        solver_kws: Optional params passed to PDE solver
    Returns:
        ret_values: Dict[str, np.ndarray] mapping of keys to
            len N arrays of field values on the N mesh nodes
    '''
    from .. import solvers

    verts = torch.as_tensor(verts * unit_m, dtype=dtype, device=device) # to meters
    cells = torch.as_tensor(cells, dtype=torch.int, device=device)

    rho_values = torch.as_tensor(rho_values, dtype=dtype, device=device)
    E_values   = torch.as_tensor(E_values, dtype=dtype, device=device)

    mu_values, lam_values = transforms.compute_lame_parameters(E_values, nu_value)
    bc_values = torch.zeros_like(verts, dtype=dtype, device=device) # meters

    solver = solvers.warp.WarpFEMSolver(
        scalar_degree=scalar_degree,
        vector_degree=vector_degree,
        **(solver_kws or {})
    )
    solver.init_geometry(verts, cells)

    u_values, residual = solver.simulate(mu_values, lam_values, rho_values, bc_values)
    return {
        'u':   u_values.detach().cpu().numpy() / unit_m, # to world units
        'bc':  bc_values.detach().cpu().numpy() / unit_m,
        'res': residual.detach().cpu().numpy()
    }


def rasterize_elasticity(
    shape: tuple,
    affine: np.ndarray,
    verts: np.ndarray,      # world coords
    cells: np.ndarray,      # tetra cells
    E_values: np.ndarray,   # Pa
    nu_value: float=0.4,    # unitless
    unit_m: float=1e-3,     # meters per world unit
    scalar_degree: int=1,
    dtype=torch.float32,
    device='cuda'
):
    from .. import solvers

    verts = torch.as_tensor(verts * unit_m, dtype=dtype, device=device) # to meters
    cells = torch.as_tensor(cells, dtype=torch.int, device=device)

    E_values = torch.as_tensor(E_values, dtype=dtype, device=device)

    solver = solvers.warp.WarpFEMSolver(scalar_degree=scalar_degree)
    solver.init_geometry(verts, cells)
    E_warp = solver.make_scalar_field(E_values)

    origin  = transforms.get_affine_origin(affine) * unit_m # to meters
    spacing = transforms.get_affine_spacing(affine) * unit_m
    bounds  = transforms.get_grid_bounds(origin, spacing, shape, align_corners=False)

    E_field = solvers.warp.rasterize_field(E_warp, shape, bounds, device)

    return E_field.detach().cpu().numpy()


def optimize_elasticity(
    verts: np.ndarray,        # world units
    cells: np.ndarray,        # tetra cells
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
        **(solver_kws or {})
    )
    module = solvers.base.PDESolverModule(solver, verts, cells, rho_values, u_obs_values)

    theta_global = torch.full((1,), theta_init, requires_grad=True, dtype=dtype, device=device)
    theta_local = torch.zeros(rho_values.shape, requires_grad=True, dtype=dtype, device=device)

    def f(theta_g, theta_l):
        E_vals = transforms.parameterize_youngs_modulus(theta_g, theta_l)
        mu_vals, lam_vals = transforms.compute_lame_parameters(E_vals, nu_value)
        return module.forward(mu_vals, lam_vals)

    utils.log('Optimizing global parameter(s)')
    optimize_lbfgs(lambda x: f(x, theta_local), theta_global, **(global_kws or {}))

    utils.log('Optimizing local parameter(s)')
    optimize_lbfgs(lambda x: f(theta_global, x), theta_local, **(local_kws or {}))

    # final forward solve
    E_opt_values = transforms.parameterize_youngs_modulus(theta_global, theta_local)
    mu_values, lam_values = transforms.compute_lame_parameters(E_opt_values, nu_value)
    u_opt_values, res_values, loss = solver.adjoint_forward(mu_values, lam_values)

    return {
        'E_opt': E_opt_values.detach().cpu().numpy(),
        'u_opt': u_opt_values.detach().cpu().numpy() / unit_m, # to world units
        'res_opt': res_values.detach().cpu().numpy()
    }


def optimize_lbfgs(fn, param, max_iter=100, tol=1e-3, eps=1e-8, lbfgs_kws=None):

    lbfgs_kws = utils.update_defaults(
        lbfgs_kws,
        lr=1.0, max_iter=20, history_size=100, line_search_fn='strong_wolfe'
    )
    optim = torch.optim.LBFGS(params=[param], **lbfgs_kws)

    loss_history = []
    grad_history = []
    param_history = []

    def closure():
        optim.zero_grad()
        loss = fn(param)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Invalid loss: {loss.item()}')
        loss.backward()
        if not torch.isfinite(param.grad).all():
            raise RuntimeError(f'Invalid gradient: {g.detach().cpu().numpy()}')
        return loss

    utils.log('iter\tloss (rel_delta)\tgrad_norm (rel_init)\tparam_norm (update_norm)')

    for i in range(max_iter):
        loss = optim.step(closure)
        curr_loss = float(loss.detach().item())
        curr_grad = float(param.detach().norm().item())
        curr_param = param.detach().cpu().numpy()
        curr_norm = np.linalg.norm(curr_param)

        if loss_history:
            prev_loss = loss_history[-1]
            loss_delta = abs(prev_loss - curr_loss) / max(abs(prev_loss), eps)
        else:
            loss_delta = np.nan

        if grad_history:
            grad_delta = curr_grad / max(grad_history[0], eps)
        else:
            grad_delta = np.nan

        if param_history:
            prev_param = param_history[-1]
            update_norm = np.linalg.norm(curr_param - prev_param)
            param_delta = update_norm / max(np.linalg.norm(prev_param), eps)
        else:
            param_delta = np.nan

        utils.log(
            f'{i}\t{curr_loss:.4e} ({loss_delta:.4e})'
            f'\t{curr_grad:.4e} ({grad_delta:.4e})'
            f'\t{curr_norm:.4e} ({param_delta:.4e})'
        )
        loss_history.append(curr_loss)
        grad_history.append(curr_grad)
        param_history.append(curr_param)

        if np.isnan(curr_loss) or np.isnan(curr_grad) or np.isnan(curr_norm):
            raise RuntimeExcepton('Optimization encountered nan')

        if loss_delta < tol or grad_delta < tol or param_delta < tol:
            utils.log('Optimization converged')
            break

    return param

