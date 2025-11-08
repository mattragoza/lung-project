from __future__ import annotations
import numpy as np
import torch

from ..core import utils, transforms, interpolation


def simulate_displacement(
    verts: np.ndarray,     # world coords
    cells: np.ndarray,     # tetra cells
    rho_cells: np.ndarray, # kg/m^3
    E_cells: np.ndarray,   # Pa
    nu_value: float=0.4,   # unitless
    unit_m: float=1e-3,    # meters per world unit
    solver_kws=None,
    dtype=torch.float32,
    device='cuda'
):
    '''
    Args:
        verts: (N, 3) mesh vertices in world coordinates
        cells: (M, 4) tetrahedral mesh cells (vertex indices)
        rho_cells: (M,) mass density array (kg/m^3)
        E_cells: (M,) Young's modulus array (Pa)
        nu_value: Poisson ratio constant (default: 0.4)
        unit_m: World space unit, in meters (default=1e-3)
        solver_kws: Optional params passed to PDE solver
    Returns:
        node_values: Dict[str, np.ndarray] mapping of keys to
            len N arrays of field values on the N mesh nodes
    '''
    from .. import solvers

    verts = torch.as_tensor(verts * unit_m, dtype=dtype, device=device) # to meters
    cells = torch.as_tensor(cells, dtype=torch.int, device=device)

    rho_cells = torch.as_tensor(rho_cells, dtype=dtype, device=device)
    E_cells   = torch.as_tensor(E_cells, dtype=dtype, device=device)

    mu_cells, lam_cells = transforms.compute_lame_parameters(E_cells, nu_value)
    bc_nodes = torch.zeros_like(verts, dtype=dtype, device=device) # meters

    utils.log('Simulating displacement using PDE solver')
    solver = solvers.warp.WarpFEMSolver(unit_m=unit_m, **(solver_kws or {}))
    solver.init_geometry(verts, cells)

    u_nodes = solver.simulate(mu_cells, lam_cells, rho_cells, bc_nodes) / unit_m # to world units
    res_nodes = solver.get_residual()

    return {
        'u':   u_nodes.detach().cpu().numpy(),
        'bc':  bc_nodes.detach().cpu().numpy(),
        'res': res_nodes.detach().cpu().numpy()
    }


def optimize_elasticity(
    verts: np.ndarray,       # world units
    cells: np.ndarray,       # tetra cells
    rho_cells: np.ndarray,   # kg/m^3
    u_obs_nodes: np.ndarray, # world units
    nu_value: float=0.4,     # unitless
    unit_m: float=1e-3,      # meters per world unit
    theta_init: float=3.,    # log Pa (base 10)
    solver_kws=None,
    global_kws=None,
    local_kws=None,
    dtype=torch.float32,
    device='cuda'
):
    from .. import solvers

    verts = torch.as_tensor(verts * unit_m, dtype=dtype, device=device) # to meters
    cells = torch.as_tensor(cells, dtype=torch.int, device=device)

    rho_cells = torch.as_tensor(rho_cells, dtype=dtype, device=device)
    u_obs_nodes = torch.as_tensor(u_obs_nodes, dtype=dtype, device=device) * unit_m # to meters

    solver = solvers.warp.WarpFEMSolver(unit_m=unit_m, **(solver_kws or {}))
    solver.init_geometry(verts, cells)

    module = solvers.base.PDESolverModule(solver, rho_cells, u_obs_nodes)

    theta_global = torch.full((1,), theta_init, requires_grad=True, dtype=dtype, device=device)
    theta_local = torch.zeros(rho_cells.shape, requires_grad=True, dtype=dtype, device=device)

    def f(theta_g, theta_l):
        E = transforms.parameterize_youngs_modulus(theta_g, theta_l)
        mu, lam = transforms.compute_lame_parameters(E, nu=nu_value)
        return module.forward(mu, lam)

    utils.log('Optimizing global parameter(s)')
    optimize_lbfgs(lambda x: f(x, theta_local), theta_global, **(global_kws or {}))

    utils.log('Optimizing local parameter(s)')
    optimize_lbfgs(lambda x: f(theta_global, x), theta_local, **(local_kws or {}))

    # final forward solve
    E_opt = transforms.parameterize_youngs_modulus(theta_global, theta_local)
    mu, lam = transforms.compute_lame_parameters(E_opt, nu=nu_value)
    u_opt = solver.simulate(mu, lam, rho_cells, u_obs_nodes) / unit_m # to world units
    r_opt = solver.get_residual()

    node_values = {
        'u_opt': u_opt.detach().cpu().numpy(),
        'r_opt': r_opt.detach().cpu().numpy(),
    }
    cell_values = {'E_opt': E_opt.detach().cpu().numpy()}

    return node_values, cell_values


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


def rasterize_field(warp_field, shape, affine, unit_m):
    origin  = transforms.get_affine_origin(affine) * unit_m # to meters
    spacing = transforms.get_affine_spacing(affine) * unit_m
    bounds  = transforms.get_grid_bounds(origin, spacing, shape, align_corners=False)
    output  = solvers.warp.rasterize_field(warp_field, shape, bounds)
    return output.detach().cpu().numpy()


