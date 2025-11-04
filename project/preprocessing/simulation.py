from __future__ import annotations
import numpy as np
import torch

from ..core import utils, transforms, interpolation


def simulate_displacement(
    mesh: meshio.Mesh,     # world units
    affine: np.ndarray,    # world units
    rho_field: np.ndarray, # kg/m^3
    E_field: np.ndarray,   # Pa
    nu_value: float=0.4,   # unitless
    unit_m: float=1e-3,    # meters per world unit
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

    tensor_kws = dict(dtype=dtype, device=device)
    pts_tensor = torch.as_tensor(pts_voxel, **tensor_kws)              # (N, 3)
    rho_tensor = torch.as_tensor(rho_field, **tensor_kws).unsqueeze(0) # (1, I, J, K)
    E_tensor   = torch.as_tensor(E_field, **tensor_kws).unsqueeze(0)   # (1, I, J, K)

    utils.log('Interpolating material fields onto mesh nodes')
    rho_nodes = interpolation.interpolate_image(rho_tensor, pts_tensor) # (N, 1)
    E_nodes   = interpolation.interpolate_image(E_tensor, pts_tensor)   # (N, 1)

    mu_nodes, lam_nodes = transforms.compute_lame_parameters(E_nodes, nu_value)

    bc_nodes = torch.zeros_like(pts_tensor, **tensor_kws) # (N, 3)

    utils.log('Initializing finite element solver')
    solver = solvers.warp.WarpFEMSolver(mesh, unit_m=unit_m, **(solver_kws or {}))
    solver.set_observed(bc_nodes)
    solver.set_density(rho_nodes)
    solver.set_elasticity(mu_nodes, lam_nodes)

    utils.log('Assembling linear system')
    solver.assemble_all()

    utils.log('Solving forward system')
    u_nodes = solver.solve_forward(ret_tensor=True)
    print(u_nodes.sum())

    utils.log('Rasterizing output field')
    shape = E_field.shape
    origin = transforms.get_affine_origin(affine) * unit_m # to meters
    spacing = transforms.get_affine_spacing(affine) * unit_m
    bounds = transforms.get_grid_bounds(origin, spacing, shape, align_corners=False)
    u_field = solvers.warp.rasterize_field(solver.u_sim_field, shape, bounds)
    print(u_field.sum())

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


def optimize_elasticity(
    mesh: meshio.Mesh,       # world units
    shape, affine,
    rho_nodes: np.ndarray,   # kg/m^3
    E_nodes: np.ndarray,     # Pa
    u_obs_nodes: np.ndarray, # world units
    nu_value: float=0.4,     # unitless
    unit_m: float=1e-3,      # meters per world unit
    theta_init: float=3.,    # log10 Pa
    solver_kws=None,
    global_kws=None,
    local_kws=None,
    dtype=torch.float32,
    device='cuda'
):
    from .. import solvers

    tensor_kws = dict(dtype=dtype, device=device)
    rho_tensor = torch.as_tensor(rho_nodes, **tensor_kws)
    u_obs_tensor = torch.as_tensor(u_obs_nodes, **tensor_kws) * unit_m # to meters
    E_tensor = torch.as_tensor(E_nodes, **tensor_kws) # Pa

    solver = solvers.warp.WarpFEMSolver(mesh, unit_m=unit_m, **(solver_kws or {}))
    module = solvers.base.PDESolverModule(solver, rho_tensor, u_obs_tensor)

    theta_global = torch.full((1,), theta_init, requires_grad=True, **tensor_kws)
    theta_local = torch.zeros(rho_nodes.shape, requires_grad=True, **tensor_kws)

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
    solver.set_elasticity(mu, lam)
    solver.assemble_all()
    u_opt = solver.solve_forward(ret_tensor=True) / unit_m # back to world units

    node_values = {
        'E_opt': E_opt.detach().cpu().numpy(),
        'u_opt': u_opt.detach().cpu().numpy()
    }
    print(node_values['E_opt'].mean())

    utils.log('Rasterizing output field')
    origin = transforms.get_affine_origin(affine) * unit_m # to meters
    spacing = transforms.get_affine_spacing(affine) * unit_m
    bounds = transforms.get_grid_bounds(origin, spacing, shape, align_corners=False)
    mu_field = solvers.warp.rasterize_field(solver.mu_field, shape, bounds)[:,:,:,0] # (I, J, K)
    E_field = transforms.compute_youngs_modulus(mu_field, nu=nu_value)
    print(E_field.sum())

    return E_field.detach().cpu().numpy(), node_values


def _update_defaults(overrides=None, **defaults):
    return defaults | (overrides or {})


def optimize_lbfgs(
    fn, param,
    max_iter=100,
    tol=1e-3,
    eps=1e-8,
    lbfgs_kws=None
):
    lbfgs_kws = _update_defaults(
        lbfgs_kws,
        lr=1.0, max_iter=20, history_size=100, line_search_fn='strong_wolfe'
    )
    optim = torch.optim.LBFGS(params=[param], **lbfgs_kws)

    def closure():
        optim.zero_grad()
        loss = fn(param)
        loss.backward()
        return loss

    loss_history = []
    grad_history = []

    for i in range(max_iter):
        loss = optim.step(closure)

        loss_history.append(loss.item())
        grad_history.append(param.grad.norm().item())

        if len(loss_history) < 2:
            rel_loss_delta = np.nan
        else:
            curr_loss = loss_history[-1]
            prev_loss = loss_history[-2]
            rel_loss_delta = abs(prev_loss - curr_loss) / max(abs(prev_loss), eps)

        if len(grad_history) < 1:
            rel_grad_norm = np.nan
        else:
            curr_norm = grad_history[-1]
            init_norm = grad_history[0]
            rel_grad_norm = curr_norm / max(init_norm, eps)

        utils.log(f'Iteration {i} | loss = {loss.item():.4e} | delta = {rel_loss_delta:.4e} | grad = {rel_grad_norm:.4}')

        if rel_loss_delta < tol or rel_grad_norm < tol:
            utils.log('Optimization converged')
            break

    return param, loss_history, grad_history

