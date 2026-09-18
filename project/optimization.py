# optimization.py

from typing import List, Dict, Tuple, Callable, Any

import numpy as np
import torch
import meshio

from .common import fileio, utils, outputs

from . import datasets, physics, evaluation


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and value > 0


def _zero_dofs(template: torch.Tensor) -> torch.nn.Parameter:
    return torch.nn.Parameter(torch.zeros_like(template))


def _clone_state(m: torch.nn.Module) -> dict:
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


class ParameterDict(torch.nn.Module):

    def __init__(self, template, dof_mapper, **kwargs):
        from .param_spec import ParameterSpec

        if not kwargs:
            raise ValueError('No target parameter(s) specified')

        super().__init__()

        self.specs = {name: ParameterSpec(**spec) for name, spec in kwargs.items()}
        self._dofs = torch.nn.ParameterDict({n: _zero_dofs(template) for n in kwargs})

        self.dof_mapper = dof_mapper

    @torch.no_grad()
    def initialize(self, scale: float):
        for name, dofs in self._dofs.items():
            dofs.uniform_(-scale, +scale)

    def forward(self, global_mean: bool = False, map_dofs: bool = False) -> dict:
        params = {}
        for name, dofs in self._dofs.items():
            if global_mean:
                dofs = dofs.mean().expand_as(dofs)
            elif map_dofs:
                dofs = torch.sparse.mm(self.dof_mapper, dofs[:, None])[:, 0]
            params[name] = self.specs[name].decode(dofs)
        return params


# ----- public entry point -----


def optimize_example(ex, config):
    utils.check_keys(
        config,
        valid={
            'parameters',
            'pde_solver',
            'physics_adapter',
            'boundary_condition',
            'optimization_kws',
            'output_name'
        },
        where='optimization'
    )

    sample = datasets.load_example(ex)
    mesh, unit_m = sample['mesh'], float(ex.metadata['unit'])

    solver = physics.get_solver(**config.get('pde_solver', {}))
    adapter = physics.get_adapter(solver, **config.get('physics_adapter', {}))
    bc_spec = physics.get_bc_spec(**config.get('boundary_condition', {}))

    dof_mapper = make_surface_mapper(mesh.points, mesh.cells_dict['tetra'])
    dof_mapper = _as_sparse_tensor(dof_mapper, torch.float32, adapter.device)

    template = adapter.initialize_scalar_field(mesh, unit_m)
    param_dict = ParameterDict(template, dof_mapper, **config.get('parameters', {}))

    def objective(params: dict):
        return adapter.mesh_simulation_loss(mesh, unit_m, params, bc_spec)[0]

    optim_kws = config.get('optimization_kws', {})
    params = run_optimization_trials(objective, param_dict, **optim_kws)

    with torch.no_grad(): # get final simulation outputs using optimized params
        sim = adapter.mesh_simulation_loss(mesh, unit_m, params, bc_spec, True)[1]

    rasters = {}
    for name, param in params.items():
        utils.log(f'Rasterizing parameter: {name}')
        rasters[name] = adapter.rasterize_scalar_field(
            mesh, unit_m, param, sample['mask'].shape[1:], sample['affine']
        )

    outname = config.get('output_name', 'optimize')
    save_optimization_results(ex, sample, rasters, sim, outname)


def run_optimization_trials(
    objective: Callable,
    param_dict: ParameterDict,
    num_trials: int = 1,
    **kwargs
) -> Dict[str, torch.Tensor]:

    if not _positive_int(num_trials):
        raise ValueError('num_trials must be a positive int')

    best_state = None
    best_loss = float('inf')

    for trial in range(num_trials):
        utils.log(f'Start optimization trial {trial + 1} / {num_trials}')

        try:
            trial_loss = run_optimization_trial(objective, param_dict, **kwargs)[-1]

        except RuntimeError as e:
            utils.warn(f'FAILED: {e}')
            continue

        if trial_loss < best_loss:
            best_state = _clone_state(param_dict)
            best_loss = trial_loss

    if best_state is None:
        raise RuntimeError('All optimization trials failed.')

    utils.log(f'Best loss: {best_loss}')
    param_dict.load_state_dict(best_state)

    with torch.no_grad():
        return param_dict(global_mean=False, map_dofs=False)


def run_optimization_trial(
    objective: Callable,
    param_dict: ParameterDict,
    init_scale: float = 0,
    global_steps: int = 0,
    local_steps: int = 100,
    **kwargs
) -> List[float]:

    if not _positive_int(global_steps) and not _positive_int(local_steps):
        raise ValueError('global_steps or local_steps must be a positive int')

    utils.log('Initializing parameters')
    param_dict.initialize(init_scale)

    loss_history = []

    if global_steps > 0:
        utils.log('Optimizing global mean(s)')
        loss_history += run_optimization_steps(
            objective, param_dict, True, global_steps, **kwargs
        )

    if local_steps > 0:
        utils.log('Optimizing local values')
        loss_history += run_optimization_steps(
            objective, param_dict, False, local_steps, **kwargs
        )

    return loss_history


def run_optimization_steps(
    objective: Callable,
    param_dict: ParameterDict,
    global_mean: bool = False,
    max_steps: int = 100,
    rtol: float = 1e-5,
    **kwargs
):
    if not _positive_int(max_steps):
        raise ValueError('max_steps must be a positive int')

    optimizer = _get_optimizer(param_dict, **kwargs)

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = objective(param_dict(global_mean))
        loss.backward()
        with torch.no_grad():
            grad_norm = _compute_grad_norm(param_dict.parameters())
        if not np.isfinite(grad_norm):
            raise RuntimeError(f'Non-finite loss gradient: {grad_norm}')
        return loss

    loss_history = []

    for step in range(max_steps + 1):

        with torch.no_grad():
            loss = objective(param_dict(global_mean)).item()

        loss_history.append(loss)
        loss_delta = _compute_loss_delta(loss_history, relative=True)

        utils.log(f'[step {step}] loss = {loss:.4e} (delta = {loss_delta:.4e})')

        if not np.isfinite(loss):
            raise RuntimeError(f'Non-finite loss value: {loss}')

        if step > 0 and loss_delta < rtol:
            utils.log(f'Optimization converged in {step} step(s)')
            break

        if step == max_steps:
            utils.log(f'Optimization reached max steps ({step})')
            break

        optimizer.step(closure)

    return loss_history


def _get_optimizer(m: torch.nn.Module, type: str, **kwargs):
    optimizer_cls = getattr(torch.optim, type)
    return optimizer_cls(m.parameters(), **kwargs)


def _compute_loss_delta(loss_history: List[float], relative: bool) -> float:
    if len(loss_history) < 2:
        return np.nan
    prev_loss, curr_loss = loss_history[-2:]
    abs_delta = abs(curr_loss - prev_loss)
    if relative:
        return abs_delta / abs(prev_loss)
    return abs_delta


def _compute_grad_norm(params: List[torch.nn.Parameter]) -> float:
    total = 0.0
    for param in params:
        if param.grad is not None:
            total += float(param.grad.pow(2).sum().cpu().item())
    return np.sqrt(total)


def save_optimization_results(ex, sample, rasters, sim_outputs, outname='optimize'):
    out = outputs.Outputs(stage=outname)

    mesh = sample['mesh'].copy()
    for name, param in sim_outputs['params'].items():
        mesh.point_data[name] = param.node_values.detach().cpu().numpy()
        mesh.cell_data[name] = [param.cell_values.detach().cpu().numpy()]

    mesh_path = out.mesh_path(ex, name='output')
    fileio.save_meshio(mesh_path, mesh)

    raster_dir = out.raster_dir(ex)
    for name, raster in rasters.items():
        raster = raster.detach().cpu().numpy()[0]
        fileio.save_nibabel(raster_dir / f'{name}_pred.nii.gz', raster, sample['affine'])

    evaluator = evaluation.Evaluator()
    metrics = evaluator.evaluate_sample(sample, rasters, sim_outputs, groupby=None)

    csv_path = out.csv_path(name='metrics')
    fileio.save_csv(csv_path, metrics)


# ----- surface extrapolation -----


def make_surface_mapper(verts, cells):
    '''
    Constuct a sparse matrix that maps surface dof values
    to the average of adjacent interior vertex dof values.
    '''
    import scipy.sparse as sp
    from .common import transforms

    # get masks for surface and interior vertices
    surface_mask = transforms.get_surface_mask(verts, cells)
    interior_mask = ~surface_mask

    A = transforms.get_vertex_adjacency(verts, cells)

    # count number of adjacent interior vertices
    neighbors = A @ sp.diags((interior_mask.astype(float)))
    n_neighbors = np.asarray(neighbors.sum(axis=1)).ravel()

    # interior vertices map to themselves,
    #   surface vertices map to the average of interior neighbors
    extrapolate = surface_mask & (n_neighbors > 0)
    keep = ~extrapolate

    if not np.any(extrapolate):
        utils.warn('WARNING: no dofs are extrapolated')

    weights = np.zeros(len(verts))
    weights[extrapolate] = 1 / n_neighbors[extrapolate]

    M = sp.diags(keep.astype(float)) + sp.diags(weights) @ neighbors
    return M.tocsr()


def _as_sparse_tensor(M, dtype, device):
    M = M.tocoo()

    indices = torch.as_tensor(
        np.vstack([M.row, M.col]), dtype=torch.long, device=device
    )
    values = torch.as_tensor(M.data, dtype=dtype, device=device)

    return torch.sparse_coo_tensor(
        indices, values, M.shape
    ).coalesce()

