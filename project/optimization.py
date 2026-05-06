from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Dict, Tuple

import numpy as np
import torch

from .core import utils, fileio

from .models import ParameterSpec
from .physics import PhysicsAdapter


@dataclass
class OptimizerSpec:
    cls: type
    kws: Dict[str, Any]
    global_steps: int = 10
    local_steps: int = 100


@dataclass
class InitializeSpec:
    num_trials: int = 1
    noise_std: float = 0


# ----- public entry point -----


def optimize_example(ex, config, output_path, raster_base=None):
    utils.check_keys(
        config,
        valid={
            'targets',
            'param_specs',
            'physics_adapter',
            'pde_solver',
            'optimizer',
            'initialize',
            'evaluator',
        },
        where='optimization'
    )
    unit_m = float(ex.metadata['unit'])
    sample = load_example(ex)
    mesh = sample['mesh']

    param_specs  = build_parameter_specs(config)
    optim_spec   = build_optimizer_spec(config)
    init_spec    = build_initialize_spec(config)
    phys_adapter = build_physics_adapter(config)

    utils.log(optim_spec)

    bc_spec = None

    utils.log('Optimizing parameters')

    params = optimize_params(
        phys=phys_adapter,
        mesh=mesh,
        unit_m=unit_m,
        bc_spec=bc_spec,
        param_specs=param_specs,
        optim_spec=optim_spec,
        init_spec=init_spec
    )

    loss, sim_output = evaluate_loss(
        phys=phys_adapter,
        mesh=mesh,
        unit_m=unit_m,
        bc_spec=bc_spec,
        params=params
    )

    utils.log(f'Final loss: {loss.item()}')
    utils.pprint(sim_output)

    if raster_base:
        utils.log('Rasterizing parameters')
        shape = sample['mask'].shape[1:]
        affine = sample['affine']

        rasters = rasterize_params(
            phys=phys_adapter,
            mesh=mesh,
            unit_m=unit_m,
            params=params,
            shape=shape,
            affine=affine
        )
    else:
        rasters = None

    utils.log(f'Evaluating outputs')

    outputs = build_eval_outputs(ex, sample, loss, sim_output, rasters)
    evaluator = build_evaluator(config)
    evaluate_outputs(evaluator, outputs)

    save_output_mesh(mesh, sim_output, output_path)
    if raster_base:
        save_output_rasters(rasters, affine, raster_base)


# ----- context configuration -----


def load_example(ex: Example) -> Dict[str, Any]:
    from . import datasets
    return datasets.torch.TorchDataset([ex])[0]


def build_parameter_specs(config) -> Dict[str, ParameterSpec]:
    target_list = config.get('targets', ['E'])
    utils.log(f'Targets: {target_list}')
    param_specs_cfg = config.get('param_specs', {})
    param_specs = {}
    for name in target_list:
        param_specs[name] = ParameterSpec(**param_specs_cfg[name])
    return param_specs


def build_optimizer_spec(config) -> OptimizerSpec:
    optimizer_kws = config.get('optimizer', {}).copy()
    optimizer_cls = getattr(torch.optim, optimizer_kws.pop('_class'))
    return OptimizerSpec(
        cls=optimizer_cls,
        kws=optimizer_kws,
        global_steps=optimizer_kws.pop('global_steps', 10),
        local_steps=optimizer_kws.pop('local_steps', 100)
    )


def build_initialize_spec(config) -> InitializeSpec:
    initialize_kws = config.get('initialize', {})
    return InitializeSpec(**initialize_kws)


def build_physics_adapter(config) -> PhysicsAdapter:
    physics_adapter_kws = config.get('physics_adapter', {})
    pde_solver_kws = config.get('pde_solver', {}).copy()
    pde_solver_cls = pde_solver_kws.pop('_class')
    return PhysicsAdapter(
        pde_solver_cls=pde_solver_cls,
        pde_solver_kws=pde_solver_kws,
        **physics_adapter_kws
    )


# ----- optimization functions -----


def optimize_params(
    phys: PhysicsAdapter,
    mesh: meshio.Mesh,
    unit_m: float,
    bc_spec: Any,
    param_specs: Dict[str, ParameterSpec],
    optim_spec: OptimizerSpec,
    init_spec: InitializeSpec,
):
    best_loss = None
    best_params = None

    for trial in range(init_spec.num_trials):
        utils.log(f'Trial {trial + 1} / {init_spec.num_trials}')

        param_dofs = initialize_param_dofs(
            phys=phys,
            mesh=mesh,
            unit_m=unit_m,
            param_specs=param_specs,
            init_spec=init_spec
        )
        params, history = run_optimization_trial(
            phys=phys,
            mesh=mesh,
            unit_m=unit_m,
            bc_spec=bc_spec,
            param_specs=param_specs,
            param_dofs=param_dofs,
            optim_spec=optim_spec
        )
        loss, _ = evaluate_loss(
            phys, mesh, unit_m, bc_spec, params
        )
        if best_loss is None or loss.item() < best_loss:
            best_loss = loss.item()
            best_params = params

    return best_params


def initialize_param_dofs(
    phys: PhysicsAdapter,
    mesh: meshio.Mesh,
    unit_m: float,
    param_specs: Dict[str, ParameterSpec],
    init_spec: InitializeSpec
) -> Dict[str, torch.nn.Parameter]:

    dofs = {}
    for name in param_specs:
        z0 = phys.init_param_field(mesh, unit_m)

        if not np.isclose(init_spec.noise_std, 0):
            noise = torch.randn(z0.shape, dtype=z0.dtype, device=z0.device)
            z0 = z0 + init_spec.noise_std * noise

        dofs[name] = torch.nn.Parameter(z0)

    return dofs


def run_optimization_trial(
    phys: PhysicsAdapter,
    mesh: meshio.Mesh,
    unit_m: float,
    bc_spec: Any,
    param_specs: Dict[str, ParameterSpec],
    param_dofs: Dict[str, torch.nn.Parameter],
    optim_spec: OptimizerSpec
) -> Tuple[dict, dict]:

    def objective(params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return phys.mesh_simulation_loss(
            mesh=mesh,
            unit_m=unit_m,
            params=params,
            bc_spec=bc_spec,
            ret_outputs=False
        )

    def local_params() -> Dict[str, torch.Tensor]:
        return {k: v.decode(param_dofs[k]) for k, v in param_specs.items()}

    def global_params() -> Dict[str, torch.Tensor]:
        return {k: v.mean().expand(v.shape) for k, v in local_params().items()}

    history: Dict[str, OptimizationHistory] = {}

    if optim_spec.global_steps > 0:
        utils.log('Stage 1: Global optimization')
        optimizer_g = optim_spec.cls(list(param_dofs.values()), **optim_spec.kws)

        def closure_g() -> torch.Tensor:
            optimizer_g.zero_grad(set_to_none=True)
            params = global_params()
            loss = objective(params)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Invalid loss: {loss.item()}')
            loss.backward()
            return loss

        history['global'] = optimize_closure(
            optimizer_g, closure_g, optim_spec.global_steps
        )

    if optim_spec.local_steps > 0:
        utils.log('Stage 2: Local optimization')
        optimizer_l = optim_spec.cls(list(param_dofs.values()), **optim_spec.kws)

        def closure_l() -> torch.Tensor:
            optimizer_l.zero_grad(set_to_none=True)
            params = local_params()
            loss = objective(params)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Invalid loss: {loss.item()}')
            loss.backward()
            return loss

        history['local'] = optimize_closure(
            optimizer_l, closure_l, optim_spec.local_steps
        )

    with torch.no_grad():
        params = clone_params(local_params())

    return params, history


def optimize_closure(optimizer, closure, max_steps: int = 100):
    history = OptimizationHistory()

    params = []
    for group in optimizer.param_groups:
        params.extend(group['params'])

    loss = None
    for step in range(max_steps):
        loss = optimizer.step(closure)

        if loss is None:
            loss = closure()

        history.update(loss.detach(), params)

        if history.converged(step):
            utils.log('Optimization converged')
            break

    if loss is None:
        loss = closure()

    return history


def clone_params(params: Dict[str, torch.Tensor]):
    return {k: v.detach().clone() for k, v in params.items()}


# ----- evaluation / output -----


def evaluate_loss(
    phys: PhysicsAdapter,
    mesh: meshio.Mesh,
    unit_m: float,
    bc_spec: Any,
    params: Dict[str, torch.Tensor]
):
    with torch.no_grad():
        loss, sim_output = phys.mesh_simulation_loss(
            mesh=mesh,
            unit_m=unit_m,
            params=params,
            bc_spec=bc_spec,
            ret_outputs=True
        )
    return loss.detach(), sim_output


def rasterize_params(
    phys: PhysicsAdapter,
    mesh: meshio.Mesh,
    unit_m: float,
    params: Dict[str, torch.Tensor],
    shape,
    affine
):
    rasters = {}
    for name, field in params.items():
        vox = phys.rasterize_scalar_field(
            mesh=mesh,
            unit_m=unit_m,
            dofs=params[name],
            shape=shape,
            affine=affine
        )
        rasters[name] = vox.cpu()
    return rasters


def build_eval_outputs(ex, sample, loss, sim_output, rasters=None):
    outputs = {
        'example': [ex],
        'mask': sample['mask'].cpu().unsqueeze(0),
        'sim': [sim_output],
        'loss': loss.detach().cpu()
    }
    if 'mat_true' in sample:
        outputs['mat_true'] = sample['mat_true'].cpu().unsqueeze(0)

    if rasters:
        for name, pred_vox in rasters.items():
            pred_key = f'{name}_pred'
            true_key = f'{name}_true'
            outputs[pred_key] = pred_vox.cpu().unsqueeze(0)
            if true_key in sample:
                outputs[true_key] = sample[true_key].cpu().unsqueeze(0)

    return outputs


def build_evaluator(config):
    from . import evaluation
    evaluator_kws = config.get('evaluator', {})
    return evaluation.EvaluatorCallback(**evaluator_kws)


def evaluate_outputs(evaluator, outputs):
    evaluator.evaluate(epoch=0, phase='optimize', batch=0, step=0, outputs=outputs)
    evaluator.on_phase_end(epoch=0, phase='optimize')


def save_output_mesh(mesh, sim_output, output_path):

    def _assign_mesh_field(m, name):
        field = sim_output.get(name)
        if field is not None:
            m.point_data[name] = field.nodes.numpy()
            m.cell_data[name] = [field.cells.numpy()]

    output_mesh = mesh.copy()
    _assign_mesh_field(output_mesh, 'E_true')
    _assign_mesh_field(output_mesh, 'E_pred')
    _assign_mesh_field(output_mesh, 'mu_true')
    _assign_mesh_field(output_mesh, 'mu_pred')
    _assign_mesh_field(output_mesh, 'lam_true')
    _assign_mesh_field(output_mesh, 'lam_pred')
    _assign_mesh_field(output_mesh, 'rho_true')
    _assign_mesh_field(output_mesh, 'rho_pred')
    _assign_mesh_field(output_mesh, 'u_true')
    _assign_mesh_field(output_mesh, 'u_pred')
    _assign_mesh_field(output_mesh, 'residual')

    utils.log(output_mesh)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_meshio(output_path, output_mesh)


def save_output_rasters(rasters, affine, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, pred_vox in rasters.items():
        output_path = output_dir / f'{name}_pred.nii.gz'
        output_array = pred_vox.detach().cpu().numpy()
        fileio.save_nibabel(output_path, output_array, affine)


# ----- optimization history -----


def compute_grad_norm(params: List[torch.nn.Parameter]) -> float:
    ssq = 0.0
    for p in params:
        if p.grad is not None:
            ssq += float(p.grad.pow(2).sum().cpu().item())
    return np.sqrt(ssq)


def flatten_params(params: List[torch.nn.Parameter]) -> float:
    arrays = [p.detach().cpu().numpy().ravel() for p in params]
    return np.concatenate(arrays)


class OptimizationHistory:

    def __init__(self):
        self.loss_history: List[float] = []
        self.grad_history: List[float] = []
        self.param_history: List[np.array] = []

        utils.log('iter\tloss (rel_delta)\tgrad_norm (rel_init)\tparam_norm (update_norm)')

    def update(self, loss: torch.Tensor, params: List[torch.nn.Parameter]):
        curr_loss = float(loss.detach().item())
        curr_grad = compute_grad_norm(params)
        curr_params = flatten_params(params)

        self.loss_history.append(curr_loss)
        self.grad_history.append(curr_grad)
        self.param_history.append(curr_params)

    def converged(self, it: int, tol: float=1e-3, eps: float=1e-12) -> bool:
        from numpy.linalg import norm

        loss_delta = np.nan
        grad_delta = np.nan
        param_delta = np.nan

        curr_loss = self.loss_history[-1]
        curr_grad = self.grad_history[-1]
        curr_param = self.param_history[-1]
        curr_norm  = norm(curr_param)

        if len(self.loss_history) > 1:
            prev_loss = self.loss_history[-2]
            loss_delta = abs(prev_loss - curr_loss) / max(abs(prev_loss), eps)

        if len(self.grad_history) > 0:
            grad_delta = curr_grad / max(self.grad_history[0], eps)

        if len(self.param_history) > 1:
            prev_param = self.param_history[-2]
            param_delta = norm(curr_param - prev_param) / max(norm(prev_param), eps)

        utils.log(
            f'{it}\t{curr_loss:.4e} ({loss_delta:.4e})'
            f'\t{curr_grad:.4e} ({grad_delta:.4e})'
            f'\t{curr_norm:.4e} ({param_delta:.4e})'
        )

        if np.isnan(curr_loss) or np.isnan(curr_grad) or np.isnan(curr_norm):
            raise RuntimeError('Optimization encountered nan value')

        return loss_delta < tol or grad_delta < tol or param_delta < tol


