from __future__ import annotations
from typing import Any, List, Dict, Tuple
import numpy as np
import torch

from .core import utils, fileio
from .models import ParameterSpec


def optimize_example(ex, config, output_path, raster_base):
    utils.check_keys(
        config,
        valid={'targets', 'param_specs', 'physics_adapter', 'pde_solver', 'optimizer', 'evaluator'},
        where='optimization'
    )
    from . import datasets, physics, models, evaluation

    # ----- load one example (CPU tensors) -----
    sample = datasets.torch.TorchDataset([ex])[0]
    mesh   = sample['mesh']
    unit_m = float(ex.metadata['unit'])
    affine = sample['affine']
    mask   = sample['mask'] # (1,I,J,K) bool

    # ----- build physics adapter -----
    physics_adapter_kws = config.get('physics_adapter', {})
    pde_solver_kws = config.get('pde_solver', {}).copy()
    pde_solver_cls = pde_solver_kws.pop('_class')

    physics_adapter = physics.PhysicsAdapter(
        pde_solver_cls=pde_solver_cls,
        pde_solver_kws=pde_solver_kws,
        **physics_adapter_kws
    )

    # ----- optimization targets -----
    opt_targets = config.get('targets', ['E'])
    utils.log(f'Targets: {opt_targets}')

    # ----- initialize param specs -----
    param_specs_cfg = config.get('param_specs', {})
    param_specs = {}
    for name in opt_targets:
        param_spec_kws = param_specs_cfg[name]
        param_specs[name] = ParameterSpec(**param_spec_kws)

    # ----- initialize free dofs -----
    z_vars = {}
    for name, spec in param_specs.items():
        z0 = physics_adapter.init_param_field(mesh, unit_m)
        z_vars[name] = torch.nn.Parameter(z0)

    # ----- optimizer config -----
    optimizer_kws = config.get('optimizer', {}).copy()
    optimizer_cls = getattr(torch.optim, optimizer_kws.pop('_class'))

    # ----- run optimization -----
    params = optimize_params_two_stage(
        physics_adapter=physics_adapter,
        mesh=mesh,
        unit_m=unit_m,
        bc_spec=None,
        z_vars=z_vars,
        param_specs=param_specs,
        optimizer_cls=optimizer_cls,
        optimizer_kws=optimizer_kws,
        global_steps=optimizer_kws.pop('global_steps', 10),
        local_steps=optimizer_kws.pop('local_steps', 100)
    )

    # ----- final forward sim -----
    loss, sim_output = physics_adapter.mesh_simulation_loss(
        mesh=mesh,
        unit_m=unit_m,
        params=params,
        bc_spec=None,
        ret_outputs=True
    )
    utils.log(f'Final loss: {loss.item()}')
    utils.pprint(sim_output)

    # ----- evaluation config -----
    evaluator_kws = config.get('evaluator', {})
    evaluator = evaluation.EvaluatorCallback(**evaluator_kws)

    outputs = {
        'example': [ex],
        'mask': mask.cpu().unsqueeze(0),
        'mat_true': sample['mat_true'].cpu().unsqueeze(0),
        'sim': [sim_output],
        'loss': loss
    }

    # ----- rasterize parameters -----
    utils.log('Rasterizing parameter fields')

    raster_base.mkdir(parents=True, exist_ok=True)
    shape = mask.shape[1:]

    for name in opt_targets:
        pred_key = f'{name}_pred'
        true_key = f'{name}_true'
        pred_vox = physics_adapter.rasterize_scalar_field(
            mesh, unit_m, params[name], shape, affine
        )
        outputs[pred_key] = pred_vox.cpu().unsqueeze(0)
        if true_key in sample:
            true_vox = sample[true_key]
            outputs[true_key] = true_vox.cpu().unsqueeze(0)

        raster_path = raster_base / f'{pred_key}.nii.gz'
        fileio.save_nibabel(raster_path, pred_vox[0].detach().cpu().numpy(), affine)

    evaluator.evaluate(epoch=0, phase='optimize', batch=0, step=0, outputs=outputs)
    evaluator.on_phase_end(epoch=0, phase='optimize')

    def _assign_mesh_field(m, name):
        m.point_data[name] = sim_output[name].nodes.numpy()
        m.cell_data[name] = [sim_output[name].cells.numpy()]

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

    print(output_mesh)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_meshio(output_path, output_mesh)


def optimize_params_two_stage(
    physics_adapter: PhysicsAdapter,
    mesh: meshio.Mesh,
    unit_m: float,
    bc_spec: Any,
    z_vars: Dictr[str, torch.nn.Parameter],
    param_specs: Dict[str, ParameterSpec],
    optimizer_cls,
    optimizer_kws,
    global_steps: int = 10,
    local_steps: int = 100,
):
    assert global_steps > 0 or local_steps > 0

    def decode_local() -> Dict[str, torch.Tensor]:
        return {k: v.decode(z_vars[k]) for k, v in param_specs.items()}

    def decode_global() -> Dict[str, torch.Tensor]:
        return {k: v.mean().expand(v.shape) for k, v in decode_local().items()}

    def objective(params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return physics_adapter.mesh_simulation_loss(
            mesh=mesh,
            unit_m=unit_m,
            params=params,
            bc_spec=None,
            ret_outputs=False
        )

    utils.log('Stage 1: Global optimization')
    optimizer_g = optimizer_cls(list(z_vars.values()), **optimizer_kws)

    def closure_g() -> torch.Tensor:
        optimizer_g.zero_grad(set_to_none=True)
        params = decode_global()
        loss = objective(params)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Invalid loss: {loss.item()}')
        loss.backward()
        return loss

    if global_steps > 0:
        optimize_closure(optimizer_g, closure_g, global_steps)

    utils.log('Stage 2: Local optimization')
    optimizer_l = optimizer_cls(list(z_vars.values()), **optimizer_kws)

    def closure_l() -> torch.Tensor:
        optimizer_l.zero_grad(set_to_none=True)
        params = decode_local()
        loss = objective(params)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Invalid loss: {loss.item()}')
        loss.backward()
        return loss

    if local_steps > 0:
        optimize_closure(optimizer_l, closure_l, local_steps)

    with torch.no_grad():
        params_final = {k: v.detach() for k, v in decode_local().items()}

    return params_final


def optimize_closure(optimizer, closure, max_steps=100):
    history = OptimizationHistory()
    params = optimizer.param_groups[0]['params']

    # main optimization loop
    loss = None
    for step in range(max_steps):
        loss = optimizer.step(closure)

        if loss is None:
            with torch.no_grad():
                loss = closure().detach()

        history.update(loss, params)
        if history.converged(step):
            utils.log('Optimization converged')
            break

    if loss is None:
        with torch.no_grad():
            loss = closure().detach()

    return loss


class OptimizationHistory:

    def __init__(self):
        self.loss_history  = []
        self.grad_history  = []
        self.param_history = []

        utils.log('iter\tloss (rel_delta)\tgrad_norm (rel_init)\tparam_norm (update_norm)')

    def update(self, loss, params):

        curr_loss = float(loss.detach().item())

        curr_grad = 0.0
        for p in params:
            if p.grad is not None:
                curr_grad += float(p.grad.pow(2).sum().cpu().item())
        curr_grad = np.sqrt(curr_grad)

        curr_param = np.concatenate([
            p.detach().cpu().numpy().ravel() for p in params
        ])

        self.loss_history.append(curr_loss)
        self.grad_history.append(curr_grad)
        self.param_history.append(curr_param)

    def converged(self, it, tol=1e-3, eps=1e-12):
        from numpy.linalg import norm
        loss_delta = grad_delta = param_delta = np.nan

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
            raise RuntimeExcepton('Optimization encountered nan value')

        return loss_delta < tol or grad_delta < tol or param_delta < tol

