from __future__ import annotations
import numpy as np
import torch

from .core import utils, fileio
    

def optimize_example(ex, config, output_path):
    utils.check_keys(
        config,
        valid={'physics_adapter', 'pde_solver', 'parameters', 'optimizer', 'evaluator'},
        where='optimization'
    )
    from . import physics, models, evaluation

    mesh_path = ex.paths['interp_mesh']
    unit_m = float(ex.metadata['unit'])
    mesh = fileio.load_meshio(mesh_path)

    physics_adapter_kws = config.get('physics_adapter', {})
    pde_solver_kws = config.get('pde_solver', {}).copy()

    physics_adapter = physics.PhysicsAdapter(
        pde_solver_cls=pde_solver_kws.pop('_class'),
        pde_solver_kws=pde_solver_kws,
        **physics_adapter_kws
    )
    param_kws = config.get('parameters', {}).copy()
    init_value = param_kws.pop('init_value')
    param_map = models.ParameterMap(**param_kws)
    param = physics_adapter.init_param_field(mesh, unit_m, init_value)

    def fn_local(x):
        E_pred = param_map(x)
        loss = physics_adapter.simulation_loss(mesh, unit_m, E_pred, bc_spec=None, ret_outputs=False)
        return loss

    def fn_global(x):
        return fn_local(x.mean().expand(x.shape))

    optimizer_kws = config.get('optimizer', {}).copy()
    optimizer_cls = getattr(torch.optim, optimizer_kws.pop('_class'))

    optimizer = optimizer_cls([param], **optimizer_kws)
    optimize_fn(fn_global, param, optimizer)

    optimizer = optimizer_cls([param], **optimizer_kws)
    optimize_fn(fn_local,  param, optimizer)

    E_pred = param_fn(param)
    loss, pde_outputs = physics_adapter.simulation_loss(mesh, unit_m, E_pred, bc_spec=None, ret_outputs=True)

    utils.pprint(pde_outputs)

    evaluator_kws = config.get('evaluator', {})
    evaluator = evaluation.Evaluator(**evaluator_kws)

    outputs = {'example': [ex], 'pde': [pde_outputs], 'loss': loss}
    evaluator.evaluate(outputs, epoch=0, phase='optimize', batch=0)
    metrics = evaluator.phase_end(epoch=0, phase='optimize')
    metrics['dataset'] = ex.dataset
    metrics['variant'] = ex.variant
    metrics['subject'] = ex.subject
    metrics['method'] = 'optimize'
    print(metrics.T)

    def _assign_mesh_field(m, name):
        m.point_data[name] = pde_outputs[name].nodes.numpy()
        m.cell_data[name] = [pde_outputs[name].cells.numpy()]

    output_mesh = mesh.copy()
    _assign_mesh_field(output_mesh, 'rho_true')
    _assign_mesh_field(output_mesh, 'rho_pred')
    _assign_mesh_field(output_mesh, 'E_true')
    _assign_mesh_field(output_mesh, 'E_pred')
    _assign_mesh_field(output_mesh, 'u_true')
    _assign_mesh_field(output_mesh, 'u_pred')
    _assign_mesh_field(output_mesh, 'residual')
    print(output_mesh)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_meshio(output_path, output_mesh)

    return metrics


def optimize_fn(fn, param, optimizer, max_iter=100):
    history = OptimizationHistory()

    def closure():
        optimizer.zero_grad()
        loss = fn(param)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Invalid loss: {loss.item()}')
        loss.backward()
        if not torch.isfinite(param.grad).all():
            raise RuntimeError(f'Invalid gradient: {g.detach().cpu().numpy()}')
        return loss

    for it in range(max_iter):
        loss = optimizer.step(closure)
        history.update(loss, param)
        if history.converged(it):
            utils.log('Optimization converged')
            break

    return param


class OptimizationHistory:

    def __init__(self):
        self.loss_history  = []
        self.grad_history  = []
        self.param_history = []

        utils.log('iter\tloss (rel_delta)\tgrad_norm (rel_init)\tparam_norm (update_norm)')

    def update(self, loss, param):
        self.loss_history.append(float(loss.detach().item()))
        self.grad_history.append(float(param.grad.detach().norm().item()))
        self.param_history.append(param.detach().cpu().numpy())

    def converged(self, it, tol=1e-3, eps=1e-12):
        from numpy.linalg import norm
        loss_delta = grad_delta = param_delta = np.nan

        curr_loss = self.loss_history[-1]
        curr_grad = self.grad_history[-1]
        curr_param = self.param_history[-1]
        curr_norm  = np.linalg.norm(curr_param)

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

