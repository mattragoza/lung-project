from __future__ import annotations
import numpy as np
import torch

from .core import utils, fileio
from .api import _check_keys
    

def optimize_example(ex, config):
    _check_keys(
        config,
        {'physics_adapter', 'pde_solver', 'parameters', 'optimizer', 'evaluator'},
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
    param_fn = models.get_output_fn(param_kws.pop('param_func'))
    param = physics_adapter.init_param_field(mesh, unit_m, **param_kws)
    param.requires_grad = True

    def fn_local(param):
        E = param_fn(param)
        loss, outputs = physics_adapter.simulation_loss(mesh, unit_m, E, bc_spec=None)
        return loss

    def fn_global(param):
        p_mean = param.mean().expand(param.shape)
        return fn_local(p_mean)

    optimizer_kws = config.get('optimizer', {}).copy()
    optimizer_cls = getattr(torch.optim, optimizer_kws.pop('_class'))
    optimizer = optimizer_cls([param], **optimizer_kws)

    optimize_fn(fn_global, param, optimizer)
    optimize_fn(fn_local,  param, optimizer)

    E = param_fn(param)
    loss, pde_outputs = physics_adapter.simulation_loss(mesh, unit_m, E, bc_spec=None)

    evaluator_kws = config.get('evaluator', {})
    evaluator = evaluation.Evaluator(**evaluator_kws)

    outputs = {'example': [ex], 'pde': [pde_outputs], 'loss': loss}
    evaluator.evaluate(outputs, epoch=0, phase='optimize', batch=0)
    result = evaluator.phase_end(epoch=0, phase='optimize')
    print(result.T)
    return result


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
        curr_loss  = float(loss.detach().item())
        curr_grad  = float(param.detach().norm().item())
        curr_param = param.detach().cpu().numpy()

        self.loss_history.append(curr_loss)
        self.grad_history.append(curr_grad)
        self.param_history.append(curr_param)

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

