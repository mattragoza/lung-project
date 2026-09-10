# physics/solvers.py

from typing import List, Dict, Tuple, Optional, Any

import torch


def _resolve_solver_name(name: str):

    if name in {'warp', 'warp.fem', 'WarpFEMSolver'}:
        from . import warp
        return warp.solver.WarpFEMSolver

    elif name in {'fenics', 'dolfin', 'FenicsFEMSolver'}:
        from . import fenics
        return fenics.FenicsFEMSolver

    raise ValueError(f'Invalid solver name: {name!r}')


class PDESolver:

    @classmethod
    def get_subclass(cls, name: str):
        return _resolve_solver_name(name)

    def bind_geometry(self, verts: torch.Tensor, cells: torch.Tensor):
        raise NotImplementedError

    def solve_forward(
        self,
        mu: torch.Tensor,
        lam: torch.Tensor,
        rho: torch.Tensor,
        u_bc: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def loss_forward(
        self,
        mu: torch.Tensor,
        lam: torch.Tensor,
        rho: torch.Tensor,
        u_bc: torch.Tensor,
        u_obs: torch.Tensor
    ) -> Tuple[dict, dict]:
        raise NotImplementedError

    def loss_backward(self, loss_grad: torch.Tensor, context: dict) -> dict:
        raise NotImplementedError

    def simulate_loss(self, mu, lam, rho, u_bc, u_obs, mask) -> dict:
        return PDELossFn.apply(self, mu, lam, rho, u_bc, u_obs, mask)


class PDELossFn(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: Any,
        solver: PDESolver,
        mu: torch.Tensor,
        lam: torch.Tensor,
        rho: torch.Tensor,
        u_bc: torch.Tensor,
        u_obs: torch.Tensor,
        mask: torch.Tensor
    ):
        ctx.solver = solver
        ctx.inputs = (mu, lam, rho, u_bc, u_obs)
        outputs, ctx.context = solver.loss_forward(mu, lam, rho, u_bc, u_obs, mask)
        return outputs['loss'], outputs['u_sim'], outputs['residual']

    @staticmethod
    def backward(
        ctx: Any,
        loss_grad: torch.Tensor,
        res_grad: Optional[torch.Tensor] = None,
        u_sim_grad: Optional[torch.Tensor] = None
    ):
        input_grads = ctx.solver.loss_backward(loss_grad, ctx.context)
        mu, lam, rho, u_bc, u_obs = ctx.inputs
        return (
            None,
            _on_device(input_grads.get('mu'), mu.device),
            _on_device(input_grads.get('lam'), lam.device),
            _on_device(input_grads.get('rho'), rho.device),
            _on_device(input_grads.get('u_bc'), u_bc.device),
            _on_device(input_grads.get('u_obs'), u_obs.device),
            None
        )


def _on_device(t, device):
    return t.to(device=device) if torch.is_tensor(t) else t

