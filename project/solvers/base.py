from __future__ import annotations
from typing import Dict, Tuple, Optional
import torch


class PDESolver:

    @classmethod
    def get_subclass(cls, name):
        if name in {'warp', 'warp.fem', 'WarpFEMSolver'}:
            from . import warp
            return warp.WarpFEMSolver
        elif name in {'fenics', 'dolfin', 'FenicsFEMSolver'}:
            from . import fenics
            return fenics.FenicsFEMSolver
        raise ValueError(f'Invalid solver class: {name}')

    def init_geometry(self, verts: torch.Tensor, cells: torch.Tensor):
        raise NotImplementedError

    def forward(
        self,
        mu: torch.Tensor,
        lam: torch.Tensor,
        rho: torch.Tensor,
        u_obs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def backward(self, loss_grad: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def zero_grad(self):
        raise NotImplementedError

    def solve(self, mu, lam, rho, u_obs):
        loss, res, u_sim = PDESolveFn.apply(self, mu, lam, rho, u_obs)
        return {'loss': loss, 'res': res, 'u_sim': u_sim}


class PDESolveFn(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        solver: PDESolver,
        mu: torch.Tensor,
        lam: torch.Tensor,
        rho: torch.Tensor,
        u_obs: torch.Tensor
    ):
        ctx.solver = solver
        solver.zero_grad()
        outputs = solver.forward(mu, lam, rho, u_obs)
        return (
            outputs['loss'],
            outputs['res'].detach(),
            outputs['u_sim'].detach()
        )

    @staticmethod
    def backward(ctx, grad_loss, grad_res=None, grad_u=None):
        grad_inputs = ctx.solver.backward(grad_loss)
        return (
            None,
            grad_inputs['mu'],
            grad_inputs['lam'],
            grad_inputs['rho'],
            grad_inputs['u_obs']
        )

