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

    def bind_geometry(self, verts: torch.Tensor, cells: torch.Tensor):
        raise NotImplementedError

    def solve(
        self,
        mu: torch.Tensor,
        lam: torch.Tensor,
        rho: torch.Tensor,
        u_bc: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        mu: torch.Tensor,
        lam: torch.Tensor,
        rho: torch.Tensor,
        u_bc: torch.Tensor,
        u_obs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def backward(self, loss_grad: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def zero_grad(self):
        raise NotImplementedError

    def loss(self, mu, lam, rho, u_bc, u_obs):
        loss, res, u_sim = PDELossFn.apply(self, mu, lam, rho, u_bc, u_obs)
        return loss, {'res': res, 'u_sim': u_sim}


class PDELossFn(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        solver: PDESolver,
        mu: torch.Tensor,
        lam: torch.Tensor,
        rho: torch.Tensor,
        u_bc: torch.Tensor,
        u_obs: torch.Tensor
    ):
        ctx.solver = solver
        solver.zero_grad()
        outputs = solver.forward(mu, lam, rho, u_bc, u_obs)
        return (
            outputs['loss'],
            outputs['res'].detach(),
            outputs['u_sim'].detach()
        )

    @staticmethod
    def backward(
        ctx,
        loss_grad: torch.Tensor,
        res_grad: Optional[torch.Tensor]=None,
        u_sim_grad: Optional[torch.Tensor]=None
    ):
        input_grads = ctx.solver.backward(loss_grad)
        return (
            None,
            input_grads.get('mu'),
            input_grads.get('lam'),
            input_grads.get('rho'),
            input_grads.get('u_bc'),
            input_grads.get('u_obs')
        )

