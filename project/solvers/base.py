from __future__ import annotations
from typing import Dict, Tuple, Optional
import torch


class PDESolver:

    @classmethod
    def get_subclass(cls, name: str):
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
        u_obs: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        raise NotImplementedError

    def backward(self, loss_grad: torch.Tensor, context: Dict) -> Dict[str, torch.Tensor]:
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
        ctx.tensors = (mu, lam, rho, u_bc, u_obs)
        solver.zero_grad()
        outputs, ctx.context = solver.forward(mu, lam, rho, u_bc, u_obs)
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
        input_grads = ctx.solver.backward(loss_grad, ctx.context)
        mu, lam, rho, u_bc, u_obs = ctx.tensors
        return (
            None,
            _on_device(input_grads.get('mu'), mu.device),
            _on_device(input_grads.get('lam'), lam.device),
            _on_device(input_grads.get('rho'), rho.device),
            _on_device(input_grads.get('u_bc'), u_bc.device),
            _on_device(input_grads.get('u_obs'), u_obs.device)
        )


def _on_device(t, device):
    return t.to(device=device) if torch.is_tensor(t) else t

