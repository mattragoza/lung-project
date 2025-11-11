from __future__ import annotations
from typing import Dict, Tuple, Optional
import meshio
import torch


class PDESolver:

    def init_geometry(self, verts: torch.Tensor, cells: torch.Tensor):
        raise NotImplementedError

    def simulate(self, mu, lam, rho, u_obs) -> torch.Tensor:
        raise NotImplementedError

    def adjoint_setup(self, rho, u_obs):
        raise NotImplementedError

    def adjoint_forward(self, mu, lam) -> torch.Tensor:
        raise NotImplementedError

    def adjoint_backward(self, loss_grad) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def zero_grad(self):
        raise NotImplementedError


class PDESolverModule(torch.nn.Module):

    def __init__(
        self,
        solver: PDESolver, 
        verts: torch.Tensor,
        cells: torch.Tensor,
        rho: torch.Tensor,
        u_obs: torch.Tensor
    ):
        super().__init__()
        self.solver = solver
        self.solver.init_geometry(verts, cells)
        self.solver.adjoint_setup(rho, u_obs)

    def forward(self, mu: torch.Tensor, lam: torch.Tensor):
        return PDESolverFn.apply(self.solver, mu, lam)


class PDESolverFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, solver: PDESolver, mu: torch.Tensor, lam: torch.Tensor):
        ctx.solver = solver
        solver.zero_grad()
        u_sim, res, loss = solver.adjoint_forward(mu, lam)
        return loss, res.detach(), u_sim.detach()

    @staticmethod
    def backward(ctx, grad_loss, grad_res=None, grad_u=None):
        mu_grad, lam_grad = ctx.solver.adjoint_backward(grad_loss)
        return None, mu_grad, lam_grad

