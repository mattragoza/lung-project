from __future__ import annotations
from typing import Dict, Tuple, Optional
import meshio
import torch


class PDESolver:

    def set_geometry(self, mesh: meshio.Mesh):
        raise NotImplementedError

    def set_params(self, mu: torch.Tensor, lam: torch.Tensor):
        raise NotImplementedError

    def set_data(self, rho: torch.Tensor, u_obs: torch.Tensor):
        raise NotImplementedError

    def get_output(self) -> torch.Tensor:
        raise NotImplementedError

    def get_residual(self) -> torch.Tensor:
        raise NotImplementedError

    def get_loss(self) -> torch.Tensor:
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

    def to(self, device):
        raise NotImplementedError


class PDESolverModule(torch.nn.Module):

    def __init__(self, solver: PDESolver, rho: torch.Tensor, u_obs: torch.Tensor):
        super().__init__()
        self.solver = solver
        self.solver.adjoint_setup(rho, u_obs)

    def forward(self, mu: torch.Tensor, lam: torch.Tensor):
        return PDESolverFn.apply(self.solver, mu, lam)

    def to(self, device):
        self.solver.to(device)


class PDESolverFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, solver: PDESolver, mu: torch.Tensor, lam: torch.Tensor):
        ctx.solver = solver
        solver.zero_grad()
        loss = solver.adjoint_forward(mu, lam)
        return loss

    @staticmethod
    def backward(ctx, loss_grad: torch.Tensor):
        mu_grad, lam_grad = ctx.solver.adjoint_backward(loss_grad)
        return None, mu_grad, lam_grad

