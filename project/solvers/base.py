from __future__ import annotations

import torch


class PDESolver:

    def __init__(self, mesh: meshio.Mesh):
        raise NotImplementedError

    def set_observed(self, u_obs: torch.Tensor):
        raise NotImplementedError

    def set_density(self, rho: torch.Tensor):
        raise NotImplementedError

    def set_elasticity(self, mu: torch.Tensor, lam: torch.Tensor):
        raise NotImplementedError

    def assemble_projector(self):
        raise NotImplementedError

    def assemble_stiffness(self):
        raise NotImplementedError

    def assemble_forcing(self):
        raise NotImplementedError

    def assemble_lifting(self):
        raise NotImplementedError

    def assemble_rhs_shift(self):
        raise NotImplementedError

    def solve_forward(self):
        raise NotImplementedError

    def solve_adjoint(self):
        raise NotImplementedError

    def compute_residual(self):
        raise NotImplementedError

    def compute_loss(self):
        raise NotImplementedError

    def adjoint_forward(self):
        raise NotImplementedError

    def adjoint_backward(self, grad_out: torch.Tensor):
        raise NotImplementedError

    def params_grad(self):
        raise NotImplementedError

    def zero_grad(self):
        raise NotImplementedError

    def assemble_all(self):
        self.assemble_projector()
        self.assemble_forcing()
        self.assemble_lifting()
        self.assemble_stiffness()
        self.assemble_rhs_shift()


class PDESolverModule(torch.nn.Module):

    def __init__(self, solver: PDESolver, rho: torch.Tensor, u_obs: torch.Tensor):
        super().__init__()
        solver.assemble_projector()
        solver.set_density(rho)
        solver.set_observed(u_obs)
        solver.assemble_forcing()
        solver.assemble_lifting()
        self.solver = solver

    def forward(self, mu: torch.Tensor, lam: torch.Tensor):
        return PDESolverFn.apply(self.solver, mu, lam)


class PDESolverFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, solver: PDESolver, mu: torch.Tensor, lam: torch.Tensor):
        solver.zero_grad()
        solver.set_elasticity(mu, lam)
        solver.assemble_stiffness()
        solver.assemble_rhs_shift()
        solver.solve_forward()
        loss = solver.adjoint_forward()
        ctx.solver = solver
        return loss

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        ctx.solver.adjoint_backward(grad_out)
        grad = ctx.solver.params_grad()
        return None, grad['mu'], grad['lam']

