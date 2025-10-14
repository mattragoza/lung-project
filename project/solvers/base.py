import meshio
import torch


class PDESolver:

	def __init__(self, mesh: meshio.Mesh):
		raise NotImplementedError

	def set_fixed(self, rho: torch.Tensor, u_obs: torch.Tensor):
		raise NotImplementedError

	def set_params(self, mu: torch.Tensor, lam: torch.Tensor):
		raise NotImplementedError

	def assemble_K(self):
		raise NotImplementedError

	def assemble_f(self):
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


class PDESolverFn(torch.autograd.Function):

	@staticmethod
	def forward(
		ctx,
		solver: PDESolver,
		mu: torch.Tensor,
		lam: torch.Tensor
	):
		solver.zero_grad()
		solver.set_params(mu, lam)
		solver.assemble_K()
		solver.solve_forward()
		loss = solver.adjoint_forward()
		ctx.solver = solver
		return loss

	@staticmethod
	def backward(ctx, grad_out):
		ctx.solver.adjoint_backward(grad_out)
		grad = ctx.solver.params_grad()
		return None, grad['mu'], grad['lam']


class PDESolverModule(torch.nn.Module):

	def __init__(self, solver: PDESolver):
		super().__init__()
		self.solver = solver

	def forward(self, mu, lam, relative=False):
		return PDESolverFn.apply(self.solver, mu, lam)

