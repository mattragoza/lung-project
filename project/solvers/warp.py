import numpy as np
import torch
import warp as wp
import warp.fem
import warp.optim.linear
import meshio

from . import base


def _as_warp_array(t, **kwargs):
    return wp.from_torch(t.contiguous().detach(), **kwargs)


def _maybe_zero_grad(a: wp.array):
    if getattr(a, 'grad', None) is not None:
        a.grad.zero_()


class WarpSolver(base.PDESolver):

    def __init__(
        self,
        mesh: meshio.Mesh,
        alpha=1e6,
        tol=1e-4,
        eps=1e-12,
        relative=False
    ):
        geometry = wp.fem.Tetmesh(
            wp.array(mesh.cells_dict['tetra'], dtype=wp.int32),
            wp.array(mesh.points * 1e-3, dtype=wp.vec3f) # mm -> m
        )
    
        # integration domains
        self.domain = wp.fem.Cells(geometry)
        self.boundary = wp.fem.BoundarySides(geometry)

        # function spaces
        self.S = wp.fem.make_polynomial_space(geometry, degree=1, dtype=wp.float32)
        self.V = wp.fem.make_polynomial_space(geometry, degree=1, dtype=wp.vec3f)

        # trial and test functions
        self.u_trial = wp.fem.make_trial(self.V, domain=self.domain)
        self.v_test  = wp.fem.make_test(self.V, domain=self.domain)

        self.ub_trial = wp.fem.make_trial(self.V, domain=self.boundary)
        self.vb_test  = wp.fem.make_test(self.V, domain=self.boundary)

        # physical fields and constants
        self.u_obs_field = self.V.make_field()
        self.u_sim_field = self.V.make_field()
        self.u_sim_field.dof_values.requires_grad = True

        self.res_field = self.V.make_field()
        self.res_field.dof_values.requires_grad = True

        self.mu_field  = self.S.make_field()
        self.lam_field = self.S.make_field()
        self.rho_field = self.S.make_field()

        self.g = wp.vec3f([0, 0, -9.81]) # m/s^2
        self.I = wp.diag(wp.vec3f(1.0))

        # hyperparameters
        self.alpha = alpha
        self.tol = tol
        self.eps = eps
        self.relative = relative

    def set_fixed(self, rho, u_obs):
        self.rho_field.dof_values   = _as_warp_array(rho, dtype=wp.float32, requires_grad=False)
        self.u_obs_field.dof_values = _as_warp_array(u_obs, dtype=wp.vec3f, requires_grad=False)

    def set_params(self, mu, lam):
        self.mu_field.dof_values  = _as_warp_array(mu, dtype=wp.float32, requires_grad=True)
        self.lam_field.dof_values = _as_warp_array(lam, dtype=wp.float32, requires_grad=True)

    def assemble_K(self):
        self.K = wp.fem.integrate(
            pde_bilinear_form,
            fields={
                'u': self.u_trial,
                'v': self.v_test,
                'mu': self.mu_field,
                'lam': self.lam_field
            },
            values={'I': self.I},
            domain=self.domain,
            output_dtype=wp.float32
        )
        wp.fem.integrate(
            dbc_form,
            fields={'u': self.ub_trial, 'v': self.vb_test},
            values={'alpha': self.alpha},
            domain=self.boundary,
            output=self.K,
            add=True
        )
        self.M = wp.optim.linear.preconditioner(self.K, ptype='diag')

    def assemble_f(self):
        self.f = wp.fem.integrate(
            pde_linear_form,
            fields={'v': self.v_test, 'rho': self.rho_field},
            values={'g': self.g},
            domain=self.domain,
            output_dtype=wp.vec3f
        )
        wp.fem.integrate(
            dbc_form,
            fields={'u': self.u_obs_field.trace(), 'v': self.vb_test},
            values={'alpha': self.alpha},
            domain=self.boundary,
            output=self.f,
            add=True
        )

    def solve_forward(self):
        return wp.optim.linear.cg(
            A=self.K,
            b=self.f,
            x=self.u_sim_field.dof_values,
            M=self.M,
            tol=self.tol
        )

    def solve_adjoint(self):
        return wp.optim.linear.cg(
            A=self.K,
            b=self.u_sim_field.dof_values.grad,
            x=self.res_field.dof_values.grad,
            M=self.M,
            tol=self.tol
        )

    def compute_residual(self):
        wp.fem.integrate(
            pde_residual_form,
            fields={
                'u': self.u_sim_field,
                'v': self.v_test,
                'mu': self.mu_field,
                'lam': self.lam_field,
                'rho': self.rho_field
            },
            values={'g': self.g, 'I': self.I},
            domain=self.domain,
            output=self.res_field.dof_values
        )
        wp.fem.integrate(
            dbc_residual_form,
            fields={
                'u_sim': self.u_sim_field.trace(),
                'u_obs': self.u_obs_field.trace(),
                'v': self.vb_test
            },
            values={'alpha': self.alpha},
            domain=self.boundary,
            output=self.res_field.dof_values,
            add=True
        )

    def compute_loss(self, relative=None):
        numer = wp.empty(1, dtype=wp.float32, requires_grad=True)
        denom = wp.empty(1, dtype=wp.float32, requires_grad=True)
        wp.fem.integrate(
            error_form,
            fields={'y_pred': self.u_sim_field, 'y_true': self.u_obs_field},
            domain=self.domain,
            output=numer
        )
        if relative is None:
            relative = self.relative
        if relative:
            wp.fem.integrate(
                norm_form,
                fields={'y_true': self.u_obs_field},
                domain=self.domain,
                output=denom
            )
        else:
            wp.fem.integrate(
                volume_form,
                domain=self.domain,
                output=denom
            )

        return numer / (denom + self.eps)

    def adjoint_forward(self):
        self.tape = wp.Tape()
        with self.tape:
            self.compute_residual()

        self.tape.record_func(
            backward=self.solve_adjoint,
            arrays=[self.res_field.dof_values, self.u_sim_field.dof_values]
        )
        with self.tape:
            self.loss = self.compute_loss()
            self.loss.requires_grad = True

        return wp.to_torch(self.loss)

    def adjoint_backward(self, grad_out):
        self.loss.grad = wp.from_torch(grad_out)
        self.tape.backward()

    def params_grad(self):
        return {
            'mu':  wp.to_torch(self.mu_field.dof_values.grad),
            'lam': wp.to_torch(self.lam_field.dof_values.grad)
        }

    def zero_grad(self):
        _maybe_zero_grad(self.mu_field.dof_values)
        _maybe_zero_grad(self.lam_field.dof_values)
        _maybe_zero_grad(self.rho_field.dof_values)
        _maybe_zero_grad(self.u_obs_field.dof_values)
        _maybe_zero_grad(self.u_sim_field.dof_values)
        _maybe_zero_grad(self.res_field.dof_values)
        if getattr(self, 'loss', None) is not None:
            _maybe_zero_grad(self.loss)


@wp.fem.integrand
def pde_bilinear_form(
    s: wp.fem.Sample,
    u: wp.fem.Field,
    v: wp.fem.Field,
    mu: wp.fem.Field,
    lam: wp.fem.Field,
    I: wp.mat33
):
    eps_u = wp.fem.D(u, s) # symmetric gradient
    eps_v = wp.fem.D(v, s)
    div_u = wp.fem.div(u, s)
    sigma_u = 2.0*mu(s)*eps_u + lam(s)*div_u*I
    return wp.ddot(sigma_u, eps_v)


@wp.fem.integrand
def pde_linear_form(
    s: wp.fem.Sample,
    v: wp.fem.Field,
    rho: wp.fem.Field,
    g: wp.vec3
):
    return rho(s) * wp.dot(g, v(s))


@wp.fem.integrand
def pde_residual_form(
    s: wp.fem.Sample,
    u: wp.fem.Field,
    v: wp.fem.Field,
    mu: wp.fem.Field,
    lam: wp.fem.Field,
    rho: wp.fem.Field,
    g: wp.vec3,
    I: wp.mat33
):
    lhs = pde_bilinear_form(s, u, v, mu, lam, I)
    rhs = pde_linear_form(s, v, rho, g)
    return rhs - lhs


@wp.fem.integrand
def dbc_form(s: wp.fem.Sample, u: wp.fem.Field, v: wp.fem.Field, alpha: float):
    return alpha * wp.dot(u(s), v(s))


@wp.fem.integrand
def dbc_residual_form(
    s: wp.fem.Sample,
    u_sim: wp.fem.Field,
    u_obs: wp.fem.Field,
    v: wp.fem.Field,
    alpha: float
):
    lhs = dbc_form(s, u_sim, v, alpha)
    rhs = dbc_form(s, u_obs, v, alpha)
    return rhs - lhs


@wp.fem.integrand
def error_form(s: wp.fem.Sample, y_pred: wp.fem.Field, y_true: wp.fem.Field):
    error = y_pred(s) - y_true(s)
    return wp.dot(error, error)


@wp.fem.integrand
def norm_form(s: wp.fem.Sample, y_true: wp.fem.Field):
    y_s = y_true(s)
    return wp.dot(y_s, y_s)


@wp.fem.integrand
def volume_form(s: wp.fem.Sample):
    return 1.0


