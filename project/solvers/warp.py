import numpy as np
import torch
import warp as wp
import warp.fem
import warp.optim.linear


class WarpFEMSolver:

    def __init__(self, geometry, alpha=1e6, tol=1e-4, eps=1e-12):
        self.geometry = geometry
    
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

        self.r_field = self.V.make_field()
        self.r_field.dof_values.requires_grad = True

        self.mu_field  = self.S.make_field()
        self.lam_field = self.S.make_field()
        self.rho_field = self.S.make_field()

        self.g = wp.vec3f([0, 0, -9.81])
        self.I = wp.diag(wp.vec3f(1.0))

        # hyperparameters
        self.alpha = alpha
        self.tol = tol
        self.eps = eps

    def assign_fixed_values(self, rho, u_obs):
        self.rho_field.dof_values   = as_warp_array(rho, dtype=wp.float32, requires_grad=True)
        self.u_obs_field.dof_values = as_warp_array(u_obs, dtype=wp.vec3f, requires_grad=True)

    def assign_param_values(self, mu, lam):
        self.mu_field.dof_values  = as_warp_array(mu, dtype=wp.float32, requires_grad=True)
        self.lam_field.dof_values = as_warp_array(lam, dtype=wp.float32, requires_grad=True)

    def assemble_pde_operator(self):
        self.K_pde = wp.fem.integrate(
            pde_bilinear_form,
            fields={
                'u': self.u_trial,
                'v': self.v_test,
                'mu': self.mu_field,
                'lam': self.lam_field
            },
            domain=self.domain,
            output_dtype=wp.float32
        )

    def assemble_pde_rhs(self):
        self.f_pde = wp.fem.integrate(
            pde_linear_form,
            fields={'v': self.v_test, 'rho': self.rho_field},
            values={'g': self.g},
            domain=self.domain,
            output_dtype=wp.vec3f
        )

    def assemble_dbc_operator(self):
        self.K_bc = wp.fem.integrate(
            dbc_form,
            fields={'u': self.ub_trial, 'v': self.vb_test},
            values={'alpha': self.alpha},
            domain=self.boundary,
            output_dtype=wp.float32
        )

    def assemble_dbc_rhs(self):
        self.f_bc = wp.fem.integrate(
            dbc_form,
            fields={'u': self.u_obs_field.trace(), 'v': self.vb_test},
            values={'alpha': self.alpha},
            domain=self.boundary,
            output_dtype=wp.vec3f
        )

    def apply_boundary_condition(self):
        self.K = self.K_pde + self.K_bc
        self.f = self.f_pde + self.f_bc
        self.M = wp.optim.linear.preconditioner(self.K, ptype='diag')

    def solve_forward_system(self):
        return wp.optim.linear.cg(
            A=self.K,
            b=self.f,
            x=self.u_sim_field.dof_values,
            M=self.M,
            tol=self.tol
        )

    def solve_adjoint_system(self):
        return wp.optim.linear.cg(
            A=self.K,
            b=self.u_sim_field.dof_values.grad,
            x=self.r_field.dof_values.grad,
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
            values={'g': self.g},
            domain=self.domain,
            output=self.r_field.dof_values
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
            output=self.r_field.dof_values,
            add=True
        )

    def compute_error(self, relative=False):
        numer = wp.empty(1, dtype=wp.float32, requires_grad=True)
        denom = wp.empty(1, dtype=wp.float32, requires_grad=True)

        wp.fem.integrate(
            error_form,
            fields={'y_pred': self.u_sim_field, 'y_true': self.u_obs_field},
            domain=self.domain,
            output=numer
        )
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


def as_warp_array(t, **kwargs):
    return wp.from_torch(t.contiguous().detach(), **kwargs)


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
    g: wp.vec3
):
    lhs = pde_bilinear_form(s, u, v, mu, lam)
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


