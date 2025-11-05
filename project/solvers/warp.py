from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np
import torch
import meshio

import warp as wp
import warp.fem
import warp.optim.linear

from . import base


def _as_warp_array(t, **kwargs):
    return wp.from_torch(t.contiguous().detach(), **kwargs)


def _maybe_zero_grad(a: wp.array):
    if getattr(a, 'grad', None) is not None:
        a.grad.zero_()


class WarpFEMSolver(base.PDESolver):

    def __init__(
        self,
        mesh: meshio.Mesh,
        reg_weight: float=0.0, # 5e-2,
        eps_reg: float=None,
        eps_div: float=1e-12,
        cg_tol: float=1e-5,
        unit_m: float=1e-3,
        relative_loss=True
    ):
        wp.init()

        # convert mesh units to meters
        self.geometry = wp.fem.Tetmesh(
            wp.array(mesh.cells_dict['tetra'], dtype=wp.int32),
            wp.array(mesh.points * unit_m, dtype=wp.vec3f)
        )
        self.geometry.build_bvh()
    
        # integration domains
        self.domain = wp.fem.Cells(self.geometry)
        self.boundary = wp.fem.BoundarySides(self.geometry)

        # function spaces
        self.S = wp.fem.make_polynomial_space(self.geometry, degree=1, dtype=wp.float32)
        self.V = wp.fem.make_polynomial_space(self.geometry, degree=1, dtype=wp.vec3f)

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
        self.relative_loss = relative_loss
        self.reg_weight = reg_weight
        self.eps_reg = eps_reg or 1. / unit_m
        self.eps_div = eps_div
        self.cg_tol = cg_tol

    # --- public interface methods ---

    def set_params(self, mu: torch.Tensor, lam: torch.Tensor):
        self.mu_field.dof_values = _as_warp_array(mu, dtype=wp.float32, requires_grad=True)
        self.lam_field.dof_values = _as_warp_array(lam, dtype=wp.float32, requires_grad=True)
        self.K = self.M = self.f_tilde = None

    def set_data(self, rho: torch.Tensor, u_obs: torch.Tensor):
        self.rho_field.dof_values = _as_warp_array(rho, dtype=wp.float32, requires_grad=False)
        self.u_obs_field.dof_values = _as_warp_array(u_obs, dtype=wp.vec3f, requires_grad=False)
        self.u0 = self.f = self.f_tilde = None

    def get_output(self) -> torch.Tensor:
        return wp.to_torch(self.u_sim_field.dof_values)

    def get_residual(self) -> torch.Tensor:
        return wp.to_torch(self.res_field.dof_values)

    def get_loss(self) -> torch.Tensor:
        return wp.to_torch(self.loss)

    def simulate(self, mu, lam, rho, u_obs) -> torch.Tensor:
        self.set_data(rho, u_obs)
        self.assemble_projector()
        self.assemble_forcing()
        self.assemble_lifting()

        self.set_params(mu, lam)
        self.assemble_stiffness()
        self.assemble_rhs_shift()
        self.solve_forward()

        return self.get_output()

    def adjoint_setup(self, rho, u_obs):
        self.set_data(rho, u_obs)
        self.assemble_projector()
        self.assemble_forcing()
        self.assemble_lifting()

    def adjoint_forward(self, mu, lam):
        self.set_params(mu, lam)
        self.assemble_stiffness()
        self.assemble_rhs_shift()
        self.solve_forward()

        self.tape = wp.Tape()
        with self.tape:
            self.compute_residual()

        self.tape.record_func(
            backward=self.solve_adjoint,
            arrays=[self.res_field.dof_values, self.u_sim_field.dof_values]
        )
        with self.tape:
            self.compute_loss()

        return self.get_loss()

    def adjoint_backward(self, loss_grad):
        self.loss.grad = wp.from_torch(loss_grad)
        self.tape.backward()
        mu_grad = wp.to_torch(self.mu_field.dof_values.grad)
        lam_grad = wp.to_torch(self.lam_field.dof_values.grad)
        return mu_grad, lam_grad

    def zero_grad(self):
        _maybe_zero_grad(self.u_obs_field.dof_values)
        _maybe_zero_grad(self.rho_field.dof_values)
        _maybe_zero_grad(self.mu_field.dof_values)
        _maybe_zero_grad(self.lam_field.dof_values)
        _maybe_zero_grad(self.u_sim_field.dof_values)
        _maybe_zero_grad(self.res_field.dof_values)
        if getattr(self, 'loss', None) is not None:
            _maybe_zero_grad(self.loss)

    # --- internal assembly methods ---

    def assemble_projector(self):
        self.P = wp.fem.integrate(
            inner_form,
            fields={'u': self.ub_trial, 'v': self.vb_test},
            domain=self.boundary,
            assembly='nodal',
            output_dtype=wp.float32
        )
        wp.fem.normalize_dirichlet_projector(self.P)
        self.u0 = self.f_tilde = None

    def assemble_stiffness(self):
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
        self.M = wp.optim.linear.preconditioner(self.K, ptype='diag')
        self.f_tilde = None

    def assemble_forcing(self):
        self.f = wp.fem.integrate(
            pde_linear_form,
            fields={'v': self.v_test, 'rho': self.rho_field},
            values={'g': self.g},
            domain=self.domain,
            output_dtype=wp.vec3f
        )
        self.f_tilde = None

    def assemble_lifting(self):
        self.u0 = self.P @ self.u_obs_field.dof_values
        self.f_tilde = None

    def assemble_rhs_shift(self):
        self.f_tilde = self.f - (self.K @ self.u0)

    # --- solving linear systems ---

    def solve_forward(self):
        wp.fem.project_linear_system(
            self.K,
            self.f_tilde,
            self.P,
            normalize_projector=False
        )
        it, res, tol = wp.optim.linear.cg(
            A=self.K,
            b=self.f_tilde,
            x=self.u_sim_field.dof_values,
            M=self.M,
            tol=self.cg_tol
        )
        self.u_sim_field.dof_values += self.u0

        self.forward_it  = it
        self.forward_res = res
        self.forward_tol = tol

    def solve_adjoint(self):
        wp.fem.project_linear_system(
            self.K,
            self.u_sim_field.dof_values.grad,
            self.P,
            normalize_projector=False
        )
        it, res, tol = wp.optim.linear.cg(
            A=self.K,
            b=self.u_sim_field.dof_values.grad,
            x=self.res_field.dof_values.grad,
            M=self.M,
            tol=self.cg_tol
        )
        self.u_sim_field.dof_values.grad.zero_()

        self.adjoint_it  = it
        self.adjoint_res = res
        self.adjoint_tol = tol

    # --- residual and loss evaluation ---

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

    def compute_loss(self):
        numer = wp.empty(1, dtype=wp.float32, requires_grad=True)
        denom = wp.empty(1, dtype=wp.float32, requires_grad=True)
        wp.fem.integrate(
            error_form,
            fields={'u': self.u_sim_field, 'v': self.u_obs_field},
            domain=self.domain,
            output=numer
        )
        if self.relative_loss:
            wp.fem.integrate(
                norm2_form,
                fields={'u': self.u_obs_field},
                domain=self.domain,
                output=denom
            )
        else:
            wp.fem.integrate(volume_form, domain=self.domain, output=denom)

        data_term = numer / (denom + self.eps_div)
        reg_term = wp.empty(1, dtype=wp.float32, requires_grad=True)
        wp.fem.integrate(
            tv_reg_form,
            fields={'mu': self.mu_field, 'lam': self.lam_field},
            values={'eps_reg': self.eps_reg, 'eps_div': self.eps_div},
            domain=self.domain,
            output=reg_term
        )
        self.loss = data_term + self.reg_weight * reg_term
        self.loss.requires_grad = True


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
def inner_form(s: wp.fem.Sample, u: wp.fem.Field, v: wp.fem.Field):
    return wp.dot(u(s), v(s))


@wp.fem.integrand
def error_form(s: wp.fem.Sample, u: wp.fem.Field, v: wp.fem.Field):
    e_s = u(s) - v(s)
    return wp.dot(e_s, e_s)


@wp.fem.integrand
def norm2_form(s: wp.fem.Sample, u: wp.fem.Field):
    u_s = u(s)
    return wp.dot(u_s, u_s)


@wp.fem.integrand
def volume_form(s: wp.fem.Sample):
    return 1.0


@wp.fem.integrand
def tv_reg_form(
    s: wp.fem.Sample,
    mu: wp.fem.Field,
    lam: wp.fem.Field,
    eps_reg: float,
    eps_div: float,
):
    # TV regularization on phi = log E
    # grad phi = grad (log E)
    # grad phi = (grad E) / E
    # grad phi = (grad (k mu)) / (k mu) (for constant nu)
    # grad phi = (grad mu) / mu
    grad_phi = wp.fem.grad(mu, s) / (mu(s) + eps_div)
    return wp.sqrt(wp.dot(grad_phi, grad_phi) + eps_reg * eps_reg)


def rasterize_field(src: wp.fem.Field, shape, bounds, background=0.0):
    '''
    Args:
        src: Warp FEM field to rasterize on voxel grid
        shape: (I, J, K) voxel grid shape
        bounds: Lower and upper grid bounds
    Returns:
        (I, J, K, C) rasterized field tensor
    '''
    I, J, K = shape
    C = wp.types.type_length(src.dtype)

    grid = wp.fem.Grid3D(
        res=wp.vec3i(shape),
        bounds_lo=wp.vec3f(bounds[0]),
        bounds_hi=wp.vec3f(bounds[1])
    )
    dst_domain = wp.fem.Cells(grid)

    space = wp.fem.make_polynomial_space(grid, degree=0, dtype=src.dtype)
    dst = space.make_field()

    src_nc = wp.fem.NonconformingField(dst_domain, src, background)
    wp.fem.interpolate(src_nc, dest=dst)

    return wp.to_torch(dst.dof_values).reshape(I, J, K, C)

