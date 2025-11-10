from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np
import torch
import meshio

import warp as wp
import warp.fem
import warp.optim.linear

wp.init()
wp.config.quiet = False

from . import base
from ..core import utils

DEBUG = False


def _as_warp_array(t, **kwargs):
    return wp.from_torch(t.contiguous().detach(), **kwargs)


def _zero_grad(a: wp.array):
    if getattr(a, 'grad', None) is not None:
        a.grad.zero_()


def _validate_scalar_field(vals: torch.Tensor, positive=False):
    if not vals.ndim == 1:
        raise ValueError(f'Invalid scalar field shape: {vals.shape}')
    if not torch.isfinite(vals).all():
        raise ValueError('Non-finite scalar field')
    if positive and not (vals > 0).all():
        raise ValueError('Non-positive scalar field')


def _validate_vector_field(vals: torch.Tensor):
    if not (vals.ndim == 2 and vals.shape[1] == 3):
        raise ValueError(f'Invalid vector field shape: {vals.shape}')
    if not torch.isfinite(vals).all():
        raise ValueError('Non-finite vector field')


class WarpFEMSolver(base.PDESolver):

    def __init__(
        self,
        relative_loss: bool=True,
        tv_reg_weight: float=0.0, # 1e-4 to 5e-2,
        eps_reg: float=1e-3,
        eps_div: float=1e-12,
        cg_tol: float=1e-5,
        scalar_degree: int=1,
        vector_degree: int=1,
        scalar_dtype=wp.float32,
        vector_dtype=wp.vec3f,
    ):
        self.relative_loss = relative_loss
        self.tv_reg_weight = tv_reg_weight
        self.eps_reg = eps_reg
        self.eps_div = eps_div
        self.cg_tol = cg_tol

        self.scalar_degree = scalar_degree
        self.vector_degree = vector_degree
        self.scalar_dtype = scalar_dtype
        self.vector_dtype = vector_dtype

    def init_geometry(self, verts: torch.Tensor, cells: torch.Tensor):

        verts = _as_warp_array(verts, dtype=self.vector_dtype)
        cells = _as_warp_array(cells, dtype=wp.int32)

        self.geometry = wp.fem.Tetmesh(cells, verts, build_bvh=True)
        self.interior = wp.fem.Cells(self.geometry)
        self.boundary = wp.fem.BoundarySides(self.geometry)

        # function spaces
        self.S = wp.fem.make_polynomial_space(self.geometry, degree=self.scalar_degree, dtype=self.scalar_dtype)
        self.V = wp.fem.make_polynomial_space(self.geometry, degree=self.vector_degree, dtype=self.vector_dtype)

        # trial and test functions
        self.u_trial  = wp.fem.make_trial(self.V, domain=self.interior)
        self.v_test   = wp.fem.make_test(self.V,  domain=self.interior)
        self.ub_trial = wp.fem.make_trial(self.V, domain=self.boundary)
        self.vb_test  = wp.fem.make_test(self.V,  domain=self.boundary)

        # physical fields and constants (dof_values created on same device as verts/cells)
        self.mu = self.S.make_field()
        self.lam = self.S.make_field()
        self.rho = self.S.make_field()

        self.u_obs = self.V.make_field()
        self.u_sim = self.V.make_field()
        self.res = self.V.make_field()

        self.mu.dof_values.requires_grad = True
        self.lam.dof_values.requires_grad = True
        self.u_sim.dof_values.requires_grad = True
        self.res.dof_values.requires_grad = True

        self.g = self.vector_dtype([0, 0, -9.81]) # m/s^2
        self.I = wp.diag(self.vector_dtype(1.0))

    # ----- public interface methods -----

    def simulate(self, mu, lam, rho, u_obs) -> Tuple[torch.Tensor, torch.Tensor]:

        wp.copy(self.mu.dof_values, _as_warp_array(mu, dtype=self.scalar_dtype))
        wp.copy(self.lam.dof_values, _as_warp_array(lam, dtype=self.scalar_dtype))
        wp.copy(self.rho.dof_values, _as_warp_array(rho, dtype=self.scalar_dtype))
        wp.copy(self.u_obs.dof_values, _as_warp_array(u_obs, dtype=self.vector_dtype))

        #print(self.mu.dof_values.device, self.lam.dof_values.device, self.rho.dof_values.device, self.u_obs.dof_values.device)

        K = self.assemble_stiffness(self.mu, self.lam)
        f = self.assemble_forcing(self.rho)
        M = self.get_preconditioner(K)

        #print(K.device, f.device, M.device)

        P = self.assemble_projector()
        u0 = self.apply_lifting(P, self.u_obs)
        f_tilde = self.shift_rhs(f, K, u0)

        #print(P.device, u0.device, f_tilde.device)

        self.solve_forward(K, self.u_sim, f_tilde, P, M, u0)
        self.compute_residual(self.mu, self.lam, self.rho, self.u_sim, self.res)
        return (
            wp.to_torch(self.u_sim.dof_values),
            wp.to_torch(self.res.dof_values)
        )

    def adjoint_setup(self, rho: torch.Tensor, u_obs: torch.Tensor):

        self.loss = wp.empty(1, dtype=self.scalar_dtype, requires_grad=True)
        self.cache = {}

        wp.copy(self.rho.dof_values, _as_warp_array(rho, dtype=self.scalar_dtype))
        wp.copy(self.u_obs.dof_values, _as_warp_array(u_obs, dtype=self.vector_dtype))

        #print(self.rho.dof_values.device, self.u_obs.dof_values.device)

        P = self.assemble_projector()
        f = self.assemble_forcing(self.rho)
        u0 = self.apply_lifting(P, self.u_obs)

        #print(P.device, f.device, u0.device)

        self.cache['P'] = P
        self.cache['f'] = f
        self.cache['u0'] = u0

    def adjoint_forward(self, mu, lam):
        wp.copy(self.mu.dof_values, _as_warp_array(mu, dtype=self.scalar_dtype))
        wp.copy(self.lam.dof_values, _as_warp_array(lam, dtype=self.scalar_dtype))

        K = self.assemble_stiffness(self.mu, self.lam)
        f = self.cache['f']
        M = self.get_preconditioner(K)

        #print(K.device, f.device, M.device)

        P = self.cache['P']
        u0 = self.cache['u0']
        f_tilde = self.shift_rhs(f, K, u0)

        #print(P.device, u0.device, f_tilde.device)

        self.solve_forward(K, self.u_sim, f_tilde, P, M, u0)

        self.tape = wp.Tape()
        with self.tape:
            self.compute_residual(self.mu, self.lam, self.rho, self.u_sim, self.res)

        def _solve_adjoint():
            self.solve_adjoint(K, self.res, self.u_sim, P, M)

        self.tape.record_func(
            backward=_solve_adjoint,
            arrays=[self.res.dof_values, self.u_sim.dof_values]
        )
        with self.tape:
            self.compute_loss(self.mu, self.lam, self.u_obs, self.u_sim, self.loss)

        return (
            wp.to_torch(self.u_sim.dof_values),
            wp.to_torch(self.res.dof_values),
            wp.to_torch(self.loss)
        )

    def adjoint_backward(self, loss_grad):
        wp.copy(self.loss.grad, _as_warp_array(loss_grad, dtype=self.scalar_dtype))
        self.tape.backward()
        return (
            wp.to_torch(self.mu.dof_values.grad),
            wp.to_torch(self.lam.dof_values.grad)
        )

    def zero_grad(self):
        _zero_grad(self.mu.dof_values)
        _zero_grad(self.lam.dof_values)
        _zero_grad(self.rho.dof_values)
        _zero_grad(self.u_obs.dof_values)
        _zero_grad(self.u_sim.dof_values)
        _zero_grad(self.res.dof_values)
        if getattr(self, 'loss', None) is not None:
            _zero_grad(self.loss)
        if getattr(self, 'tape', None) is not None:
            self.tape.reset()

    # ----- internal assembly methods -----

    def assemble_stiffness(self, mu: wp.fem.Field, lam: wp.fem.Field):
        K = wp.fem.integrate(
            pde_bilinear_form,
            fields={
                'u': self.u_trial,
                'v': self.v_test,
                'mu': mu,
                'lam': lam
            },
            values={'I': self.I},
            domain=self.interior,
            output_dtype=self.scalar_dtype
        )
        return K

    def assemble_forcing(self, rho: wp.fem.Field):
        f = wp.fem.integrate(
            pde_linear_form,
            fields={'v': self.v_test, 'rho': rho},
            values={'g': self.g},
            domain=self.interior,
            output_dtype=self.vector_dtype
        )
        return f

    def get_preconditioner(self, K):
        M = wp.optim.linear.preconditioner(K, ptype='diag')
        return M

    def assemble_projector(self):
        P = wp.fem.integrate(
            inner_form,
            fields={'u': self.ub_trial, 'v': self.vb_test},
            domain=self.boundary,
            assembly='nodal',
            output_dtype=self.scalar_dtype
        )
        wp.fem.normalize_dirichlet_projector(P)
        return P

    def apply_lifting(self, P, u_obs):
        u0 = P @ u_obs.dof_values
        return u0

    def shift_rhs(self, f, K, u0):
        f_tilde = f - (K @ u0)
        return f_tilde

    # --- solving linear systems ---

    def solve_forward(self, K, u_sim, f_tilde, P, M, u0):
        wp.fem.project_linear_system(K, f_tilde, P, normalize_projector=False)
        it, cg_res, tol = wp.optim.linear.cg(
            A=K, x=u_sim.dof_values, b=f_tilde, M=M, tol=self.cg_tol
        )
        if not np.isfinite(cg_res):
            print(it, cg_res, tol)
            raise RuntimeError(f'Non-finite CG residual')
        u_sim.dof_values += u0

    def solve_adjoint(self, K, res, u_sim, P, M):
        wp.fem.project_linear_system(K, u_sim.dof_values.grad, P, normalize_projector=False)
        it, cg_res, tol = wp.optim.linear.cg(
            A=K, x=res.dof_values.grad, b=u_sim.dof_values.grad, M=M, tol=self.cg_tol
        )
        #u_sim.dof_values.grad.zero_()
        if not np.isfinite(cg_res):
            print(it, cg_res, tol)
            raise RuntimeError(f'Non-finite CG residual')

    # --- residual and loss evaluation ---

    def compute_residual(self, mu, lam, rho, u_sim, res):
        wp.fem.integrate(
            pde_residual_form,
            fields={
                'u': u_sim,
                'v': self.v_test,
                'mu': mu,
                'lam': lam,
                'rho': rho
            },
            values={'g': self.g, 'I': self.I},
            domain=self.interior,
            output=res.dof_values
        )

    def compute_loss(self, mu, lam, u_obs, u_sim, loss):
        num = wp.empty(1, dtype=self.scalar_dtype, requires_grad=True)
        den = wp.empty(1, dtype=self.scalar_dtype, requires_grad=True)

        wp.fem.integrate(
            error_form,
            fields={'u': u_sim, 'v': u_obs},
            domain=self.interior,
            output=num
        )
        if self.relative_loss:
            wp.fem.integrate(
                norm2_form,
                fields={'u': u_obs},
                domain=self.interior,
                output=den
            )
        else:
            wp.fem.integrate(volume_form, domain=self.interior, output=den)

        err = num / (den + self.eps_div)
        reg = wp.empty(1, dtype=self.scalar_dtype, requires_grad=True)

        wp.fem.integrate(
            tv_reg_form,
            fields={'mu': self.mu, 'lam': self.lam},
            values={'eps_reg': self.eps_reg, 'eps_div': self.eps_div},
            domain=self.interior,
            output=reg
        )
        wp.copy(self.loss, err + reg * self.tv_reg_weight)


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

