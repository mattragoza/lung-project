from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np
import torch

import warp as wp
import warp.fem
import warp.optim.linear

wp.init()
wp.config.quiet = True

from . import base
from ..core import utils


def _as_warp_array(t: torch.Tensor, **kwargs) -> wp.array:
    return wp.from_torch(t.contiguous().detach(), **kwargs)

def _copy_warp_array(src, dst):
    try:
        wp.copy(src=src, dest=dst)
    except RuntimeError as exc:
        print(src.shape, dst.shape)
        raise exc

def _torch_grad(a: wp.array) -> torch.Tensor | None:
    return wp.to_torch(a.grad) if a.requires_grad else None


class WarpFEMSolver(base.PDESolver):

    def __init__(
        self,
        use_relative: bool=False,
        tv_reg_weight: float=0.0, # 1e-4 to 5e-2
        eps_reg: float=1e-3,
        eps_div: float=1e-12,
        cg_tol:  float=1e-5,
        scalar_degree: int=1,
        vector_degree: int=1,
        scalar_dtype=wp.float32,
        vector_dtype=wp.vec3f,
        device=None
    ):
        self.use_relative  = use_relative
        self.tv_reg_weight = tv_reg_weight

        self.eps_reg = eps_reg
        self.eps_div = eps_div
        self.cg_tol  = cg_tol

        self.scalar_degree = scalar_degree
        self.vector_degree = vector_degree
        self.scalar_dtype  = scalar_dtype
        self.vector_dtype  = vector_dtype

        self.device = device or wp.get_device(device)

        self._initialized = False

    def bind_geometry(self, verts: torch.Tensor, cells: torch.Tensor):

        # IMPORTANT NOTE:
        # - ScopedDevice doesn't affect wp.from_torch, only internal arrays
        # - if the geometry is defined from cpu but other arrays are cuda,
        #   rasterization fails silently- it produces all background values
        # - therefore we explicitly move the geometry to the solver device

        verts = verts.to(self.device)
        cells = cells.to(self.device)

        with wp.ScopedDevice(self.device):
            verts = _as_warp_array(verts, dtype=self.vector_dtype)
            cells = _as_warp_array(cells, dtype=wp.int32)

            self.geometry = wp.fem.Tetmesh(cells, verts, build_bvh=True)

            self.interior = wp.fem.Cells(self.geometry)
            self.boundary = wp.fem.BoundarySides(self.geometry)

            self.S = wp.fem.make_polynomial_space(self.geometry, degree=self.scalar_degree, dtype=self.scalar_dtype)
            self.V = wp.fem.make_polynomial_space(self.geometry, degree=self.vector_degree, dtype=self.vector_dtype)

            self.u_trial  = wp.fem.make_trial(self.V, domain=self.interior)
            self.v_test   = wp.fem.make_test(self.V,  domain=self.interior)
            self.ub_trial = wp.fem.make_trial(self.V, domain=self.boundary)
            self.vb_test  = wp.fem.make_test(self.V,  domain=self.boundary)

            self.g = self.vector_dtype([0, 0, -9.81]) # m/s^2
            self.I = wp.diag(self.vector_dtype(1.0))
            self._initialized = True

    def make_scalar_field(self, values=None, requires_grad=None):
        assert self._initialized, 'Geometry not initialized'
        s = self.S.make_field() # controlled by ScopedDevice
        if values is not None:
            array_vals = _as_warp_array(values, dtype=self.scalar_dtype)
            _copy_warp_array(src=array_vals, dst=s.dof_values)
        if requires_grad is not None:
            s.dof_values.requires_grad = requires_grad
        elif values is not None:
            s.dof_values.requires_grad = values.requires_grad
        return s

    def make_vector_field(self, values=None, requires_grad=None):
        assert self._initialized, 'Geometry not initialized'
        v = self.V.make_field() # controlled by ScopedDevice
        if values is not None:
            array_vals = _as_warp_array(values, dtype=self.vector_dtype)
            _copy_warp_array(src=array_vals, dst=v.dof_values)
        if requires_grad is not None:
            v.dof_values.requires_grad = requires_grad
        elif values is not None:
            v.dof_values.requires_grad = values.requires_grad
        return v

    def rasterize_scalar_field(self, values, shape, bounds):
        field = self.make_scalar_field(values)
        return rasterize_field(field, shape, bounds)

    def rasterize_vector_field(self, values, shape, bounds):
        field = self.make_vector_field(values)
        return rasterize_field(field, shape, bounds)

    def solve(self, mu, lam, rho, u_bc):

        with wp.ScopedDevice(self.device):
            mu = self.make_scalar_field(mu)
            lam = self.make_scalar_field(lam)
            rho = self.make_scalar_field(rho)
            u_bc = self.make_vector_field(u_bc)

            K = self.assemble_stiffness(mu, lam)
            f = self.assemble_forcing(rho)
            M = self.get_preconditioner(K)

            P = self.assemble_projector()
            u0 = self.apply_lifting(P, u_bc)
            f_tilde = self.shift_rhs(f, K, u0)
            u_sim = self.make_vector_field()
            self.solve_forward(K, u_sim, f_tilde, P, M, u0)

        return wp.to_torch(u_sim.dof_values)

    def forward(self, mu, lam, rho, u_bc, u_obs, mask):

        with wp.ScopedDevice(self.device):
            mu = self.make_scalar_field(mu)
            lam = self.make_scalar_field(lam)
            rho = self.make_scalar_field(rho)
            u_bc = self.make_vector_field(u_bc)
            u_obs = self.make_vector_field(u_obs)
            mask = self.make_scalar_field(mask)

            K = self.assemble_stiffness(mu, lam)
            f = self.assemble_forcing(rho)
            M = self.get_preconditioner(K)

            P = self.assemble_projector()
            u0 = self.apply_lifting(P, u_bc)
            f_tilde = self.shift_rhs(f, K, u0)

            u_sim = self.make_vector_field(requires_grad=True)
            self.solve_forward(K, u_sim, f_tilde, P, M, u0)

            tape = wp.Tape()
            with tape:
                res = self.make_vector_field(requires_grad=True)
                self.compute_residual(mu, lam, rho, u_sim, res)

            tape.record_func(
                backward=lambda: self.solve_adjoint(K, res, u_sim, P, M),
                arrays=[res.dof_values, u_sim.dof_values]
            )
            with tape:
                loss = self.compute_loss(mu, lam, u_sim, u_obs, mask)

        outputs = {
            'u_sim': wp.to_torch(u_sim.dof_values),
            'res':   wp.to_torch(res.dof_values),
            'loss':  wp.to_torch(loss)
        }
        context = { # track variables for backward pass
            'mu':    mu,
            'lam':   lam,
            'rho':   rho,
            'u_bc':  u_bc,
            'u_obs': u_obs,
            'u_sim': u_sim,
            'res':   res,
            'loss':  loss,
            'tape':  tape
        }
        return outputs, context

    def backward(self, loss_grad, context):
        input_grads = {}
        with wp.ScopedDevice(self.device):
            context['loss'].grad = _as_warp_array(loss_grad)
            context['tape'].backward()
            input_grads['mu'] = _torch_grad(context['mu'].dof_values)
            input_grads['lam'] = _torch_grad(context['lam'].dof_values)
            input_grads['rho'] = _torch_grad(context['rho'].dof_values)
            input_grads['u_bc'] = _torch_grad(context['u_bc'].dof_values)
            input_grads['u_obs'] = _torch_grad(context['u_obs'].dof_values)
            for key in list(context.keys()): # try to explicitly free warp arrays
                context[key] = None
            context.clear()
        return input_grads

    def zero_grad(self):
        if hasattr(self, 'tape'):
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

    def apply_lifting(self, P, u_bc):
        u0 = P @ u_bc.dof_values
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

    def compute_loss(self, mu, lam, u_sim, u_obs, mask):
        num = wp.empty(1, dtype=self.scalar_dtype, requires_grad=True)
        den = wp.empty(1, dtype=self.scalar_dtype, requires_grad=True)

        wp.fem.integrate(
            error_form,
            fields={'u': u_sim, 'v': u_obs, 'w': mask},
            domain=self.interior,
            output=num
        )
        if self.use_relative:
            wp.fem.integrate(
                norm2_form,
                fields={'u': u_obs, 'w': mask},
                domain=self.interior,
                output=den
            )
        else:
            wp.fem.integrate(
                volume_form,
                fields={'w': mask},
                domain=self.interior,
                output=den
            )

        err = num / (den + self.eps_div)
        reg = wp.empty(1, dtype=self.scalar_dtype, requires_grad=True)

        wp.fem.integrate(
            tv_reg_form,
            fields={'mu': mu, 'lam': lam},
            values={'eps_reg': self.eps_reg, 'eps_div': self.eps_div},
            domain=self.interior,
            output=reg
        )
        loss = err + reg * self.tv_reg_weight
        loss.requires_grad = True
        return loss


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
def error_form(s: wp.fem.Sample, u: wp.fem.Field, v: wp.fem.Field, w: wp.fem.Field):
    r_s = u(s) - v(s)
    return w(s) * wp.dot(r_s, r_s)


@wp.fem.integrand
def rel_error_form(
    s: wp.fem.Sample,
    u: wp.fem.Field,
    v: wp.fem.Field,
    w: wp.fem.Field,
    eps_div: float
):
    u_s, v_s = u(s), v(s)
    r_s = (u_s - v_s) / (v_s + eps_div)
    return w(s) * wp.dot(r_s, r_s)


@wp.fem.integrand
def norm2_form(s: wp.fem.Sample, u: wp.fem.Field, w: wp.fem.Field):
    u_s = u(s)
    return w(s) * wp.dot(u_s, u_s)


@wp.fem.integrand
def volume_form(s: wp.fem.Sample, w: wp.fem.Field):
    return w(s)


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
        shape: (I, J, K) voxel grid shape (spatial dims)
        bounds: Lower and upper grid bounds (in world meters)
    Returns:
        (C, I, J, K) rasterized field tensor
    '''
    I, J, K = shape
    C = wp.types.type_length(src.dtype)

    grid = wp.fem.Grid3D(
        res=wp.vec3i(shape),
        bounds_lo=wp.vec3f(bounds[0]),
        bounds_hi=wp.vec3f(bounds[1])
    )
    dst_domain = wp.fem.Cells(grid)

    dst_space = wp.fem.make_polynomial_space(grid, degree=0, dtype=src.dtype)
    dst = dst_space.make_field()

    src_nc = wp.fem.NonconformingField(dst_domain, src, background)
    wp.fem.interpolate(src_nc, dest=dst)

    return wp.to_torch(dst.dof_values).reshape(I, J, K, C).permute(3,0,1,2)

