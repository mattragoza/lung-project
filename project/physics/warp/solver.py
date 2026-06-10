from typing import Dict, Tuple, Optional

import numpy as np
import torch

import warp as wp
import warp.fem
import warp.optim.linear

from . import forms
from .. import solvers


def _as_warp_array(t: torch.Tensor, **kwargs) -> wp.array:
    return wp.from_torch(t.contiguous().detach(), **kwargs)


def _copy_warp_array(src: wp.array, dst: wp.array) -> None:
    try:
        wp.copy(src=src, dest=dst)
    except RuntimeError as exc:
        print(src.shape, dst.shape)
        raise exc


def _get_torch_grad(a: wp.array) -> torch.Tensor | None:
    return wp.to_torch(a.grad) if a.requires_grad else None


def _make_warp_field(space, values=None, requires_grad=None):
    f = space.make_field()
    if values is not None:
        a = _as_warp_array(values, dtype=f.dof_values.dtype)
        _copy_warp_array(src=a, dst=f.dof_values)
    if requires_grad is not None:
        f.dof_values.requires_grad = bool(requires_grad)
    elif values is not None:
        f.dof_values.requires_grad = values.requires_grad
    return f


class WarpFEMSolver(solvers.PDESolver):

    def __init__(
        self,
        material_type: str,
        relative_loss: bool = True,
        tv_reg_weight: float = 1e-4,
        newton_steps: int = 100,
        newton_alpha: float = 1.0,
        newton_rtol: float = 1e-5,
        cg_maxiter: int = 0,
        cg_rtol: float = 1e-5,
        eps_reg: float = 1e-3,
        eps_div: float = 1e-6,
        scalar_degree: int = 1,
        vector_degree: int = 1,
        scalar_dtype = wp.float32,
        vector_dtype = wp.vec3f,
        device: str = 'cuda'
    ):
        # physical material model
        self.material = forms.WarpMaterial.get_subclass(material_type)

        # objective function
        self.relative_loss = bool(relative_loss)
        self.tv_reg_weight = float(tv_reg_weight)

        # Newton's method settings
        self.newton_steps = int(newton_steps)
        self.newton_alpha = float(newton_alpha)
        self.newton_rtol = float(newton_rtol)

        # conjugate gradient settings
        self.cg_maxiter = int(cg_maxiter)
        self.cg_rtol = float(cg_rtol)

        # numerical stability
        self.eps_reg = float(eps_reg)
        self.eps_div = float(eps_div)

        # mesh field representations
        self.scalar_degree = scalar_degree
        self.vector_degree = vector_degree
        self.scalar_dtype  = scalar_dtype
        self.vector_dtype  = vector_dtype

        self.device = device or wp.get_device(device)

        self._initialized = False

    # ----- geometry initialization -----

    def bind_geometry(self, verts: torch.Tensor, cells: torch.Tensor):
        '''
        Args:
            verts: (N, 3) float tensor of vertex coords, in meters
            cells: (M, 4) int tensor of tetra cell vertex indices
        '''
        if not (verts.ndim == 2 and verts.shape[-1] == 3):
            raise ValueError(f'Invalid verts shape: {verts.shape!r}')
        if not (cells.ndim == 2 and cells.shape[-1] == 4):
            raise ValueError(f'Invalid cells shape: {cells.shape!r}')

        # ScopedDevice doesn't affect wp.from_torch, only new arrays
        # so we need to explicitly move geometry to solver device
        verts = verts.to(str(self.device))
        cells = cells.to(str(self.device))

        with wp.ScopedDevice(self.device):
            self.verts = _as_warp_array(verts, dtype=self.vector_dtype)
            self.cells = _as_warp_array(cells, dtype=wp.int32)

            self.init_geometric_domain()
            self.init_function_spaces()
            self.init_trial_and_test()
            self.init_constants()

        self._initialized = True

    def init_geometric_domain(self):
        self.geometry = wp.fem.Tetmesh(self.cells, self.verts, build_bvh=True)
        self.interior = wp.fem.Cells(self.geometry)
        self.boundary = wp.fem.BoundarySides(self.geometry)

    def init_function_spaces(self):
        self.S = wp.fem.make_polynomial_space(
            geo=self.geometry,
            dtype=self.scalar_dtype,
            degree=self.scalar_degree
        )
        self.V = wp.fem.make_polynomial_space(
            geo=self.geometry,
            dtype=self.vector_dtype,
            degree=self.vector_degree
        )

    def init_trial_and_test(self):
        self.u_trial  = wp.fem.make_trial(self.V, domain=self.interior)
        self.v_test   = wp.fem.make_test(self.V, domain=self.interior)
        self.ub_trial = wp.fem.make_trial(self.V, domain=self.boundary)
        self.vb_test  = wp.fem.make_test(self.V, domain=self.boundary)

    def init_constants(self):
        self.g = self.vector_dtype([0., 0., -9.81]) # m/s^2
        self.I = wp.diag(self.vector_dtype(1.))

    def require_initialized_geometry(self):
        if not self._initialized:
            raise RuntimeError('Geometry not initialized')

    # ----- physical field initialization -----

    def make_scalar_field(self, values=None, requires_grad=None):
        self.require_initialized_geometry()
        return _make_warp_field(self.S, values, requires_grad)

    def make_vector_field(self, values=None, requires_grad=None):
        self.require_initialized_geometry()
        return _make_warp_field(self.V, values, requires_grad)

    def make_input_fields(self, mu, lam, rho, u_bc):
        mu = self.make_scalar_field(mu)
        lam = self.make_scalar_field(lam)
        rho = self.make_scalar_field(rho)
        u_bc = self.make_vector_field(u_bc)
        return mu, lam, rho, u_bc

    def make_target_fields(self, u_obs, mask):
        u_obs = self.make_vector_field(u_obs)
        mask = self.make_scalar_field(mask)
        return u_obs, mask

    def init_unknown_field(self, u_bc, requires_grad=None):
        u = self.make_vector_field(requires_grad=requires_grad)
        P = self.assemble_boundary_projector()
        u.dof_values += P @ u_bc.dof_values
        return u, P

    # ----- public solver interface -----

    def solve(self, mu, lam, rho, u_bc):
        with wp.ScopedDevice(self.device):
            mu, lam, rho, u_bc = self.make_input_fields(mu, lam, rho, u_bc)

            init_material = self.material.get_linear()
            u, P = self.init_unknown_field(u_bc, requires_grad=False)
            J, M = self.solve_newtons_method(init_material, mu, lam, rho, u, P)

            if not self.material.is_linear:
                J, M = self.solve_newtons_method(self.material, mu, lam, rho, u, P)

        return wp.to_torch(u.dof_values)

    def forward(self, mu, lam, rho, u_bc, u_obs, mask):
        with wp.ScopedDevice(self.device):
            mu, lam, rho, u_bc = self.make_input_fields(mu, lam, rho, u_bc)

            init_material = self.material.get_linear()
            u, P = self.init_unknown_field(u_bc, requires_grad=True)
            J, M = self.solve_newtons_method(init_material, mu, lam, rho, u, P)

            if not self.material.is_linear:
                J, M = self.solve_newtons_method(self.material, mu, lam, rho, u, P)

            u_obs, mask = self.make_target_fields(u_obs, mask)

            tape = wp.Tape()
            with tape:
                res = self.assemble_residual(
                    self.material, mu, lam, rho, u, requires_grad=True
                )

            tape.record_func(
                backward=lambda: self.solve_adjoint_system(J, res, u, P, M),
                arrays=[res.dof_values, u.dof_values]
            )
            with tape:
                loss = self.evaluate_loss(mu, lam, rho, u, u_obs, mask)

        outputs = {
            'u_sim': wp.to_torch(u.dof_values),
            'res': wp.to_torch(res.dof_values),
            'loss': wp.to_torch(loss),
        }
        context = { # track variables for backward pass
            'mu': mu,
            'lam': lam,
            'rho': rho,
            'u_bc': u_bc,
            'u_obs': u_obs,
            'u_sim': u,
            'res': res,
            'loss': loss,
            'tape': tape,
        }
        return outputs, context

    def backward(self, loss_grad, context):
        input_grads = {}

        with wp.ScopedDevice(self.device):
            context['loss'].grad = _as_warp_array(loss_grad)
            context['tape'].backward()

            input_grads['mu'] = _get_torch_grad(context['mu'].dof_values)
            input_grads['lam'] = _get_torch_grad(context['lam'].dof_values)
            input_grads['rho'] = _get_torch_grad(context['rho'].dof_values)
            input_grads['u_bc'] = _get_torch_grad(context['u_bc'].dof_values)
            input_grads['u_obs'] = _get_torch_grad(context['u_obs'].dof_values)

            for key in context: # try to explicitly free warp context
                context[key] = None
            context.clear()

        return input_grads

    def zero_grad(self):
        if getattr(self, 'tape', None) is not None:
            self.tape.reset()

    # ----- solving systems of equations -----

    def solve_newtons_method(self, material, mu, lam, rho, u, P):
        J = M = None
        ares0 = None

        def _array_norm(a: wp.array) -> float:
            return torch.linalg.norm(wp.to_torch(a)).item()

        for step in range(self.newton_steps):
            r = self.assemble_residual(material, mu, lam, rho, u)
            J = self.assemble_jacobian(material, mu, lam, u)

            wp.fem.project_linear_system(J, r.dof_values, P, normalize_projector=False)

            ares = _array_norm(r.dof_values)
            if ares0 is None:
                ares0 = ares + self.eps_div

            rres = ares / ares0
            if rres < self.newton_rtol:
                print('    Newton solver converged.')
                break

            M = wp.optim.linear.preconditioner(J, ptype='diag')

            du = self.make_vector_field()
            cg_iter, cg_ares, cg_atol = wp.optim.linear.cg(
                A=J,
                x=du.dof_values,
                b=r.dof_values,
                M=M,
                tol=self.cg_rtol,
                maxiter=self.cg_maxiter
            )
            cg_rres = cg_ares / (cg_atol / self.cg_rtol)


            step_id = 'linear' if material.is_linear else f'step {step+1}'
            print(
                f'    {step_id}: '
                f'cg_iter = {cg_iter:d} '
                #f'cg_ares = {cg_ares:.4e} '
                #f'cg_atol = {cg_atol:.4f} '
                f'cg_rres = {cg_rres:.4e} '
                #f'cg_rtol = {self.cg_rtol:.4e}'
                f'ares = {ares:.4e}',
                f'rres = {rres:.4e}',
            )

            if not np.isfinite(cg_ares):
                raise RuntimeError('Non-finite CG residual in forward solve')


            u.dof_values += self.newton_alpha * du.dof_values

            if material.is_linear:
                break

        return J, M

    def solve_adjoint_system(self, J, r, u, P, M):
        wp.fem.project_linear_system(J, u.dof_values.grad, P, normalize_projector=False)

        cg_it, cg_ares, cg_atol = wp.optim.linear.cg(
            A=J,
            x=r.dof_values.grad,
            b=u.dof_values.grad,
            M=M,
            tol=self.cg_rtol,
            maxiter=self.cg_maxiter
        )

        if not np.isfinite(cg_ares):
            raise RuntimeError('Non-finite CG residual in adjoint solve')

    # ----- internal assembly -----

    def assemble_residual(self, material, mu, lam, rho, u, requires_grad=None):
        res = self.make_vector_field(requires_grad=requires_grad)
        wp.fem.integrate(
            material.residual_form,
            fields={
                'u': u,
                'v': self.v_test,
                'mu': mu,
                'lam': lam,
                'rho': rho
            },
            values={'g': self.g, 'I': self.I},
            domain=self.interior,
            output=res.dof_values
        )
        return res

    def assemble_jacobian(self, material, mu, lam, u):
        J = wp.fem.integrate(
            material.jacobian_form,
            fields={
                'u': u,
                'du': self.u_trial,
                'v': self.v_test,
                'mu': mu,
                'lam': lam
            },
            values={'I': self.I},
            domain=self.interior,
            output_dtype=self.scalar_dtype
        )
        return J

    def assemble_boundary_projector(self):
        P = wp.fem.integrate(
            forms.inner_product_form,
            fields={'u': self.ub_trial, 'v': self.vb_test},
            domain=self.boundary,
            assembly='nodal',
            output_dtype=self.scalar_dtype
        )
        wp.fem.normalize_dirichlet_projector(P)
        return P

    # ----- loss evaluation -----

    def evaluate_loss(self, mu, lam, rho, u_sim, u_obs, mask):

        # loss numerator, denominator, and regularization term
        num = wp.empty(1, dtype=self.scalar_dtype, requires_grad=True)
        den = wp.empty(1, dtype=self.scalar_dtype, requires_grad=True)
        reg = wp.empty(1, dtype=self.scalar_dtype, requires_grad=True)

        wp.fem.integrate(
            forms.squared_error_form,
            fields={'u': u_sim, 'v': u_obs, 'w': mask},
            domain=self.interior,
            output=num
        )

        if self.relative_loss:
            wp.fem.integrate(
                forms.squared_norm_form,
                fields={'u': u_obs, 'w': mask},
                domain=self.interior,
                output=den
            )
        else:
            wp.fem.integrate(
                forms.volume_form,
                fields={'w': mask},
                domain=self.interior,
                output=den
            )

        error = num / (den + self.eps_div)

        wp.fem.integrate(
            forms.tv_regularization_form,
            fields={'mu': mu, 'lam': lam, 'rho': rho},
            values={'eps_reg': self.eps_reg, 'eps_div': self.eps_div},
            domain=self.interior,
            output=reg
        )

        loss = error + self.tv_reg_weight * reg
        loss.requires_grad = True
        return loss

    # ----- rasterization helpers -----

    def rasterize_scalar_field(self, values, shape, bounds):
        field = self.make_scalar_field(values)
        return rasterize_field(field, shape, bounds)

    def rasterize_vector_field(self, values, shape, bounds):
        field = self.make_vector_field(values)
        return rasterize_field(field, shape, bounds)


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

