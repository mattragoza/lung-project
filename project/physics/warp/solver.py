# physics/warp/solver.py

import numpy as np
import torch

import warp as wp
import warp.fem
import warp.optim.linear

from ..solvers import PDESolver
from . import materials, forms


def _as_warp_array(a, **kwargs) -> wp.array:
    if isinstance(a, torch.Tensor):
        return wp.from_torch(a.contiguous().detach(), **kwargs)
    return wp.array(a, **kwargs)


def _copy_warp_array(src: wp.array, dst: wp.array):
    try:
        wp.copy(dst, src=src)
    except RuntimeError:
        print(src.shape, dst.shape)
        raise


def _warp_array_norm(a: wp.array) -> float:
    return torch.linalg.norm(wp.to_torch(a)).item()


def _get_torch_grad(a: wp.array) -> torch.Tensor | None:
    return wp.to_torch(a.grad) if a.requires_grad else None


def _init_warp_field(space, values=None, requires_grad=None):
    f = space.make_field()

    if values is not None:
        array = _as_warp_array(values, dtype=f.dof_values.dtype)
        _copy_warp_array(array, f.dof_values)

    if requires_grad is not None:
        f.dof_values.requires_grad = bool(requires_grad)

    elif values is not None:
        f.dof_values.requires_grad = values.requires_grad

    return f


class WarpFEMSolver(PDESolver):

    def __init__(
        self,
        material_type: str = 'linear',
        g_vector: tuple = (0., 0., -9.81), # m/s^2
        relative_loss: bool = True,
        tv_reg_weight: float = 1e-4,
        eps_reg: float = 1e-3, # avoids non-diffability
        eps_div: float = 1e-6, # avoids division by zero
        newton_steps: int = 100,
        newton_tries: int = 32,
        newton_alpha: float = 1.0, # initial step size
        newton_beta: float = 0.5,  # backtracking factor
        newton_rtol: float = 1e-5,
        cg_maxiter: int = 1000,
        cg_rtol: float = 1e-5,
        scalar_degree: int = 1,
        vector_degree: int = 1,
        scalar_dtype = wp.float32,
        vector_dtype = wp.vec3f,
        device: str = 'cuda',
        verbose: bool = False
    ):
        self.material = materials.WarpMaterial.get_subclass(material_type)
        self.g_vector = g_vector

        # objective function
        self.relative_loss = bool(relative_loss)
        self.tv_reg_weight = float(tv_reg_weight)

        self.eps_reg = float(eps_reg)
        self.eps_div = float(eps_div)

        # Newton's method settings
        self.newton_steps = int(newton_steps)
        self.newton_tries = int(newton_tries)
        self.newton_alpha = float(newton_alpha)
        self.newton_beta = float(newton_beta)
        self.newton_rtol = float(newton_rtol)

        # conjugate gradient settings
        self.cg_maxiter = int(cg_maxiter)
        self.cg_rtol = float(cg_rtol)

        # field representation settings
        self.scalar_degree = int(scalar_degree)
        self.vector_degree = int(vector_degree)

        self.scalar_dtype = scalar_dtype
        self.vector_dtype = vector_dtype

        self.device = device or wp.get_device(device)
        self.verbose = verbose

        self._geometry_initialized = False

    # ----- geometry initialization -----

    def bind_geometry(self, verts: torch.Tensor, cells: torch.Tensor):
        '''
        Args:
            verts: (N, 3) float tensor of vertex coords in meters
            cells: (M, 4) int tensor of tetra cell vertex indices
        '''
        if verts.ndim != 2 or verts.shape[-1] != 3:
            raise ValueError(f'Invalid verts shape: {verts.shape!r}')

        if cells.ndim != 2 or cells.shape[-1] != 4:
            raise ValueError(f'Invalid cells shape: {cells.shape!r}')

        # ScopedDevice doesn't affect wp.from_torch, only new arrays,
        # so we need to explicitly move geometry to the solver device
        device = str(self.device)

        verts = _as_warp_array(verts.to(device), dtype=self.vector_dtype)
        cells = _as_warp_array(cells.to(device), dtype=wp.int32)

        with wp.ScopedDevice(self.device):
            self.geometry = wp.fem.Tetmesh(cells, verts, build_bvh=True)

            self.interior = wp.fem.Cells(self.geometry)
            self.boundary = wp.fem.BoundarySides(self.geometry)

            self.S = wp.fem.make_polynomial_space(
                self.geometry, self.scalar_dtype, degree=self.scalar_degree
            )
            self.V = wp.fem.make_polynomial_space(
                self.geometry, self.vector_dtype, degree=self.vector_degree
            )

            self.u_interior = wp.fem.make_trial(self.V, domain=self.interior)
            self.u_boundary = wp.fem.make_trial(self.V, domain=self.boundary)

            self.v_interior = wp.fem.make_test(self.V, domain=self.interior)
            self.v_boundary = wp.fem.make_test(self.V, domain=self.boundary)

            self.g = self.vector_dtype(self.g_vector)

        self._geometry_initialized = True

    # ----- physical field initialization -----

    def init_scalar_field(self, values=None, requires_grad=None):
        if not self._geometry_initialized:
            raise RuntimeError('Geometry not initialized')
        return _init_warp_field(self.S, values, requires_grad)

    def init_vector_field(self, values=None, requires_grad=None):
        if not self._geometry_initialized:
            raise RuntimeError('Geometry not initialized')
        return _init_warp_field(self.V, values, requires_grad)

    def init_input_fields(self, mu, lam, rho, u_bc):
        return (
            self.init_scalar_field(mu),
            self.init_scalar_field(lam),
            self.init_scalar_field(rho),
            self.init_vector_field(u_bc)
        )

    def init_target_fields(self, u_obs, mask):
        return (
            self.init_vector_field(u_obs),
            self.init_scalar_field(mask)
        )

    def init_unknown_field(self, u_bc, P, requires_grad=None):
        return self.init_vector_field(P @ u_bc.dof_values, requires_grad)

    # ----- public solver interface -----

    def solve_forward(self, mu, lam, rho, u_bc):

        with wp.ScopedDevice(self.device):
            mu, lam, rho, u_bc = self.init_input_fields(mu, lam, rho, u_bc)
            P = self.assemble_projector(normalize=True)
            u_sim = self.init_unknown_field(u_bc, P, requires_grad=False)
            J, M = self.solve_forward_system(mu, lam, rho, u_sim, P)

        return wp.to_torch(u_sim.dof_values)

    def loss_forward(self, mu, lam, rho, u_bc, u_obs, mask):

        with wp.ScopedDevice(self.device):
            mu, lam, rho, u_bc = self.init_input_fields(mu, lam, rho, u_bc)
            P = self.assemble_projector(normalize=True)
            u_sim = self.init_unknown_field(u_bc, P, requires_grad=True)
            J, M = self.solve_forward_system(mu, lam, rho, u_sim, P)

            tape = wp.Tape()
            with tape:
                res = self.assemble_residual(mu, lam, rho, u_sim, True)

            def backward():
                return self.solve_adjoint_system(J, res, u_sim, P, M)

            tape.record_func(backward, [res.dof_values, u_sim.dof_values])

            u_obs, mask = self.init_target_fields(u_obs, mask)
            with tape:
                loss = self.evaluate_loss(mu, lam, rho, u_sim, u_obs, mask)

        # torch tensors returned to caller
        outputs = {
            'loss': wp.to_torch(loss),
            'u_sim': wp.to_torch(u_sim.dof_values),
            'residual': wp.to_torch(res.dof_values)
        }

        # warp objects needed for backward pass
        context = { 
            'mu': mu, 'lam': lam, 'rho': rho, 'u_bc': u_bc, 'u_obs': u_obs,
            'loss': loss,
            'tape': tape
        }

        return outputs, context

    def loss_backward(self, loss_grad, context):
        input_grads = {}

        with wp.ScopedDevice(self.device):
            context['loss'].grad = _as_warp_array(loss_grad)
            context['tape'].backward()

        for key in ['mu', 'lam', 'rho', 'u_bc', 'u_obs']:
            input_grads[key] = _get_torch_grad(context[key].dof_values)

        for key in context: # try to explicitly free memory
            context[key] = None

        return input_grads

    # ----- solving systems of equations -----

    def solve_forward_system(self, mu, lam, rho, u, P):
        if self.material.is_linear:
            return self.solve_linear_system(mu, lam, rho, u, P)
        return self.solve_newton_method(mu, lam, rho, u, P)

    def solve_linear_system(self, mu, lam, rho, u, P):

        r = self.assemble_residual(mu, lam, rho, u)
        J = self.assemble_jacobian(mu, lam, u)

        self.project_linear_system(J, r.dof_values, P)
        M = wp.optim.linear.preconditioner(J, ptype='diag')

        du = self.init_vector_field()

        cg_iter, cg_ares, cg_atol = wp.optim.linear.cg(
            A=J,
            x=du.dof_values,
            b=r.dof_values,
            M=M,
            tol=self.cg_rtol,
            maxiter=self.cg_maxiter
        )

        if not np.isfinite(cg_ares):
            raise RuntimeError('Non-finite CG residual in linear solve')

        u.dof_values += du.dof_values

        return J, M

    def solve_adjoint_system(self, J, r, u, P, M):
        self.project_linear_system(J, u.dof_values.grad, P)

        cg_iter, cg_ares, cg_atol = wp.optim.linear.cg(
            A=J,
            x=r.dof_values.grad,
            b=u.dof_values.grad,
            M=M,
            tol=self.cg_rtol,
            maxiter=self.cg_maxiter
        )

        if not np.isfinite(cg_ares):
            raise RuntimeError('Non-finite CG residual in adjoint solve')

    def solve_newton_method(self, mu, lam, rho, u, P):
        base_ares = None # initial residual norm

        for step in range(self.newton_steps):
            r = self.assemble_residual(mu, lam, rho, u)
            J = self.assemble_jacobian(mu, lam, u)

            self.project_linear_system(J, r.dof_values, P)
            M = wp.optim.linear.preconditioner(J, ptype='diag')

            ares = _warp_array_norm(r.dof_values)
            if base_ares is None:
                base_ares = ares + self.eps_div

            rres = ares / base_ares
            if rres < self.newton_rtol:
                if self.verbose:
                    print(f'Newton solver converged at step {step + 1}.')
                break

            du = self.init_vector_field()

            cg_iter, cg_ares, cg_atol = wp.optim.linear.gmres(
                A=J,
                x=du.dof_values,
                b=r.dof_values,
                M=M,
                tol=self.cg_rtol,
                maxiter=self.cg_maxiter
            )

            if not np.isfinite(cg_ares):
                raise RuntimeError(f'Non-finite residual at Newton step {step + 1}')

            alpha = self.adaptive_step_size(mu, lam, rho, u, du, P, ares)

            if alpha <= 0:
                raise RuntimeError(f'Line search failed at Newton step {step + 1}')

            if self.verbose:
                cg_rres = cg_ares / cg_atol * self.cg_rtol

                print(
                    f'Newton step {step + 1}: '
                    f'cg_iter = {cg_iter:d} '
                    f'cg_rres = {cg_rres:.2e} '
                    f'nm_ares = {ares:.2e} '
                    f'nm_rres = {rres:.2e} '
                    f'alpha = {alpha:.2e}'
                )

            u.dof_values += alpha * du.dof_values

        return J, M

    def adaptive_step_size(self, mu, lam, rho, u, du, P, init_norm):
        alpha = self.newton_alpha # initial step size

        u_curr = self.init_vector_field()
        for t in range(self.newton_tries):

            _copy_warp_array(u.dof_values, u_curr.dof_values)
            u_curr.dof_values += alpha * du.dof_values

            r = self.assemble_residual(mu, lam, rho, u_curr)
            r.dof_values -= P @ r.dof_values

            curr_norm = _warp_array_norm(r.dof_values)
            if np.isfinite(curr_norm) and curr_norm < init_norm:
                return alpha

            alpha = alpha * self.newton_beta # backtracking factor

        return 0.0

    # ----- internal assembly methods -----

    def assemble_residual(self, mu, lam, rho, u, requires_grad=None):
        res = self.init_vector_field(requires_grad=requires_grad)

        wp.fem.integrate(
            self.material.residual_form,
            fields={
                'u': u,
                'v': self.v_interior,
                'mu': mu,
                'lam': lam,
                'rho': rho
            },
            values={'g': self.g},
            domain=self.interior,
            output=res.dof_values
        )

        return res

    def assemble_jacobian(self, mu, lam, u):

        J = wp.fem.integrate(
            self.material.jacobian_form,
            fields={
                'u': u,
                'v': self.v_interior,
                'du': self.u_interior,
                'mu': mu,
                'lam': lam
            },
            domain=self.interior,
            output_dtype=self.scalar_dtype,
            bsr_options={'construction': 'row_compress'}
        )

        return J

    def assemble_projector(self, normalize=False):

        P = wp.fem.integrate(
            forms.inner_product_form,
            fields={
                'u': self.u_boundary,
                'v': self.v_boundary
            },
            domain=self.boundary,
            assembly='nodal',
            output_dtype=self.scalar_dtype
        )

        if normalize:
            wp.fem.normalize_dirichlet_projector(P)

        return P

    def project_linear_system(self, A, b, P, normalize=False):
        wp.fem.project_linear_system(
            system_matrix=A,
            system_rhs=b,
            projector_matrix=P,
            normalize_projector=normalize
        )

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

        wp.fem.integrate(
            forms.TV_reg_form,
            fields={'mu': mu},
            values={'eps_div': self.eps_div, 'eps_reg': self.eps_reg},
            domain=self.interior,
            output=reg
        )

        loss = num / (den + self.eps_div) + self.tv_reg_weight * reg
        loss.requires_grad = True

        return loss

    # ----- rasterization helpers -----

    def rasterize_scalar_field(self, values, shape, bounds):
        field = self.init_scalar_field(values)
        return rasterize_field(field, shape, bounds)

    def rasterize_vector_field(self, values, shape, bounds):
        field = self.init_vector_field(values)
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
    C = getattr(src.dtype, '_length_', 1)

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

