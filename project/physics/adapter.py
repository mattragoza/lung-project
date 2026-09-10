# physics/adapter.py

from typing import Dict, Tuple, Optional, Any

import meshio
import numpy as np
import torch

from ..common import utils, transforms
from . import context, solvers


ELASTIC_KEYS  = ('E', 'nu', 'G', 'K', 'mu', 'lam')
MATERIAL_KEYS = ELASTIC_KEYS + ('rho',)

VALID_ELASTIC_PARAMS = {
    ('E', 'nu'),
    ('E', 'K'),
    ('G', 'K'),
    ('mu', 'lam')
}


class PhysicsAdapter:
    '''
    PhysicsAdapter manages the interface to physical simulation
    for inverse parameter estimation and image-driven learning.

    It owns the PDE solver and provides functionality for:
    - mapping elasticity moduli to canonical Lame parameters
    - managing boundary conditions and observed displacements
    - simulating displacement for material parameters and BCs
    - computing displacement loss wrt observed displacements
    - interpolating/rasterizing between mesh and voxel fields
    '''
    def __init__(
        self,
        pde_solver: solvers.PDESolver,
        elastic_params: Tuple[str, str] = ('E', 'nu'),
        default_rho: float = 1e3,
        noise_level: float = 0.,
        random_seed: int = 0,
        use_cache: bool = True
    ):
        self.pde_solver = pde_solver

        elastic_params = tuple(elastic_params)

        if elastic_params not in VALID_ELASTIC_PARAMS:
            raise ValueError(f'Invalid elasticity parameters: {elastic_params}')

        self.elastic_params = elastic_params

        self.default_rho = float(default_rho)
        self.noise_level = float(noise_level)
        self.random_seed = int(random_seed)
        self.use_cache = bool(use_cache)

        self.ctx_cache: Dict[Any, context.PhysicsContext] = {}

    # ----- solver attributes -----

    @property
    def scalar_degree(self) -> int:
        return self.pde_solver.scalar_degree

    @property
    def vector_degree(self) -> int:
        return self.pde_solver.vector_degree

    @property
    def device(self) -> str:
        return self.pde_solver.device

    # ----- public interface -----

    def voxel_simulation_loss(
        self,
        mesh: meshio.Mesh,
        unit_m: float,
        affine: torch.Tensor,
        params: Dict[str, torch.Tensor],
        **kwargs
    ):
        from ..core.interpolation import interpolate_tensor

        ctx = self.get_physics_context(mesh, unit_m)

        affine = affine.to(self.device)
        points = ctx.points[self.scalar_degree].to(self.device)
        voxels = transforms.world_to_voxel_coords(points, affine)

        params = {
            name: interpolate_tensor(volume.to(self.device), voxels)[:,0]
                for name, volume in params.items()
        }
        return self.mesh_simulation_loss(mesh, unit_m, params, **kwargs)

    def mesh_simulation_loss(
        self,
        mesh: meshio.Mesh,
        unit_m: float,
        params: Dict[str, torch.Tensor],
        bc_spec: Optional[Any] = None,
        ret_outputs: bool = False,
        p_obs: float = 1.0
    ):
        ctx = self.get_physics_context(mesh, unit_m)

        mu, lam, rho = self.get_canonical_parameters(ctx, params)
        u_bc, u_obs = self.get_observation_pair(ctx, bc_spec)

        if p_obs is not None and p_obs < 1.0:
            mask = (torch.rand(u_obs.shape[0]) < p_obs).float()
        else:
            mask = torch.ones(u_obs.shape[0], dtype=torch.float)

        self.pde_solver.bind_geometry(ctx.verts, ctx.cells)
        loss, u_sim, res = self.pde_solver.simulate_loss(
            mu, lam, rho, u_bc, u_obs, mask
        )

        if not ret_outputs:
            return loss, None

        outputs = {
            'ctx': ctx,
            'params': {
                key: _as_mesh_field(ctx, val, self.scalar_degree)
                    for key, val in params.items()
            },
            'u_bc': _as_mesh_field(ctx, u_bc, self.vector_degree),
            'u_obs': _as_mesh_field(ctx, u_obs, self.vector_degree),
            'u_sim': _as_mesh_field(ctx, u_sim, self.vector_degree),
            'residual': _as_mesh_field(ctx, res, self.vector_degree),
        }

        return loss, outputs

    def initialize_param_field(
        self,
        mesh: meshio.Mesh,
        unit_m: float,
        fill_value: float = 0.0
    ) -> torch.Tensor:

        ctx = self.get_physics_context(mesh, unit_m)

        if self.scalar_degree == 0:
            shape = ctx.cells.shape[:1]

        elif self.scalar_degree == 1:
            shape = ctx.verts.shape[:1]

        return torch.full(shape, fill_value, requires_grad=True)

    def rasterize_scalar_field(
        self,
        mesh: meshio.Mesh,
        unit_m: float,
        dofs: torch.Tensor,
        shape: Tuple[int, int, int],
        affine: torch.Tensor
    ) -> torch.Tensor:

        ctx = self.get_physics_context(mesh, unit_m)

        lo, hi = transforms.get_grid_bounds(shape, affine)
        bounds = (lo * unit_m, hi * unit_m)

        self.pde_solver.bind_geometry(ctx.verts, ctx.cells)
        raster = self.pde_solver.rasterize_scalar_field(dofs, shape, bounds).cpu()

        # warp only handles spacing, not the full affine
        #   we handle affine orientation by flipping dims
        dims_to_flip = [i + 1 for i, v in enumerate(torch.diag(affine)) if v < 0]

        return torch.flip(raster, dims=dims_to_flip)

    def simulate_displacement(
        self,
        mesh: meshio.Mesh,
        unit_m: float,
        bc_spec: Any,
        params: Optional[Dict[str, torch.Tensor]] = None
    ) -> context.MeshField:

        ctx = self.get_physics_context(mesh, unit_m)

        mu, lam, rho = self.get_canonical_parameters(ctx, params)
        u_bc = self.get_boundary_condition(ctx, bc_spec)

        self.pde_solver.bind_geometry(ctx.verts, ctx.cells)
        u_sim = self.pde_solver.solve_forward(mu, lam, rho, u_bc)

        return _as_mesh_field(ctx, u_sim, self.vector_degree)

    # ----- context lifecycle -----

    def get_physics_context(
        self, mesh: meshio.Mesh, unit_m: float
    ) -> context.PhysicsContext:

        if not self.use_cache:
            return context.PhysicsContext(mesh, unit_m)

        key = (str(mesh.path), round(unit_m, 4))
        if key not in self.ctx_cache:
            self.ctx_cache[key] = context.PhysicsContext(mesh, unit_m)

        return self.ctx_cache[key]

    def clear_cache(self):
        self.ctx_cache.clear()

    # ----- material parameters -----

    def get_canonical_parameters(
        self,
        ctx: context.PhysicsContext,
        overrides: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        rho = self.resolve_material_parameter(ctx, 'rho', overrides)

        moduli = {}
        for key in self.elastic_params:
            moduli[key] = self.resolve_material_parameter(ctx, key, overrides)

        mu, lam = _compute_lame_parameters(moduli)

        _validate_material_parameters(mu, lam, rho)

        return mu, lam, rho

    def resolve_material_parameter(
        self,
        ctx: context.PhysicsContext,
        key: str,
        overrides: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:

        overrides = overrides or {}

        if key in overrides:
            return overrides[key]

        try:
            return ctx.fields[key][self.scalar_degree]
        except (KeyError, IndexError):
            pass

        if key == 'rho':
            if self.scalar_degree == 0:
                shape = ctx.cells.shape[:1]
            elif self.scalar_degree == 1:
                shape = ctx.verts.shape[:1]
            return torch.full(shape, self.default_rho)

        raise KeyError(f'No value provided for parameter: {key}')

    # ----- boundary conditions / observations -----

    def get_boundary_condition(
        self, ctx: context.PhysicsContext, bc_spec: Any
    ) -> torch.Tensor:

        shape = ctx.points[self.vector_degree].shape

        if bc_spec is None or bc_spec.type == 'zero':
            return torch.zeros(shape, dtype=torch.float)

        elif bc_spec.type == 'constant':
            return torch.full(shape, bc_spec.value, dtype=torch.float)

        elif bc_spec.type == 'mesh_key':
            return ctx.fields[bc_spec.value][self.vector_degree]

        raise ValueError(f'Invalid bc_spec: {bc_spec!r}')

    def get_observation_pair(
        self, ctx: context.PhysicsContext, bc_spec: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if bc_spec not in ctx.obs_cache: # simulate and cache
            mu, lam, rho = self.get_canonical_parameters(ctx)
            u_bc = self.get_boundary_condition(ctx, bc_spec)

            self.pde_solver.bind_geometry(ctx.verts, ctx.cells)
            u_sim = self.pde_solver.solve_forward(mu, lam, rho, u_bc)

            ctx.obs_cache[bc_spec] = (
                _as_mesh_field(ctx, u_bc, self.vector_degree),
                _as_mesh_field(ctx, u_sim, self.vector_degree),
            )

        u_bc_field, u_obs_field = ctx.obs_cache[bc_spec]
        u_bc = u_bc_field[self.vector_degree]
        u_obs = u_obs_field[self.vector_degree]

        if self.noise_level is not None:
            u_obs = self.add_observation_noise(u_obs, self.noise_level, self.random_seed)

        return u_bc, u_obs

    def add_observation_noise(self, u_obs, noise_ratio, random_seed=None):
        rng = torch.Generator(device=u_obs.device)
        rng.manual_seed(random_seed)

        u_rms = torch.sqrt(torch.mean(u_obs**2))
        sigma = u_rms * noise_ratio

        noise = torch.randn(*u_obs.shape, generator=rng)
        return u_obs + sigma * noise


def _compute_lame_parameters(
    params: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:

    keys = set(params)

    if keys == {'E', 'nu'}:
        E, nu = (params['E'], params['nu'])
        mu = E / (2*(1 + nu))
        lam = E * nu / ((1 + nu)*(1 - 2*nu))
        return mu, lam

    elif keys == {'E', 'K'}:
        E, K = (params['E'], params['K'])
        mu = 3 * K * E / (9*K - E)
        lam = K - (2/3)*mu
        return mu, lam

    elif keys == {'G', 'K'}:
        G, K = (params['G'], params['K'])
        return G, K - (2/3)*G

    elif keys == {'mu', 'lam'}:
        return params['mu'], params['lam']

    raise KeyError(f'Unsupported elasticity parameters: {keys}')


def _validate_material_parameters(
    mu: torch.Tensor,
    lam: torch.Tensor,
    rho: torch.Tensor,
    max_ratio: float = 1e2
):
    if not torch.all(torch.isfinite(rho)):
        raise ValueError('Non-finite density values (rho)')

    if not torch.all(rho > 0):
        raise ValueError('Non-positive density values (rho)')

    if not torch.all(torch.isfinite(mu)):
        raise ValueError('Non-finite shear modulus values (mu or G)')

    if not torch.all(mu > 0):
        raise ValueError('Non-positive shear modulus values (mu or G)')

    if not torch.all(torch.isfinite(lam)):
        raise ValueError('Non-finite Lame parameter values (lambda)')

    K = lam + (2/3) * mu
    if not torch.all(torch.isfinite(K)):
        raise ValueError('Non-finite bulk modulus values (K)')

    if not torch.all(K > 0):
        raise ValueError('Non-positive bulk modulus values (K; requires nu < 0.5)')

    ratio = K / mu
    if torch.any(ratio > max_ratio):
        utils.warn(f'Material is nearly incompressible (K/G = {ratio.max().item()})')



def _as_mesh_field(
    ctx: context.PhysicsContext,
    values: torch.Tensor,
    degree: int
) -> context.MeshField:

    values = values.detach().cpu()

    if degree == 0:
        cell_values = values
        node_values = transforms.cell_to_node_values(
            cell_values, ctx.volume, ctx.incidence
        )

    elif degree == 1:
        node_values = values
        cell_values = transforms.node_to_cell_values(
            node_values, ctx.incidence
        )

    else:
        raise ValueError(f'Invalid degree: {degree}')

    return context.MeshField(cell_values, node_values)

