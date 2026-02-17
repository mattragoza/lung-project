from __future__ import annotations
from typing import Optional, List, Dict, Tuple, Iterable, Any
from dataclasses import dataclass
import numpy as np
import meshio
import torch

from .core import utils, transforms, interpolation
from . import solvers


@dataclass
class MeshField:
    cells: Optional[torch.Tensor] = None
    nodes: Optional[torch.Tensor] = None

    def __getitem__(self, degree: int) -> torch.Tensor:
        if degree == 0 and self.cells is not None:
            return self.cells
        if degree == 1 and self.nodes is not None:
            return self.nodes
        raise ValueError(f'No values for degree {degree}')


class PhysicsContext:
    '''
    Stores immutable CPU tensors derived from mesh data.
    '''
    def __init__(self, mesh: meshio.Mesh, unit_m: float):

        # domain geometry
        cells_np = mesh.cells_dict['tetra']
        verts_np = mesh.points * unit_m # meters

        volume_np = transforms.compute_cell_volume(verts_np, cells_np)

        def _cpu(a: np.ndarray, dtype=torch.float):
            return torch.as_tensor(a, dtype=dtype, device='cpu')

        self.cells = _cpu(cells_np, dtype=torch.int)
        self.verts = _cpu(verts_np, dtype=torch.float)

        self.volume = _cpu(volume_np)
        self.adjacency = transforms.compute_node_adjacency(self.verts, self.cells, self.volume)

        # points used for voxel interpolation (world units)
        cell_points = mesh.points[cells_np].mean(axis=1)
        node_points = mesh.points

        self.points = MeshField(_cpu(cell_points), _cpu(node_points))

        # generic mesh-attached fields
        self.fields: Dict[str, MeshField] = {}

        def add_field(name, dtype):
            cell_vals = None
            node_vals = None
            if 'tetra' in mesh.cell_data_dict.get(name, {}):
                cell_vals = _cpu(mesh.cell_data_dict[name]['tetra'], dtype)
            if name in mesh.point_data:
                node_vals = _cpu(mesh.point_data[name], dtype)
            self.fields[name] = MeshField(cell_vals, node_vals)

        # material labels
        add_field('material', dtype=torch.int)

        # material parameters
        for name in {'rho', 'E', 'nu', 'G', 'K', 'mu', 'lam'}:
            add_field(name, dtype=torch.float)

        # boundary condition cache
        self.bc_cache = {}


class PhysicsAdapter:
    '''
    PhysicsAdapter owns the PDE solver and the logic for:
    - deriving/caching displacement observations (u_obs)
    - converting material params to canonical (mu, lam, rho)
    - interpolating between voxel- and mesh-domain params
    - running physics solve/loss and packaging outputs
    '''
    def __init__(
        self,
        default_nu: float,
        default_rho: float,
        scalar_degree: int,
        vector_degree: int,
        pde_solver_cls: str,
        pde_solver_kws=None,
        use_cache: bool=True,
        device: str='cuda'
    ):
        self.default_nu  = float(default_nu)
        self.default_rho = float(default_rho)

        assert scalar_degree in {0, 1}, scalar_degree
        assert vector_degree in {0, 1}, vector_degree

        self.scalar_degree = scalar_degree
        self.vector_degree = vector_degree

        self.device = device

        self.pde_solver_cls = solvers.base.PDESolver.get_subclass(pde_solver_cls)
        self.pde_solver_kws = pde_solver_kws or {}
        self.pde_solver = self.make_pde_solver()

        self.use_cache = use_cache
        self._cache: Dict[Any, PhysicsContext] = {}

    # ----- solver / context lifecycle -----

    def make_pde_solver(self) -> solvers.base.PDESolver:
        return self.pde_solver_cls(
            scalar_degree=self.scalar_degree,
            vector_degree=self.vector_degree,
            device=self.device,
            **self.pde_solver_kws
        )

    def get_pde_context(self, mesh: meshio.Mesh, unit_m: float) -> PhysicsContext:
        if not self.use_cache:
            return PhysicsContext(mesh, unit_m)
        key = (str(mesh.path), round(unit_m, 4))
        if key not in self._cache:
            self._cache[key] = PhysicsContext(mesh, unit_m)
        return self._cache[key]

    def clear_cache(self):
        self._cache.clear()

    # ----- material parameters -----

    def get_material_params(
        self,
        ctx: PhysicsContext
    ) -> Dict[str, torch.Tensor]:
        return {
            'E': ctx.fields['E'][self.scalar_degree],
            'rho': ctx.fields['rho'][self.scalar_degree]
        }

    def get_canonical_params(
        self,
        ctx: PhysicsContext,
        params: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        params = dict(params)
        rho = params.pop('rho', None)
        if rho is None:
            template = next(iter(params.values()))
            rho = torch.full_like(template, self.default_rho)

        elast_keys = tuple(sorted(params.keys()))

        if elast_keys == ('E',):
            E, nu = params['E'], self.default_nu
            mu = E / (2*(1 + nu))
            lam = E * nu / ((1 + nu)*(1 - 2*nu))

        elif elast_keys == ('E', 'nu'):
            E, nu = params['E'], params['nu']
            mu = E / (2*(1 + nu))
            lam = E * nu / ((1 + nu)*(1 - 2*nu))

        elif elast_keys == ('G', 'K'):
            G, K = params['G'], params['K']
            mu, lam = G, K - (2/3)*G

        elif elast_keys == ('mu', 'lam'):
            mu, lam = params['mu'], params['lam']
        else:
            raise ValueError(f'Unsupported elasticity keys: {elast_keys}')

        return mu, lam, rho

    # ----- boundary conditions -----

    def get_boundary_condition(self, ctx: PhysicsContext, bc_spec: Any) -> torch.Tensor:
        template = ctx.points[self.vector_degree]
        return torch.zeros_like(template) # TODO implement different BCs

    # ----- displacement observations -----

    def get_observations(
        self,
        ctx: PhysicsContext,
        bc_spec: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if bc_spec in ctx.bc_cache:
            return ctx.bc_cache[bc_spec]

        params = self.get_material_params(ctx)
        mu, lam, rho = self.get_canonical_params(ctx, params)
        u_bc = self.get_boundary_condition(ctx, bc_spec)

        self.pde_solver.bind_geometry(ctx.verts, ctx.cells)
        u_obs = self.pde_solver.solve(mu, lam, rho, u_bc)

        ctx.bc_cache[bc_spec] = (u_bc.cpu(), u_obs.detach().cpu())
        return ctx.bc_cache[bc_spec]

    # ----- public API -----

    def init_param_field(
        self,
        mesh: meshio.Mesh,
        unit_m: float,
        fill_value: float = 0.0
    ) -> torch.Tensor:

        ctx = self.get_pde_context(mesh, unit_m)
        params = self.get_material_params(ctx)
        template = next(iter(params.values()))
        return torch.full_like(template, fill_value, requires_grad=True)

    def simulate(
        self,
        mesh: meshio.Mesh,
        unit_m: float,
        bc_spec: Any,
        params: Dict[str, torch.Tensor] = None
    ):
        ctx = self.get_pde_context(mesh, unit_m)
        if params is None:
            params = self.get_material_params(ctx)

        u_bc = self.get_boundary_condition(ctx, bc_spec)
        mu, lam, rho = self.get_canonical_params(ctx, params)

        self.pde_solver.bind_geometry(ctx.verts, ctx.cells)
        u_sim = self.pde_solver.solve(mu, lam, rho, u_bc)

        def _numpy(t: torch.Tensor) -> np.ndarray:
            return t.detach().cpu().numpy()

        return {'u_bc': _numpy(u_bc), 'u_sim': _numpy(u_sim)}

    def mesh_simulation_loss(
        self,
        mesh: meshio.Mesh,
        unit_m: float,
        params: Dict[str, torch.Tensor],
        bc_spec: Any,
        ret_outputs: bool = False,
        p_obs: float = 1.0
    ):
        ctx = self.get_pde_context(mesh, unit_m)
        u_bc, u_obs = self.get_observations(ctx, bc_spec)
        mu, lam, rho = self.get_canonical_params(ctx, params)

        if p_obs < 1.0:
            mask = (torch.rand(u_obs.shape[0]) < p_obs).float()
        else:
            mask = torch.ones(u_obs.shape[0], dtype=torch.float)

        self.pde_solver.bind_geometry(ctx.verts, ctx.cells)

        loss, outputs = self.pde_solver.loss(mu, lam, rho, u_bc, u_obs, mask)

        if not ret_outputs:
            return loss

        true_params = self.get_material_params(ctx)
        mu_t, lam_t, rho_t = self.get_canonical_params(ctx, true_params)
    
        return loss, self._package_outputs(
            ctx,
            true_native=true_params,
            pred_native=params,
            mu_true=mu_t,
            lam_true=lam_t,
            rho_true=rho_t,
            mu_pred=mu,
            lam_pred=lam, 
            rho_pred=rho,
            u_true=u_obs,
            u_pred=outputs['u_sim'],
            pde_res=outputs['res'],
        )

    def voxel_simulation_loss(
        self,
        mesh: meshio.Mesh,
        unit_m: float,
        affine: torch.Tensor,
        params: Dict[str, torch.Tensor],
        **kwargs
    ):
        ctx = self.get_pde_context(mesh, unit_m)
        params = self.interpolate_voxel_params(ctx, affine, params)
        return self.mesh_simulation_loss(mesh, unit_m, params, **kwargs)

    def interpolate_voxel_params(
        self,
        ctx: PhysicsContext,
        affine: torch.Tensor,
        params: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:

        points = ctx.points[self.scalar_degree].to(self.device) # world units
        affine = affine.to(self.device) # voxel -> world mapping
        voxels = transforms.world_to_voxel_coords(points, affine)
        interp = lambda x: interpolation.interpolate_image(x, voxels)[:,0]
        return {k: interp(v.to(self.device)) for k, v in params.items()}

    def rasterize_scalar_field(
        self,
        mesh: meshio.Mesh,
        unit_m: float,
        dofs: torch.Tensor,
        shape: Tuple[int, int, int],
        affine: torch.Tensor
    ) -> torch.Tensor:

        ctx = self.get_pde_context(mesh, unit_m)
        self.pde_solver.bind_geometry(ctx.verts, ctx.cells)
        bounds = transforms.get_grid_bounds(shape, affine, unit_m)
        return self.pde_solver.rasterize_scalar_field(dofs, shape, bounds).cpu()

    # ----- output packaging -----

    def _package_outputs(
        self,
        ctx: PhysicsContext,
        true_native: Dict[str, torch.Tensor],
        pred_native: Dict[str, torch.Tensor],
        mu_true: torch.Tensor,
        mu_pred: torch.Tensor,
        lam_true: torch.Tensor,
        lam_pred: torch.Tensor,
        rho_true: torch.Tensor,
        rho_pred: torch.Tensor,
        u_true: torch.Tensor,
        u_pred: torch.Tensor,
        pde_res: torch.Tensor
    ) -> Dict[str, Any]:

        ret = {
            'volume':   ctx.volume,
            'material': ctx.fields.get('material'),
            'mu_pred':  _as_mesh_field(ctx, mu_pred, self.scalar_degree),
            'mu_true':  _as_mesh_field(ctx, mu_true, self.scalar_degree),
            'lam_pred': _as_mesh_field(ctx, lam_pred, self.scalar_degree),
            'lam_true': _as_mesh_field(ctx, lam_true, self.scalar_degree),
            'rho_pred': _as_mesh_field(ctx, rho_pred, self.scalar_degree),
            'rho_true': _as_mesh_field(ctx, rho_true, self.scalar_degree),
            'u_pred':   _as_mesh_field(ctx, u_pred, self.vector_degree),
            'u_true':   _as_mesh_field(ctx, u_true, self.vector_degree),
            'residual': _as_mesh_field(ctx, pde_res, self.vector_degree)
        }
        for name in pred_native:
            ret[f'{name}_pred'] = _as_mesh_field(ctx, pred_native[name], self.scalar_degree)
        for name in true_native:
            ret[f'{name}_true'] = _as_mesh_field(ctx, true_native[name], self.scalar_degree)

        return ret


def _as_mesh_field(
    ctx: PhysicsContext,
    values: torch.Tensor,
    degree: int
) -> MeshField:
    '''
    Convert values at cell or node dofs into both representations.
    '''
    if degree == 0:
        cell_vals = values.detach().cpu()
        node_vals = transforms.cell_to_node_values(ctx.verts, ctx.cells, cell_vals, ctx.volume)
    elif degree == 1:
        node_vals = values.detach().cpu()
        cell_vals = transforms.node_to_cell_values(ctx.cells, node_vals)
    else:
        raise ValueError(f'Cannot convert degree {degree}')
    return MeshField(cell_vals, node_vals)

