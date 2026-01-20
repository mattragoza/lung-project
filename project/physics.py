from __future__ import annotations
from typing import Optional, Tuple, Any
from dataclasses import dataclass
import meshio
import torch

from .core import utils, transforms, interpolation
from . import solvers


@dataclass
class MeshField:
    cells: Optional[torch.Tensor] = None
    nodes: Optional[torch.Tensor] = None

    def __getitem__(self, degree: int):
        if degree == 0 and self.cells is not None:
            return self.cells
        if degree == 1 and self.nodes is not None:
            return self.nodes
        raise ValueError(f'No values for degree {degree}')


class PhysicsContext:

    def __init__(self, mesh: meshio.Mesh, unit_m: float):

        cells = mesh.cells_dict['tetra']
        verts = mesh.points * unit_m # world units -> meters

        volume = transforms.compute_cell_volume(verts, cells)
        adjacency = transforms.compute_node_adjacency(verts, cells, volume)

        # world points for interpolating voxel fields
        pts_nodes = mesh.points
        pts_cells = mesh.points[cells].mean(axis=1)

        # ground truth material labels and properties
        mat_cells = mesh.cell_data_dict['material']['tetra']
        rho_cells = mesh.cell_data_dict['rho']['tetra'] # kg/m^3
        rho_nodes = mesh.point_data['rho']
        E_cells = mesh.cell_data_dict['E']['tetra'] # Pa
        E_nodes = mesh.point_data['E']

        # image values interpolated on mesh points
        img_cells = mesh.cell_data_dict['image']['tetra']
        img_nodes = mesh.point_data['image']

        imf_cells = (img_cells + transforms.node_to_cell_values(cells, img_nodes)) / 2
        imf_nodes = (img_nodes + transforms.cell_to_node_values(adjacency, img_cells)) / 2

        # store CPU tensors- let adapter/solver manage device
        def _cpu(a, dtype=None):
            return torch.as_tensor(a, dtype=dtype or torch.float, device='cpu')

        self.verts = _cpu(verts) # meters
        self.cells = _cpu(cells, torch.int)

        self.volume    = _cpu(volume)
        self.adjacency = transforms.compute_node_adjacency(self.verts, self.cells, self.volume)

        self.material = MeshField(_cpu(mat_cells, torch.int), None)
        self.points   = MeshField(_cpu(pts_cells), _cpu(pts_nodes)) # world units
        self.rho      = MeshField(_cpu(rho_cells), _cpu(rho_nodes)) # kg/m^3
        self.E        = MeshField(_cpu(E_cells),   _cpu(E_nodes))   # Pa
        self.image    = MeshField(_cpu(img_cells), _cpu(img_nodes))
        self.image_f  = MeshField(_cpu(imf_cells), _cpu(imf_nodes))

        self.bc_cache = {}


class PhysicsAdapter:

    def __init__(
        self,
        nu_value: float,
        rho_known: bool,
        scalar_degree: int,
        vector_degree: int,
        pde_solver_cls: str,
        pde_solver_kws=None,
        rho_bias: float=None,
        device='cuda',
        cache=False
    ):
        self.nu_value  = float(nu_value)
        self.rho_known = bool(rho_known)
        self.rho_bias  = float(rho_bias) if not rho_known else None

        assert scalar_degree in {0, 1}, scalar_degree
        assert vector_degree in {0, 1}, vector_degree
        self.scalar_degree = scalar_degree
        self.vector_degree = vector_degree
        self.device = device

        self.pde_solver_cls = solvers.base.PDESolver.get_subclass(pde_solver_cls)
        self.pde_solver_kws = pde_solver_kws or {}
        self.pde_solver = self.make_pde_solver()

        self.ctx_cache = {} if cache else None

    # ----- context / solver helpers -----

    def make_pde_solver(self) -> solvers.base.PDESolver:
        return self.pde_solver_cls(
            scalar_degree=self.scalar_degree,
            vector_degree=self.vector_degree,
            device=self.device,
            **self.pde_solver_kws
        )

    def get_context(self, mesh: meshio.Mesh, unit_m: float) -> PhysicsContext:
        if self.ctx_cache is None:
            return PhysicsContext(mesh, unit_m)
        key = (str(mesh.path), round(unit_m, 4))
        if key not in self.ctx_cache:
            self.ctx_cache[key] = PhysicsContext(mesh, unit_m)
        return self.ctx_cache[key]

    def clear_cache(self):
        self.ctx_cache.clear()

    # ----- material / BCs / observations -----

    def get_density(self, ctx: PhysicsContext) -> torch.Tensor:
        if self.rho_known:
            return ctx.rho[self.scalar_degree]
        return ctx.image_f[self.scalar_degree] * self.rho_bias

    def get_boundary_condition(self, ctx, bc_spec) -> torch.Tensor:
        template = ctx.points[self.vector_degree]
        return torch.zeros_like(template)

    def get_observations(self, ctx, bc_spec) -> Tuple[torch.Tensor, torch.Tensor]:
        if bc_spec not in ctx.bc_cache:
            E = ctx.E[self.scalar_degree]
            rho = ctx.rho[self.scalar_degree]
            u_bc = self.get_boundary_condition(ctx, bc_spec)
            mu, lam = transforms.compute_lame_parameters(E, self.nu_value)
            self.pde_solver.bind_geometry(ctx.verts, ctx.cells)
            u_obs = self.pde_solver.solve(mu, lam, rho, u_bc)
            ctx.bc_cache[bc_spec] = (u_bc.cpu(), u_obs.detach().cpu())
        return ctx.bc_cache[bc_spec]

    # ----- public API methods -----

    def init_param_field(self, mesh: meshio.Mesh, unit_m: float, fill_value: float=0.0):
        ctx = self.get_context(mesh, unit_m)
        template = ctx.E[self.scalar_degree]
        return torch.full_like(template, fill_value, requires_grad=True)

    def simulate(self, mesh: meshio.Mesh, unit_m: float, bc_spec: Any):
        ctx = self.get_context(mesh, unit_m)
        E = ctx.E[self.scalar_degree]
        rho = ctx.rho[self.scalar_degree]
        u_bc = self.get_boundary_condition(ctx, bc_spec)
        mu, lam = transforms.compute_lame_parameters(E, self.nu_value)
        self.pde_solver.bind_geometry(ctx.verts, ctx.cells)
        u_sim = self.pde_solver.solve(mu, lam, rho, u_bc)
        return {
            'u_bc': u_bc.detach().cpu().numpy(),
            'u_sim': u_sim.detach().cpu().numpy()
        }

    def simulation_loss(
        self,
        mesh: meshio.Mesh,
        unit_m: float,
        E: torch.Tensor,
        bc_spec: Any,
        ret_outputs: bool=False
    ):
        ctx = self.get_context(mesh, unit_m)
        rho = self.get_density(ctx)
        u_bc, u_obs = self.get_observations(ctx, bc_spec)
        mu, lam = transforms.compute_lame_parameters(E, self.nu_value)
        self.pde_solver.bind_geometry(ctx.verts, ctx.cells)
        loss, outputs = self.pde_solver.loss(mu, lam, rho, u_bc, u_obs)
        if ret_outputs:
            inputs = {'E': E, 'rho': rho, 'u_bc': u_bc, 'u_obs': u_obs}
            return loss, self._package(ctx, inputs, outputs)
        return loss

    def voxel_simulation_loss(
        self,
        mesh: meshio.Mesh,
        unit_m: float,
        affine: torch.Tensor,
        E_vox: torch.Tensor,
        bc_spec: Any,
        ret_outputs: bool=True
    ):
        ctx = self.get_context(mesh, unit_m)
        points = ctx.points[self.scalar_degree].to(self.device)
        voxels = transforms.world_to_voxel_coords(points, affine.to(self.device))
        E = interpolation.interpolate_image(E_vox.to(self.device), voxels)[...,0]
        return self.simulation_loss(mesh, unit_m, E, bc_spec, ret_outputs)

    # ----- packaging for evaluation -----

    def _package(self, ctx, inputs, outputs):
        return {
            'volume':   ctx.volume,
            'material': ctx.material,
            'rho_true': ctx.rho,
            'E_true':   ctx.E,
            'rho_pred': _as_mesh_field(ctx, inputs['rho'], self.scalar_degree),
            'E_pred':   _as_mesh_field(ctx, inputs['E'],   self.scalar_degree),
            'u_true':   _as_mesh_field(ctx, inputs['u_obs'],  self.vector_degree),
            'u_pred':   _as_mesh_field(ctx, outputs['u_sim'], self.vector_degree),
            'residual': _as_mesh_field(ctx, outputs['res'],   self.vector_degree)
        }


def _as_mesh_field(ctx, values, degree: int) -> MeshField:
    if degree == 0:
        cell_vals = values.detach().cpu()
        node_vals = transforms.cell_to_node_values(ctx.verts, ctx.cells, cell_vals, ctx.volume)
    elif degree == 1:
        node_vals = values.detach().cpu()
        cell_vals = transforms.node_to_cell_values(ctx.cells, node_vals)
    else:
        raise ValueError(f'Cannot convert degree {degree}')
    return MeshField(cell_vals, node_vals)

