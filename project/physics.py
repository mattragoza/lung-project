from typing import Optional
from dataclasses import dataclass
from .core import utils, transforms, interpolation
from . import solvers
import meshio
import torch


@dataclass
class MeshField:
    cells: Optional[torch.Tensor] = None
    nodes: Optional[torch.Tensor] = None

    def __getitem__(self, degree: int):
        if degree == 0 and self.cells is not None:
            return self.cells
        if degree == 1 and self.nodes is not None:
            return self.nodes
        raise ValueError(f'No dof values for degree {degree}')

    def to(self, device):
        cells = None if self.cells is None else self.cells.to(device) 
        nodes = None if self.nodes is None else self.nodes.to(device)
        return Dofs(cells, nodes)


class PhysicsContext:

    def __init__(
        self,
        solver: solvers.base.PDESolver,
        mesh:   meshio.Mesh,
        unit_m: float
    ):
        cells = mesh.cells_dict['tetra']
        verts = mesh.points * unit_m # world units to meters

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

        # convert to CPU tensors; solver will manage device
        def _cpu(a, dtype=None):
            return torch.as_tensor(a, dtype=dtype or torch.float, device='cpu')

        self.verts = _cpu(verts) # meters
        self.cells = _cpu(cells, torch.int)

        self.material = MeshField(_cpu(mat_cells, torch.int), None)
        self.points   = MeshField(_cpu(pts_cells), _cpu(pts_nodes)) # world units
        self.rho      = MeshField(_cpu(rho_cells), _cpu(rho_nodes)) # kg/m^3
        self.E        = MeshField(_cpu(E_cells),   _cpu(E_nodes))   # Pa
        self.image    = MeshField(_cpu(img_cells), _cpu(img_nodes))

        self.solver = solver
        self.solver.init_geometry(self.verts, self.cells)

        self._cache = {}


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
        dtype=torch.float32,
        device='cuda'
    ):
        self.nu_value  = float(nu_value)
        self.rho_known = bool(rho_known)
        self.rho_bias  = float(rho_bias) if not rho_known else None

        assert scalar_degree in {0, 1}, scalar_degree
        assert vector_degree in {0, 1}, vector_degree

        self.scalar_degree = scalar_degree
        self.vector_degree = vector_degree

        self.pde_solver_cls = solvers.base.PDESolver.get_subclass(pde_solver_cls)
        self.pde_solver_kws = pde_solver_kws or {}

        self.dtype  = dtype
        self.device = device
        self._cache = {}

    def get_solver(self) -> solvers.base.PDESolver:
        return self.pde_solver_cls(
            scalar_degree=self.scalar_degree,
            vector_degree=self.vector_degree,
            device=self.device,
            **self.pde_solver_kws
        )

    def get_context(self, mesh: meshio.Mesh, unit_m: float) -> PhysicsContext:
        key = id(mesh), str(unit_m)
        if key not in self._cache:
            solver = self.get_solver()
            self._cache[key] = PhysicsContext(solver, mesh, unit_m)
        return self._cache[key]

    def get_bc(self, ctx, bc_spec):
        assert bc_spec is None, 'TODO'
        points = ctx.points[self.vector_degree]
        return torch.zeros_like(points)

    def get_obs(self, ctx, bc_spec):
        if bc_spec not in ctx._cache:
            E = ctx.E[self.scalar_degree]
            rho = ctx.rho[self.scalar_degree]
            u_bc = self.get_bc(ctx, bc_spec)
            loss, out = self._simulate(ctx, E, rho, u_bc)
            ctx._cache[bc_spec] = out['u_sim'].detach().cpu()
        return ctx._cache[bc_spec]

    def simulate_forward(self, mesh, unit_m, bc_spec):
        ctx = self.get_context(mesh, unit_m)
        E = ctx.E[self.scalar_degree]
        rho = ctx.rho[self.scalar_degree]
        u_bc = self.get_bc(ctx, bc_spec)
        return self._simulate(ctx, E, rho, u_bc)

    def simulate_inverse(self, mesh, unit_m, E, bc_spec):
        ctx = self.get_context(mesh, unit_m)
        rho = ctx.rho[self.scalar_degree]
        u_obs = ctx.u_obs(ctx, bc_spec)
        return self._simulate(ctx, E, rho, u_obs)

    def simulate_training(self, mesh, unit_m, affine, E_vox, bc_spec):
        ctx = self.get_context(mesh, unit_m)
        points = ctx.points[self.scalar_degree].to(affine.device)
        voxels = transforms.world_to_voxel_coords(points, affine)
        E = interpolation.interpolate_image(E_vox, voxels)[...,0]
        rho = ctx.rho[self.scalar_degree]
        u_obs = ctx.get_obs(ctx, bc_spec)
        return self._simulate(ctx, E, rho, u_obs)

    def _simulate(self, ctx, E, rho, u_obs):
        inputs = {'E': E, 'rho': rho, 'u_obs': u_obs}
        mu, lam = transforms.compute_lame_parameters(E, self.nu_value)
        outputs = ctx.solver.solve(mu, lam, rho, u_obs)
        return outputs.pop('loss'), self._package(ctx, inputs, outputs)

    def _package(self, ctx, inputs, outputs):
        return {}

