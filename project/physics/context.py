from typing import Optional, Dict, Tuple, Any
from dataclasses import dataclass

import meshio
import numpy as np
import torch

from . import bcs

from ..core import transforms


@dataclass
class MeshField:
    cells: Optional[torch.Tensor] = None
    nodes: Optional[torch.Tensor] = None

    def __getitem__(self, degree: int) -> torch.Tensor:
        if degree == 0 and self.cells is not None:
            return self.cells
        if degree == 1 and self.nodes is not None:
            return self.nodes
        raise IndexError(f'No values for degree {degree}')


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
        cell_points = _cpu(mesh.points[cells_np].mean(axis=1))
        node_points = _cpu(mesh.points)

        self.points = MeshField(cell_points, node_points)

        # generic mesh-attached fields
        self.fields: Dict[str, MeshField] = {}

        def _add_field(name, dtype) -> bool:
            cell_vals = node_vals = None
            if 'tetra' in mesh.cell_data_dict.get(name, {}):
                cell_vals = _cpu(mesh.cell_data_dict[name]['tetra'], dtype)
            if name in mesh.point_data:
                node_vals = _cpu(mesh.point_data[name], dtype)
            if cell_vals is None and node_vals is None:
                return False
            self.fields[name] = MeshField(cell_vals, node_vals)
            return True

        # categorical labels
        for name in {'region', 'material'}:
            _add_field(name, dtype=torch.int)

        # material parameters
        for name in {'rho', 'E', 'nu', 'G', 'K', 'mu', 'lam'}:
            _add_field(name, dtype=torch.float)

        # observation cache: bc_spec -> (u_bc, u_obs)
        self.obs_cache: Dict[Any, Tuple[MeshField, MeshField]] = {}

        if _add_field('u_true', dtype=torch.float):
            bc_spec = bcs.BoundaryConditionSpec(name='u_true')
            u_bc_field = u_obs_field = self.fields['u_true'] 
            self.obs_cache[bc_spec] = (u_bc_field, u_obs_field)

