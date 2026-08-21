from typing import Optional, Dict, Tuple, Any
from dataclasses import dataclass

import meshio
import numpy as np
import torch

from . import bcs

from ..core import transforms


def _cpu_tensor(array, dtype=torch.float):
    return torch.as_tensor(array, dtype=dtype, device='cpu')


def _get_mesh_field(mesh, name, dtype=torch.float, scale=1.0):
    cell_values = node_values = None

    cell_data = mesh.cell_data_dict.get(name, {})
    if 'tetra' in cell_data:
        cell_values = _cpu_tensor(cell_data['tetra'] * scale, dtype)

    if name in mesh.point_data:
        node_values = _cpu_tensor(mesh.point_data[name] * scale, dtype)

    if cell_values is None and node_values is None:
        return None

    return MeshField(cell_values, node_values)


@dataclass
class MeshField:
    cell_values: Optional[torch.Tensor] = None
    node_values: Optional[torch.Tensor] = None

    def __getitem__(self, degree: int) -> torch.Tensor:
        if degree == 0 and self.cell_values is not None:
            return self.cell_values
        if degree == 1 and self.node_values is not None:
            return self.node_values
        raise IndexError(f'No values for degree {degree}')


class PhysicsContext:
    '''
    Stores immutable CPU-side representations of geometry,
    material fields, and displacement observations used by 
    the physics adapter and PDE solver.

    Unit contract
    -------------
    Input mesh:
        mesh.points     : world coordinates
        u_* fields      : world displacement units

    Attributes:
        self.verts      : meters (for physical simulation)
        self.volume     : cubic meters
        self.points     : world units (for voxel sampling)
        self.fields:
            u_*         : meters
            rho         : kg/m^3
            E, G, K, mu, lam
                        : Pa
            nu          : unitless

    NOTE: The input mesh is expected to live in the "world"
    coordinate system; the unit_m argument gives the mapping
    from world units to meters. This factor applies to both
    mesh vertices and any displacement field(s) on the mesh.
    '''
    def __init__(self, mesh: meshio.Mesh, unit_m: float):

        # ----- mesh topology and geometry -----

        cells = mesh.cells_dict['tetra']
        verts_w = mesh.points
        verts_m = verts_w * unit_m
        volume_m3 = transforms.compute_cell_volume(verts_m, cells)

        self.cells = _cpu_tensor(cells, dtype=torch.int)
        self.verts = _cpu_tensor(verts_m, dtype=torch.float)
        self.volume = _cpu_tensor(volume_m3, dtype=torch.float)

        self.adjacency = transforms.compute_incidence_matrix(
            verts=self.verts,
            cells=self.cells,
            volume=self.volume
        )

        # ----- world-space sampling points -----

        self.points = MeshField(
            cell_values=_cpu_tensor(verts_w[cells].mean(axis=1)),
            node_values=_cpu_tensor(verts_w)
        )

        # ----- material properties / labels -----

        self.fields: Dict[str, MeshField] = {}

        for name in {'region', 'material'}:
            field = _get_mesh_field(mesh, name, dtype=torch.int)
            if field is not None:
                self.fields[name] = field

        for name in {'rho', 'E', 'nu', 'G', 'K', 'mu', 'lam'}:
            field = _get_mesh_field(mesh, name, dtype=torch.float)
            if field is not None:
                self.fields[name] = field

        # ----- displacement observations -----

        self.obs_cache: Dict[Any, Tuple[MeshField, MeshField]] = {}

        for name in sorted(mesh.cell_data_dict | mesh.point_data):
            if not name.startswith('u_'):
                continue

            field = _get_mesh_field(
                mesh, name, dtype=torch.float, scale=unit_m
            )
            if field is None:
                continue

            self.fields[name] = field

            bc_spec = bcs.BoundaryConditionSpec(type='mesh_key', value=name)
            self.obs_cache[bc_spec] = (field, field)

