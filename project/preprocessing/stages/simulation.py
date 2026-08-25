# preprocessing/stages/simulation.py

from typing import List, Dict, Tuple, Any
from pathlib import Path

from ...core import utils, fileio


def simulate_displacement_field(
    mesh_path: Path,
    output_path: Path,
    unit_m: float,
    config: Dict[str, Any]
):
    utils.check_keys(
        config,
        valid={'physics_adapter', 'pde_solver', 'boundary_condition', 'output_key'},
        where='displacement_simulation'
    )
    from ... import physics

    mesh = fileio.load_meshio(mesh_path) # world coordinates
    utils.log(mesh)

    if len(mesh.cells) != 1 or mesh.cells[0].type != 'tetra':
        block_types = [block.type for block in mesh.cells]
        raise ValueError(f'Expected exactly one tetra cell block: {block_types}')

    adapter = physics.api.get_adapter(config)
    bc_spec = physics.api.get_bc_spec(config)

    u_sim_field = adapter.simulate_displacement(mesh, unit_m, bc_spec) # meters

    cell_values = u_sim_field.cell_values.detach().cpu().numpy() / unit_m
    node_values = u_sim_field.node_values.detach().cpu().numpy() / unit_m

    output_key = config.get('output_key', 'u')
    mesh.cell_data[output_key] = [cell_values]
    mesh.point_data[output_key] = node_values

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_meshio(output_path, mesh)

