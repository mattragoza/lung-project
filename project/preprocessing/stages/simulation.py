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

    mesh = fileio.load_meshio(mesh_path)
    utils.log(mesh)

    if len(mesh.cells) != 1 or mesh.cells[0].type != 'tetra':
        block_types = [b.type for b in mesh.cells]
        raise ValueError(f'Expected exactly one tetra cell block: {block_types}')

    adapter = physics.api.get_adapter(config)
    bc_spec = physics.api.get_bc_spec(config)

    u_sim = adapter.simulate(mesh, unit_m, bc_spec)

    output_key = config.get('output_key', 'u')
    mesh.point_data[output_key] = u_sim.nodes.detach().cpu().numpy()
    mesh.cell_data[output_key] = [u_sim.cells.detach().cpu().numpy()]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_meshio(output_path, mesh)

