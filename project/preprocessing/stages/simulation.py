# preprocessing/stages/simulation.py

from typing import List, Dict, Tuple, Any
from pathlib import Path

from ...core import utils, fileio


def simulate_displacement(
    mesh_path, output_path, unit_m, config, random_seed=0
):
    utils.check_keys(
        config,
        valid={'physics_adapter', 'pde_solver'},
        where='displacement_simulation'
    )
    import numpy as np
    from ... import physics

    mesh = fileio.load_meshio(mesh_path)
    utils.log(mesh)

    physics_adapter_kws = config.get('physics_adapter', {})
    pde_solver_kws = config.get('pde_solver', {}).copy()

    physics_adapter = physics.PhysicsAdapter(
        pde_solver_cls=pde_solver_kws.pop('_class'),
        pde_solver_kws=pde_solver_kws,
        **physics_adapter_kws
    )
    bc_spec = None #physics_adapter.get_bc_spec(random_seed)
    outputs = physics_adapter.simulate(mesh, unit_m, bc_spec)

    for k, v in outputs.items():
        utils.log((k, v.shape, v.dtype, v.mean()))

        if v.shape[0] == mesh.points.shape[0]:
            mesh.point_data[k] = v.astype(np.float32)

        elif v.shape[0] == mesh.cells_dict['tetra'].shape[0]:
            mesh.cell_data[k] = [v.astype(np.float32)]

        else:
            raise ValueError(f'Invalid mesh field shape: {v.shape}')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_meshio(output_path, mesh)

