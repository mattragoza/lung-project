# preprocessing/stages/meshes.py

from typing import List, Dict, Tuple, Any
from pathlib import Path
import meshio

from ...core import utils, fileio


def repair_surface_mesh(input_path, output_path, config):
    utils.check_keys(
        config,
        valid={'run_pymeshfix'},
        where='surface_mesh'
    )
    from .. import surface_meshing

    mesh = fileio.load_trimesh(input_path).to_mesh()

    utils.log('Repairing surface mesh')
    use_pymeshfix = config.get('run_pymeshfix')
    mesh = surface_meshing.repair_surface_mesh(mesh, use_pymeshfix)
    mesh = meshio.Mesh(points=mesh.vertices, cells=[('triangle', mesh.faces)])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_meshio(output_path, mesh)


def generate_volume_mesh(mask_path, output_path, config, random_seed=0):
    utils.check_keys(
        config,
        valid={'use_affine_spacing', 'mesh_parameters'},
        where='mesh_generation'
    )
    from .. import volume_meshing

    nifti = fileio.load_nibabel(mask_path)

    use_affine = config.get('use_affine_spacing', False)
    pygalmesh_kws = config.get('mesh_parameters', {})

    utils.log('Generating tetrahedral mesh')
    mesh = volume_meshing.generate_mesh_from_mask(
        mask=nifti.get_fdata(),
        affine=nifti.affine,
        use_affine=use_affine,
        random_seed=random_seed,
        pygalmesh_kws=pygalmesh_kws
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_meshio(output_path, mesh)

