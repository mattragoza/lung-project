# preprocessing/stages/fields.py

from typing import List, Dict, Tuple, Any
from pathlib import Path
import numpy as np

from ...core import utils, fileio, transforms, interpolation


def interpolate_mesh_fields(
    mesh_path: Path,
    image_path: Path,
    disp_path: Path,
    fields_dir: Path,
    output_path: Path,
    config: Dict[str, Any]
):
    utils.check_keys(
        config,
        valid={'displacement_key', 'interpolate_args'},
        where='mesh_interpolation'
    )

    mesh = fileio.load_meshio(mesh_path)
    if len(mesh.cells) != 1 or mesh.cells[0].type != 'tetra':
        block_types = [b.type for b in mesh.cells]
        raise ValueError(f'Expected exactly one tetra cell block: {block_types}')

    nifti = fileio.load_nibabel(image_path)
    image = nifti.get_fdata(dtype=np.float32)

    disp = fileio.load_nibabel(disp_path).get_fdata(dtype=np.float32)
    if disp.ndim != 4 or disp.shape[-1] != 3:
        raise ValueError(f'Invalid displacement field shape: {disp.shape}')

    density = fileio.load_nibabel(fields_dir / 'density.nii.gz').get_fdata(dtype=np.float32)
    elastic = fileio.load_nibabel(fields_dir / 'youngs_modulus.nii.gz').get_fdata(dtype=np.float32)
    poisson = fileio.load_nibabel(fields_dir / 'poisson_ratio.nii.gz').get_fdata(dtype=np.float32)

    if density.ndim != 3:
        raise ValueError(f'Invalid density field shape: {density.shape}')
    if elastic.ndim != 3:
        raise ValueError(f'Invalid elastic field shape: {elastic.shape}')
    if poisson.ndim != 3:
        raise ValueError(f'Invalid poisson field shape: {poisson.shape}')

    utils.log(f'Interpolating fields onto mesh vertices')

    node_voxels = transforms.world_to_voxel_coords(mesh.points, nifti.affine)

    u_key = config.get('displacement_key', 'u')
    kwargs = config.get('interpolate_args', {})

    mesh.point_data['image'] = interpolation.interpolate_array(image, node_voxels, **kwargs)
    mesh.point_data[u_key] = interpolation.interpolate_array(disp, node_voxels, **kwargs)
    mesh.point_data['E'] = interpolation.interpolate_array(elastic, node_voxels, **kwargs)
    mesh.point_data['nu'] = interpolation.interpolate_array(poisson, node_voxels, **kwargs)
    mesh.point_data['rho'] = interpolation.interpolate_array(density, node_voxels, **kwargs)

    utils.log(f'Interpolating fields onto tetra cell centers')

    tetra_cells = mesh.cells_dict['tetra']
    cell_voxels = node_voxels[tetra_cells].mean(axis=1)

    mesh.cell_data['image'] = [interpolation.interpolate_array(image, cell_voxels, **kwargs)]
    mesh.cell_data[u_key] = [interpolation.interpolate_array(disp, cell_voxels, **kwargs)]
    mesh.cell_data['E'] = [interpolation.interpolate_array(elastic, cell_voxels, **kwargs)]
    mesh.cell_data['nu'] = [interpolation.interpolate_array(poisson, cell_voxels, **kwargs)]
    mesh.cell_data['rho'] = [interpolation.interpolate_array(density, cell_voxels, **kwargs)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_meshio(output_path, mesh)


def interpolate_materials(
    mesh_path: Path,
    regions_path: Path,
    materials_path: Path,
    output_path: Path,
    config: Dict[str, Any]
):
    utils.check_keys(config, valid={'material_catalog'}, where='material_mesh')
    from .. import materials

    mesh = fileio.load_meshio(mesh_path)
    reg_map = fileio.load_nibabel(regions_path).get_fdata().astype(int)
    mat_map = fileio.load_nibabel(materials_path).get_fdata().astype(int)

    # infer the region -> material label mapping from masks
    region_to_material = materials.infer_material_by_region(reg_map, mat_map)

    # assign material labels to cells using their region labels
    reg_cells = mesh.cell_data_dict['region']['tetra'].astype(int)
    mat_cells = region_to_material[reg_cells]
    mesh.cell_data['material'] = [mat_cells]

    utils.log('Loading material catalog')
    mat_df = materials.load_material_catalog(config['material_catalog'])
    utils.log(mat_df)

    # get material properties from material labels and assign to cells
    rho_cells, nu_cells, E_cells = materials.assign_material_properties(mat_cells, mat_df)
    mesh.cell_data['rho'] = [rho_cells]
    mesh.cell_data['nu']  = [nu_cells]
    mesh.cell_data['E']   = [E_cells]

    # map material properties from cells to nodes
    verts, cells = mesh.points, mesh.cells_dict['tetra']
    volume = transforms.compute_cell_volume(verts, cells)
    cells_to_nodes = transforms.compute_incidence_matrix(verts, cells, volume)
    mesh.point_data['material'] = transforms.cell_to_node_labels(verts, cells, mat_cells)
    mesh.point_data['rho'] = transforms.cell_to_node_values(cells_to_nodes, rho_cells)
    mesh.point_data['nu']  = transforms.cell_to_node_values(cells_to_nodes, nu_cells)
    mesh.point_data['E']   = transforms.cell_to_node_values(cells_to_nodes, E_cells)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_meshio(output_path, mesh)


def interpolate_image(
    image_path, mesh_path, output_path, config
):
    utils.check_keys(config, valid={'order', 'mode'}, where='image_interpolation')
    from .. import image_synthesis

    nifti = fileio.load_nibabel(image_path)
    image = nifti.get_fdata().astype(float)
    affine = nifti.affine

    mesh = fileio.load_meshio(mesh_path)

    utils.log('Interpolating image at mesh vertices')
    pts_world = mesh.points
    pts_voxel = transforms.world_to_voxel_coords(pts_world, affine)
    values = image_synthesis.interpolate_volume(image, pts_voxel, **config)
    mesh.point_data['image'] = values.astype(np.float32)

    utils.log('Interpolating image at tetra cell centroids')
    pts_world = mesh.points[mesh.cells_dict['tetra']].mean(axis=1)
    pts_voxel = transforms.world_to_voxel_coords(pts_world, affine)
    values = image_synthesis.interpolate_volume(image, pts_voxel, **config)
    mesh.cell_data['image'] = [values.astype(np.float32)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_meshio(output_path, mesh)




