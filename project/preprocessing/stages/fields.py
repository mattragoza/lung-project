# preprocessing/stages/fields.py

from typing import List, Dict, Tuple, Any
from pathlib import Path
import numpy as np

from ...core import utils, fileio, transforms, interpolation


def interpolate_materials(
    regions_path, materials_path, mesh_path, output_path, config
):
    utils.check_keys(config, valid={'material_catalog'}, where='material_mesh')
    from .. import materials

    reg_mask = fileio.load_nibabel(regions_path).get_fdata().astype(int)
    mat_mask = fileio.load_nibabel(materials_path).get_fdata().astype(int)
    mesh = fileio.load_meshio(mesh_path)

    # infer the region -> material label map from masks
    region_to_material = materials.infer_material_by_region(reg_mask, mat_mask)

    # assign material labels to cells using region cell labels
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
    cells_to_nodes = transforms.compute_node_adjacency(verts, cells, volume)
    mesh.point_data['mat'] = transforms.cell_to_node_labels(verts, cells, mat_cells)
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


def interpolate_mesh_fields(
    mesh_path, image_path, disp_path, output_path, config
):
    utils.check_keys(config, valid={}, where='mesh_interpolation')
    from .. import image_synthesis

    mesh = fileio.load_meshio(mesh_path)

    nifti = fileio.load_nibabel(image_path)
    image = nifti.get_fdata().astype(float)
    affine = nifti.affine

    disp = fileio.load_nibabel(disp_path).get_fdata()

    utils.log('Interpolating fields onto mesh vertices')
    pts_world = mesh.points
    pts_voxel = transforms.world_to_voxel_coords(pts_world, affine)

    img_values = image_synthesis.interpolate_volume(image, pts_voxel, **config)
    mesh.point_data['image'] = img_values.astype(np.float32)

    disp_values = image_synthesis.interpolate_volume(disp, pts_voxel, **config)
    mesh.point_data['u_true'] = disp_values.astype(np.float32)

    utils.log('Interpolating fields onto cell centers')
    pts_world = mesh.points[mesh.cells_dict['tetra']].mean(axis=1)
    pts_voxel = transforms.world_to_voxel_coords(pts_world, affine)

    img_values = image_synthesis.interpolate_volume(image, pts_voxel, **config)
    mesh.cell_data['image'] = [img_values.astype(np.float32)]

    disp_values = image_synthesis.interpolate_volume(disp, pts_voxel, **config)
    mesh.cell_data['u_true'] = [disp_values.astype(np.float32)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_meshio(output_path, mesh)

