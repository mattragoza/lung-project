# preprocessing/operations/field_interpolation.py

from typing import Dict, Any

import numpy as np

from ...common import transforms, interpolation


def interpolate_mesh_fields(
    mesh: 'meshio.Mesh',
    fields: Dict[str, np.ndarray],
    affine: np.ndarray,
    interp_kws: Dict[str, Any]
) -> 'meshio.Mesh':
    '''
    Interpolate voxel fields at mesh nodes and cell centers.

    Args:
        mesh: meshio.Mesh with single tetrahedral cell block
        fields: dict of (I, J, K) or (I, J, K, C) voxel fields
        affine: voxel to world affine transformation matrix
        interp_kws: kwargs to interpolation.interpolate_array
    Returns:
        mesh: meshio.Mesh
    '''
    from ...common.interpolation import interpolate_array

    cell_blocks = [block.type for block in mesh.cells]
    if cell_blocks != ['tetra']:
        raise ValueError(f'Expected one tetra cell block: {cell_blocks!r}')

    node_voxels = transforms.world_to_voxel_coords(mesh.points, affine)
    cell_voxels = node_voxels[mesh.cells_dict['tetra']].mean(axis=1)

    for name, array in fields.items():
        array = np.asarray(array)
        if array.ndim not in {3, 4}:
            raise ValueError(f'Invalid field shape for {name!r}: {array.shape}')

        mesh.point_data[name] = interpolate_array(array, node_voxels, **interp_kws)
        mesh.cell_data[name] = [interpolate_array(array, cell_voxels, **interp_kws)]

    return mesh


def assign_mesh_materials(
    mesh: 'meshio.Mesh',
    region_labels: np.ndarray,
    material_labels: np.ndarray,
    material_catalog: 'pd.DataFrame'
) -> 'meshio.Mesh':
    '''
    Assign region/material labels and properties to mesh nodes and cells.
    '''
    from . import material_properties

    cell_blocks = [block.type for block in mesh.cells]
    if cell_blocks != ['tetra']:
        raise ValueError(f'Expected one tetra cell block: {cell_blocks!r}')

    region_to_material = material_properties.infer_material_by_region(
        region_labels, material_labels
    )

    region_cells = mesh.cell_data_dict['region']['tetra'].astype(int)
    material_cells = region_to_material[region_cells]
    mesh.cell_data['material'] = [material_cells]

    E_cells, nu_cells, rho_cells = material_properties.assign_material_properties(
        material_cells, material_catalog
    )

    mesh.cell_data['E'] = [E_cells]
    mesh.cell_data['nu'] = [nu_cells]
    mesh.cell_data['rho'] = [rho_cells]

    verts = mesh.points
    cells = mesh.cells_dict['tetra']
    volume = transforms.compute_cell_volume(verts, cells)
    A = transforms.compute_incidence_matrix(verts, cells)

    mesh.point_data['material'] = transforms.cell_to_node_labels(
        verts, cells, material_cells
    )

    mesh.point_data['E'] = transforms.cell_to_node_values(E_cells, volume, A)
    mesh.point_data['nu'] = transforms.cell_to_node_values(nu_cells, volume, A)
    mesh.point_data['rho'] = transforms.cell_to_node_values(rho_cells, volume, A)

    return mesh

