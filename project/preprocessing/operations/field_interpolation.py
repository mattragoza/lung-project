# preprocessing/field_interpolation.py

from typing import Dict

import numpy as np

from ..common import transforms


def interpolate_mesh_fields(
	mesh: 'meshio.Mesh',
	fields: Dict[str, np.ndarray],
	affine: np.ndarray,
	**kwargs
):
	from ..common.interpolation import interpolate_array

    if len(mesh.cells) != 1 or mesh.cells[0].type != 'tetra':
        block_types = [block.type for block in mesh.cells]
        raise ValueError(f'Expected exactly one tetra cell block: {block_types}')

	node_voxels = transforms.world_to_voxel_coords(mesh.points, affine)

	cells = mesh.cells_dict['tetra']
	cell_voxels = node_voxels[cells].mean(axis=1)

	for name, array in fields.items():
		mesh.point_data[name] = interpolate_array(array, node_voxels, **kwargs)
		mesh.cell_data[data] = [interpolate_array(array, cell_voxels, **kwargs)]

	return mesh

