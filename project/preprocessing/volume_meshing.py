from typing import Dict, Set, Any
import numpy as np
import scipy as sp
import meshio

from ..core import utils, transforms


def _affine_linear(A):
    return A[:3,:3]


def _affine_origin(A):
    return A[:3,3]


def _affine_spacing(A):
    return np.linalg.norm(_affine_linear(A), axis=0)


def generate_mesh_from_mask(
    mask: np.ndarray,
    affine: np.ndarray,
    use_affine_spacing: bool=True,
    pygalmesh_kws: Dict[str, Any]=None,
    random_seed: int=0
) -> meshio.Mesh:
    '''
    Generate a tetrahedral mesh from a voxel mask
    using pygalmesh and apply post-processing.

    Args:
        mask: (I, J, K) voxel mask (binary or region)
        affine: (4, 4) voxel to world coordinate map
        use_world_spacing: If True, use affine spacing
            when generating the mesh, otherwise mesh in
            voxel coordinates. The returned mesh will be
            in the world coordinate system either way.
        pygalmesh_kws: Meshing params passed to pyglamesh
        random_seed: int
    Returns:
        meshio.Mesh, in world coordinates
    '''
    import pygalmesh

    spacing = _affine_spacing(affine) if use_affine_spacing else np.ones(3)

    mesh = pygalmesh.generate_from_array(
        mask.astype(np.uint16),
        voxel_size=spacing,
        seed=random_seed,
        **(pygalmesh_kws or {})
    )

    # clean mesh and correct cell labels
    mesh = remove_unreferenced_points(mesh)
    mesh = assign_cell_labels(mesh, mask, new_key='label', old_key='medit:ref')

    # pygalmesh uses voxel spacing to generate the mesh,
    # not the full affine. To convert to world coords, we
    # normalize by voxel spacing, then apply the affine.

    mesh.points = transforms.voxel_to_world_coords(mesh.points / spacing, affine)

    num_components = count_connected_components(mesh, cell_type='tetra')
    utils.log(f'Mesh has {num_components} connected component(s)')
    if num_components == 0:
        utils.warn('WARNING: mesh has no connected components')

    utils.log(mesh)

    # return only the tetra cells, not triangles
    return split_mesh_by_cell_type(mesh)['tetra']


# --- cell region labels ---


def assign_cell_labels(
    mesh: meshio.Mesh,
    mask: np.ndarray,
    new_key: str='label',
    old_key: str='medit:ref'
) -> meshio.Mesh:
    '''
    Update cell labels based on mask values.
    '''
    # when providing a multi-label mask to pygalmesh,
    # the generated mesh contains region boundaries
    # and labels in an auto-generated cell_data key.

    # here we construct a deterministic map from the
    # auto-generated labels to the voxel mask labels and
    # then update the cell labels using the label map.

    label_map = construct_label_map(mesh, mask, old_key)

    new_cell_data = {new_key: [
        label_map[a] for a in mesh.cell_data[old_key]
    ]}
    mesh = meshio.Mesh(
        points=mesh.points,
        cells=mesh.cells,
        cell_data=new_cell_data,
        point_data=mesh.point_data,
    )

    # sanity check - no cells should be labeled as mask background
    assert count_labeled_cells(mesh, 'tetra', new_key, value=0) == 0

    return mesh


def construct_label_map(
    mesh: meshio.Mesh,
    mask: np.ndarray,
    key: str='medit:ref'
) -> np.ndarray:
    '''
    Build a mapping from existing mesh cell labels to
    mask labels by majority vote at the cell centroids.
    '''
    # interpolate mask values at tetra centers
    points = _centroids(mesh, cell_type='tetra')
    mask_values = sp.ndimage.map_coordinates(
        mask.astype(int),
        points.T,
        order=0,
        mode='nearest'
    )
    # get most common mask label for each mesh label
    old_labels = mesh.cell_data_dict[key]['tetra']
    label_map = -np.ones(old_labels.max() + 1, dtype=int)

    for l in np.unique(old_labels):
        values = mask_values[old_labels == l]
        most_common = np.bincount(values).argmax()
        label_map[l] = int(most_common)

    return label_map


def count_labeled_cells(mesh, cell_type, key, value):
    labels = mesh.cell_data_dict[key][cell_type]
    return np.sum(labels == value)


def _centroids(mesh, cell_type):
    cells = mesh.cells_dict[cell_type]
    return mesh.points[cells].mean(axis=1)


# --- unreferenced points ---


def get_referenced_point_indices(mesh: meshio.Mesh) -> np.ndarray:
    '''
    Return unique indices of points referenced by any cell.
    '''
    point_inds = np.concatenate([b.data.ravel() for b in mesh.cells])
    return np.unique(point_inds)


def check_referenced_points(mesh: meshio.Mesh) -> np.ndarray:
    '''
    Return boolean mask indicating which points are referenced.
    '''
    point_used = np.zeros(len(mesh.points), dtype=bool)
    point_inds = get_referenced_point_indices(mesh)
    point_used[point_inds] = True
    return point_used


def count_unreferenced_points(mesh: meshio.Mesh) -> int:
    point_used = check_referenced_points(mesh)
    return int((~point_used).sum())


def remove_unreferenced_points(mesh: meshio.Mesh) -> meshio.Mesh:
    '''
    Return a new mesh with no unreferenced points.
    '''
    c1 = count_unreferenced_points(mesh)

    point_inds = get_referenced_point_indices(mesh)
    mesh = filter_mesh_points(mesh, point_inds)

    c2 = count_unreferenced_points(mesh)
    utils.log(f'Removed {c1 - c2} unreferenced point(s)')

    assert c2 == 0, (c1, c2) # sanity check

    return mesh


def filter_mesh_points(mesh: meshio.Mesh, point_inds: np.ndarray) -> meshio.Mesh:
    '''
    Return a new mesh containing only selected points,
    reindexing its cells and cell data accordingly.
    '''
    # filter points and point data
    new_points = mesh.points[point_inds]
    new_point_data = {}
    for k, v in mesh.point_data.items():
        assert v.shape[0] == len(mesh.points)
        new_point_data[k] = v[point_inds]
    
    # build mapping from old to new point indices
    index_map = -np.ones(mesh.points.shape[0], dtype=int)
    index_map[point_inds] = np.arange(len(new_points), dtype=int)

    # reindex cells and cell data
    new_cells = []
    new_cell_data = {k: [] for k in mesh.cell_data.keys()}

    for i, block in enumerate(mesh.cells):
        reindexed = index_map[block.data]
        valid = (reindexed >= 0).all(axis=1)
        if not np.any(valid):
            utils.warn(f'WARNING: no valid {block.type} cells after filtering points')
        reindexed = reindexed[valid]
        new_cells.append(meshio.CellBlock(block.type, reindexed))

        for k, v_list in mesh.cell_data.items():
            vals = np.asarray(v_list[i])
            assert vals.shape[0] == block.data.shape[0]
            new_cell_data[k].append(vals[valid])

    return meshio.Mesh(
        points=new_points,
        cells=new_cells,
        point_data=new_point_data,
        cell_data=new_cell_data,
    )


# --- connected components ---


def count_connected_components(mesh: meshio.Mesh, cell_type='tetra') -> int:
    '''
    Count the number of connected mesh components.
    '''
    import collections
    cells = mesh.cells_dict[cell_type]

    # build cell adjacency list
    adjacent = build_cell_adjacency(mesh, cell_type)

    # traverse cell graph
    visited = np.zeros(len(cells), dtype=bool)
    num_components = 0

    for start in range(len(cells)):
        if visited[start]:
            continue
        num_components += 1
        queue = collections.deque([start])
        while queue:
            current = queue.pop()
            if visited[current]:
                continue
            visited[current] = True
            queue.extend(adjacent[current])

    return num_components


def build_cell_adjacency(mesh: meshio.Mesh, cell_type='tetra') -> Dict[int, Set]:
    '''
    Build cell adjacency list for the provided mesh.
    '''
    import collections
    cells = mesh.cells_dict[cell_type]

    if cell_type == 'tetra':
        face_inds = [[0,1,2], [0,1,3], [0,2,3], [1,2,3]]
    elif cell_type == 'triangle':
        face_inds = [[0,1], [1,2], [2,0]]
    else:
        raise ValueError(f'Unrecognized cell type: {cell_type}')

    # build mapping from faces to incident cells
    face_to_cells = collections.defaultdict(list)
    for cell_id, cell in enumerate(cells):
        for inds in face_inds:
            face = tuple(sorted(cell[inds]))
            face_to_cells[face].append(cell_id)

    # build adjacency list
    adjacent = collections.defaultdict(set)
    for face, incident_cells in face_to_cells.items():
        if len(incident_cells) == 2:
            a, b = incident_cells
            adjacent[a].add(b)
            adjacent[b].add(a)

    return adjacent


# --- mesh splitting ---


def split_mesh_by_cell_type(mesh: meshio.Mesh) -> Dict[str, meshio.Mesh]:
    split = {}
    for block in mesh.cells:
        block_data = {k: [mesh.cell_data_dict[k][block.type]] for k in mesh.cell_data}
        split[block.type] = meshio.Mesh(
            points=mesh.points,
            cells=[block],
            cell_data=block_data
        )
    return split


def split_mesh_by_cell_label(
    mesh: meshio.Mesh,
    label_key='medit:ref',
    cell_type='tetra'
) -> Dict[str, meshio.Mesh]:

    block_index = next(
        i for i, b in enumerate(mesh.cells) if b.type == cell_type
    )
    labels = mesh.cell_data[label_key][block_index]
    split = {}
    for label in np.unique(labels):
        mask = (labels == label)
        sub_cells = mesh.cells[block_index].data[mask]
        split[label] = meshio.Mesh(
            points=mesh.points,
            cells=[meshio.CellBlock(cell_type, sub_cells)]
        )
    return split

