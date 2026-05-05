from typing import Dict, Set, Any
import collections
import numpy as np
import meshio

from ..core import utils, transforms


def _get_pygalmesh_spacing(affine: np.ndarray, use_affine: bool):
    if use_affine:
        return transforms.get_affine_spacing(affine)
    return np.ones(3, dtype=np.float32)


def generate_mesh_from_mask(
    mask: np.ndarray,
    affine: np.ndarray,
    use_affine: bool = False,
    random_seed: int = 0,
    pygalmesh_kws: Dict[str, Any] = None,
    raw_label_key: str = 'medit:ref',
    new_label_key: str = 'region'
) -> meshio.Mesh:
    '''
    Generate a tetrahedral mesh from a voxel mask using pygalmesh.

    Args:
        mask: 3D input voxel mask (binary or region labels)
        affine: (4, 4) voxel to world coordinate transform
        use_affine: If True, use world spacing when generating the mesh,
            otherwise mesh in voxel coordinates. The returned mesh will
            be converted to world coordinates either way.
        random_seed: int random seed passed to pygalmesh
        pygalmesh_kws: mesh parameters passed to pyglamesh
    Returns:
        meshio.Mesh: Generated mesh in world coordinates
    '''
    spacing = _get_pygalmesh_spacing(affine, use_affine)

    utils.log('Running pygalmesh generation')
    raw_mesh = run_pygalmesh_generate(
        mask=mask,
        spacing=spacing,
        random_seed=random_seed,
        **(pygalmesh_kws or {})
    )

    utils.log('Extracting tetrahedral cells')
    mesh = extract_cell_type(raw_mesh, cell_type='tetra')

    utils.log('Inferring cell label map')
    label_map = infer_label_map(mesh, mask, spacing, raw_label_key)

    utils.log('Reindexing cell labels')
    mesh = reindex_cell_labels(mesh, label_map, raw_label_key, new_label_key)

    utils.log('Removing background cells')
    mesh = remove_labeled_cells(mesh, 'tetra', new_label_key, label_val=0)

    utils.log('Removing unreferenced points')
    mesh = remove_unreferenced_points(mesh)

    utils.log('Converting to world coordinates')
    mesh = convert_to_world_coords(mesh, spacing, affine)

    n_components = count_connected_components(mesh, cell_type='tetra')
    if n_components != 1:
        utils.warn(f'WARNING: mesh has {n_components} components')

    utils.log(mesh)
    return mesh


def run_pygalmesh_generate(
    mask: np.ndarray,
    spacing: np.ndarray,
    random_seed: int,
    **pygalmesh_kws
) -> meshio.Mesh:
    import pygalmesh

    if not np.issubdtype(mask.dtype, np.integer):
        utils.log('WARNING: mask is not an integer dtype')

    mesh = pygalmesh.generate_from_array(
        mask.astype(np.uint16),
        voxel_size=spacing,
        seed=random_seed,
        **pygalmesh_kws
    )
    utils.log(mesh)

    if len(mesh.points) == 0:
        raise RuntimeError('mesh has zero vertices')

    if count_cell_type(mesh, cell_type='tetra') == 0:
        raise RuntimeError('mesh has no tetra cells')

    return mesh


# ----- cell type filtering -----


def extract_cell_type(mesh: meshio.Mesh, cell_type: str) -> meshio.Mesh:

    cells = mesh.cells_dict.get(cell_type)
    if cells is None:
        raise RuntimeError(f'mesh has no {cell_type} cells')

    new_cell_data = {}
    for key, data_by_type in mesh.cell_data_dict.items():
        if cell_type in data_by_type:
            new_cell_data[key] = [data_by_type[cell_type]]

    return meshio.Mesh(
        points=mesh.points,
        cells=[(cell_type, cells)],
        point_data=mesh.point_data,
        cell_data=new_cell_data
    )


def count_cell_type(mesh: meshio.Mesh, cell_type: str) -> int:
    return len(mesh.cells_dict.get(cell_type, []))


# ----- cell label reindexing -----


def infer_label_map(
    mesh: meshio.Mesh,
    mask: np.ndarray,
    spacing: np.ndarray,
    label_key: str = 'medit:ref',
    cell_type: str = 'tetra'
) -> np.ndarray:
    '''
    Infer a mapping from generated mesh cell labels to
    mask labels by majority vote at the cell centroids.

    When pygalmesh generates a mesh from a region label map,
    it produces cell labels that do not directly map to the
    labels in the input mask.

    We infer the mapping from generated cell labels to the
    original mask labels by interpolating the mask at each
    cell centroid and voting on the label assignment.
    '''
    import scipy as sp

    raw_labels = mesh.cell_data_dict[label_key][cell_type]

    if not np.issubdtype(raw_labels.dtype, np.integer):
        raise RuntimeError('raw labels are not an integer dtype')
    if np.any(raw_labels < 0):
        raise RuntimeError('raw labels include negative value(s)')

    # interpolate mask at cell centers
    cells = mesh.cells_dict[cell_type]
    centroids = mesh.points[cells].mean(axis=1)
    vox_coords = centroids / spacing

    mask_values = sp.ndimage.map_coordinates(
        mask.astype(int),
        coordinates=vox_coords.T,
        order=0,
        mode='nearest'
    )

    # vote on the mapping from raw labels to mask labels
    label_map = -np.ones(raw_labels.max() + 1, dtype=int)

    for label in np.unique(raw_labels):
        counts = np.bincount(mask_values[raw_labels == label])
        label_map[label] = int(counts.argmax())

    utils.log(label_map)
    if np.any(label_map[np.unique(raw_labels)] < 0):
        raise RuntimeError(f'raw label(s) were dropped from mapping')

    return label_map


def reindex_cell_labels(
    mesh: meshio.Mesh,
    label_map: np.ndarray,
    old_key: str = 'medit:ref',
    new_key: str = 'label'
) -> meshio.Mesh:

    new_cell_data = {} # copy existing data
    for key, data_list in mesh.cell_data.items():
        new_cell_data[key] = [array.copy() for array in data_list]

    new_cell_data[new_key] = [
        label_map[np.asarray(array, dtype=int)]
            for array in mesh.cell_data[old_key]
    ]

    return meshio.Mesh(
        points=mesh.points,
        cells=mesh.cells,
        point_data=mesh.point_data,
        cell_data=new_cell_data
    )


# ----- cell label filtering -----


def remove_labeled_cells(
    mesh: meshio.Mesh,
    cell_type: str,
    label_key: str,
    label_val: int = 0
) -> meshio.Mesh:

    labels = mesh.cell_data_dict[label_key][cell_type]
    to_remove = (labels == label_val)

    if np.all(to_remove):
        raise RuntimeError(f'all cells have label {label_val}')

    if np.any(to_remove):
        mesh = extract_selected_cells(mesh, cell_type, ~to_remove)

    if count_labeled_cells(mesh, cell_type, label_key, label_val) > 0:
        raise RuntimeError('failed to remove cells')

    return mesh


def extract_selected_cells(
    mesh: meshio.Mesh,
    cell_type: str,
    sel: np.ndarray
) -> meshio.Mesh:

    cells = mesh.cells_dict.get(cell_type)
    if cells is None:
        raise RuntimeError(f'mesh has no {cell_type} cells')

    if not np.any(sel):
        raise RuntimeError('no cells were selected')
    sel_cells = cells[sel]

    new_cell_data = {}
    for key, data_by_type in mesh.cell_data_dict.items():
        if cell_type in data_by_type:
            new_cell_data[key] = [data_by_type[cell_type][sel]]

    return meshio.Mesh(
        points=mesh.points,
        cells=[(cell_type, sel_cells)],
        point_data=mesh.point_data,
        cell_data=new_cell_data
    )


def count_labeled_cells(
    mesh: meshio.Mesh,
    cell_type: str,
    label_key: str,
    label_val: int
) -> int:
    values = mesh.cell_data_dict[label_key][cell_type]
    return np.sum(values == label_val, dtype=int)


# ----- vertex filtering -----


def remove_unreferenced_points(mesh: meshio.Mesh) -> meshio.Mesh:
    '''
    Return a new mesh with no unreferenced points.
    '''
    point_inds = get_referenced_point_indices(mesh)
    mesh = extract_selected_points(mesh, point_inds)

    if count_unreferenced_points(mesh) > 0:
        raise RuntimeError('failed to remove points')

    return mesh


def get_referenced_point_indices(mesh: meshio.Mesh) -> np.ndarray:
    '''
    Return unique indices of points referenced by any cell.
    '''
    point_inds = np.concatenate([cb.data.ravel() for cb in mesh.cells])
    return np.unique(point_inds)


def extract_selected_points(
    mesh: meshio.Mesh, point_inds: np.ndarray
) -> meshio.Mesh:
    '''
    Return a new mesh with only the selected points,
    reindexing its cells and cell data accordingly.
    '''
    # filter points and point data
    new_points = mesh.points[point_inds]
    new_point_data = {}
    for k, v in mesh.point_data.items():
        if len(v) != len(mesh.points):
            raise RuntimeError(f'length mismatch: {len(v)} vs. {len(mesh.points)}')
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
            utils.warn(f'WARNING: no {block.type} cells after filtering points')
        reindexed = reindexed[valid]
        new_cells.append(meshio.CellBlock(block.type, reindexed))

        for key, data_list in mesh.cell_data.items():
            vals = np.asarray(data_list[i])
            if len(vals) != len(block.data):
                raise RuntimeError(f'length mismatch: {len(vals)} vs. {len(block.data)}')
            new_cell_data[key].append(vals[valid])

    return meshio.Mesh(
        points=new_points,
        cells=new_cells,
        point_data=new_point_data,
        cell_data=new_cell_data,
    )


def count_unreferenced_points(mesh: meshio.Mesh) -> int:
    point_used = check_referenced_points(mesh)
    return int((~point_used).sum())


def check_referenced_points(mesh: meshio.Mesh) -> np.ndarray:
    '''
    Return boolean mask indicating which points are referenced.
    '''
    point_used = np.zeros(len(mesh.points), dtype=bool)
    point_inds = get_referenced_point_indices(mesh)
    point_used[point_inds] = True
    return point_used


# ----- geometric transformations -----


def convert_to_world_coords(
    mesh: meshio.Mesh, spacing: np.ndarray, affine: np.ndarray
) -> meshio.Mesh:
    vox_points = mesh.points / spacing
    new_points = transforms.voxel_to_world_coords(vox_points, affine)
    return meshio.Mesh(
        points=new_points,
        cells=mesh.cells,
        point_data=mesh.point_data,
        cell_data=mesh.cell_data
    )


# ----- connected mesh components -----


def count_connected_components(mesh: meshio.Mesh, cell_type='tetra') -> int:
    '''
    Count the number of connected mesh components.
    '''
    cells = mesh.cells_dict.get(cell_type)
    if cells is None:
        raise RuntimeError(f'mesh has no {cell_type} cells')

    # neighbors[cell_A] = {cell_B | cells A and B share a face}
    neighbors = build_cell_adjacency(mesh, cell_type)

    # traverse cell graph via shared faces
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
            queue.extend(neighbors[current])

    return num_components


def build_cell_adjacency(mesh: meshio.Mesh, cell_type='tetra') -> Dict[int, Set]:
    '''
    Build cell adjacency list for the provided mesh.
    '''
    cells = mesh.cells_dict.get(cell_type)
    if cells is None:
        raise RuntimeError(f'mesh has no {cell_type} cells')

    if cell_type == 'tetra':
        face_inds = [[0,1,2], [0,1,3], [0,2,3], [1,2,3]]
    elif cell_type == 'triangle':
        face_inds = [[0,1], [1,2], [2,0]]
    else:
        raise ValueError(f'Invalid cell type: {cell_type!r}')

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

