from collections import defaultdict, deque
from itertools import combinations, permutations
import numpy as np
import scipy as sp
import meshio
import pygalmesh
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import fenics as fe
from mpi4py import MPI
import dolfin


def generate_mesh_from_mask(mask, resolution, **kwargs):

    mesh = pygalmesh.generate_from_array(
        mask.astype(np.uint16),
        voxel_size=resolution,
        verbose=True,
        **kwargs
    )
    mesh = assign_cell_labels(mesh, mask, resolution)
    assert count_labeled_cells(mesh, 'tetra', 'label', 0) == 0

    num_before = count_unused_points(mesh)
    mesh = remove_unused_points(mesh)

    num_after = count_unused_points(mesh)
    print(f'Removed {num_before - num_after} unused points')
    assert num_after == 0

    print('Final mesh: ', mesh)

    num_components = count_connected_components(mesh)
    assert num_components > 0
    print(f'Mesh has {num_components} components')

    return split_cells_by_type(mesh)


def assign_cell_labels(mesh, mask, resolution, new_key='label', old_key='medit:ref'):
    label_map = construct_label_map(mesh, mask, resolution, old_key)
    new_cell_data = {
        new_key: [label_map[l] for l in mesh.cell_data[old_key]]
    }
    return meshio.Mesh(
        points=mesh.points,
        cells=mesh.cells,
        cell_data=new_cell_data
    )


def construct_label_map(mesh, mask, resolution, label_key='medit:ref'):

    tetrahedra = mesh.cells_dict['tetra']
    barycenters = mesh.points[tetrahedra].mean(axis=1)
    mask_coords = barycenters / resolution
    mask_values = sp.ndimage.map_coordinates(
        mask, mask_coords.T, order=0, mode='nearest'
    ).astype(int)

    mesh_labels = mesh.cell_data_dict[label_key]['tetra']
    label_map = -np.ones(mesh_labels.max() + 1, dtype=int)
    for mesh_label in np.unique(mesh_labels):
        values = mask_values[mesh_labels == mesh_label]
        most_common = np.bincount(values).argmax()
        label_map[mesh_label] = most_common

    return label_map


def remove_unused_points(mesh):

    # get indices of points used in cells
    used_point_indices = get_used_point_indices(mesh)

    # filter unused points
    new_points = mesh.points[used_point_indices]

    # filter point data, if present
    new_point_data = {}
    for key, value in mesh.point_data.items():
        new_point_data[key] = value[used_point_indices]

    # build mapping from old to new point indices
    index_map = -np.ones(mesh.points.shape[0], dtype=int)
    index_map[used_point_indices] = np.arange(len(used_point_indices))
    
    # reindex points in cell blocks
    new_cells = []
    for block in mesh.cells:
        new_data = index_map[block.data]
        new_block = meshio.CellBlock(block.type, new_data)
        new_cells.append(new_block)

    return meshio.Mesh(
        points=new_points,
        cells=new_cells,
        cell_data=mesh.cell_data
    )


def get_used_point_indices(mesh):
    return np.unique(
        np.concatenate([block.data.ravel() for block in mesh.cells])
    )


def check_used_points(mesh):
    point_is_used = np.zeros(mesh.points.shape[0], dtype=bool)
    used_point_indices = get_used_point_indices(mesh)
    point_is_used[used_point_indices] = True
    return point_is_used


def count_unused_points(mesh):
    point_is_used = check_used_points(mesh)
    return (~point_is_used).sum()


def count_labeled_cells(mesh, cell_type, label_key, label_value):
    cell_labels = mesh.cell_data_dict[label_key][cell_type]
    cell_count = np.sum(cell_labels == label_value)
    return cell_count


def split_points_by_label(mesh, label_key='medit:ref'):
    labels = mesh.point_data[label_key]
    split = {}
    for label in np.unique(labels):
        mask = (labels == label)
        sub_points = mesh.points[mask]
        split[label] = meshio.Mesh(
            points=sub_points,
            cells=[], # no cells, just point cloud
        )
    return split


def split_cells_by_label(mesh, label_key='medit:ref', cell_type='tetra'):
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


def split_cells_by_type(mesh):
    split = {}
    for block in mesh.cells:
        block_data = {k: [mesh.cell_data_dict[k][block.type]] for k in mesh.cell_data}
        split[block.type] = meshio.Mesh(
            points=mesh.points,
            cells=[block],
            cell_data=block_data
        )
    return split


def count_connected_components(mesh, cell_type='tetra'):
    cells = mesh.cells_dict[cell_type]

    if cell_type == 'tetra':
        face_indices = [[0,1,2], [0,1,3], [0,2,3], [1,2,3]]
    elif cell_type == 'triangle':
        face_indices = [[0,1], [1,2], [2,0]]
    else:
        raise ValueError(f'Unrecognized cell type: {cell_type}')

    # build mapping from faces to incident cells
    face_to_cells = defaultdict(list)
    for cell_id, cell in enumerate(cells):
        for inds in face_indices:
            face = tuple(sorted(cell[inds]))
            face_to_cells[face].append(cell_id)

    # build adjacency list
    adjacency = defaultdict(set)
    for face, incident_cells in face_to_cells.items():
        if len(incident_cells) == 2:
            a, b = incident_cells
            adjacency[a].add(b)
            adjacency[b].add(a)

    # traverse graph
    visited = np.zeros(len(cells), dtype=bool)
    n_components = 0

    for start in range(len(cells)):
        if visited[start]:
            continue
        n_components += 1
        queue = deque([start])
        while queue:
            current = queue.pop()
            if visited[current]:
                continue
            visited[current] = True
            queue.extend(adjacency[current])

    return n_components


def apply_affine_to_mesh(mesh, resolution, affine):
    # pygalmesh uses voxel spacing to generate the mesh,
    # but not the full affine. we need to use the spacing
    # and affine to convert the mesh to world coordinates.
    resolution = np.asarray(resolution)
    R = affine[:3,:3] @ np.diag(1.0 / resolution)
    t = affine[:3, 3]
    new_points = mesh.points @ R.T + t
    return meshio.Mesh(
        points=new_points,
        cells=mesh.cells,
        cell_data=mesh.cell_data,
        point_data=mesh.point_data
    )


## fenics mesh functions


def load_mesh_with_fenics(mesh_file, label_key='label', verbose=False):
    if verbose:
        print(f'Loading {mesh_file}...', end=' ')
    mesh = fe.Mesh()
    with fe.XDMFFile(MPI.COMM_WORLD, str(mesh_file)) as f:
        f.read(mesh)
        dim = mesh.geometry().dim()
        if label_key:
            cell_labels = dolfin.MeshFunction('size_t', mesh, dim)
            f.read(cell_labels, label_key)
        else:
            cell_labels = None
    if verbose:
        print(mesh.num_vertices())
    return mesh, cell_labels


def check_disconnected_nodes(mesh):
    tdim = mesh.topology().dim()
    mesh.init(0, tdim)
    vertex_to_cell = mesh.topology()(0, tdim)
    unconnected_nodes = []
    num_vertices = mesh.num_vertices()
    for v in range(num_vertices):
        if len(vertex_to_cell(v)) == 0:
            unconnected_nodes.append(v)
    print(len(unconnected_nodes), num_vertices)
    return unconnected_nodes

