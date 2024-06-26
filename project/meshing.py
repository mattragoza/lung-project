from collections import defaultdict
from itertools import permutations
import numpy as np
import meshio
import pygalmesh
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def estimate_limit(x, expand=0.1):
    x_min, x_max = np.min(x), np.max(x)
    x_range = (x_max - x_min)
    x_min -= expand * x_range / 2
    x_max += expand * x_range / 2
    return x_min, x_max


def plot_mesh(vertices, facets, figsize=(6,6), **kwargs):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d', aspect='equal')
    polys = Poly3DCollection(vertices[facets], **kwargs)
    ax.add_collection(polys)
    ax.set_xlim(estimate_limit(vertices.flatten()))
    ax.set_ylim(estimate_limit(vertices.flatten()))
    ax.set_zlim(estimate_limit(vertices.flatten()))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    return fig, ax


def compute_angles_to_xy_plane(vertices, facets):
    edges1 = vertices[facets[:,1]] - vertices[facets[:,0]]
    edges2 = vertices[facets[:,2]] - vertices[facets[:,1]]
    facet_normals = np.cross(edges1, edges2)
    norms = np.linalg.norm(facet_normals, axis=1)
    xy_normal = np.array([0, 0, 1])
    angle_cos = (facet_normals @ xy_normal) / norms
    return angle_cos


def compute_angles_to_interior(vertices, facets, tetras):
    edges1 = vertices[facets[:,1]] - vertices[facets[:,0]]
    edges2 = vertices[facets[:,2]] - vertices[facets[:,1]]
    facet_normals = np.cross(edges1, edges2)
    facet_norms = np.linalg.norm(facet_normals, axis=1)
    
    facet_centroids = vertices[facets].mean(axis=1)
    tetra_centroids = vertices[tetras].mean(axis=1)
    facet_tetra_dists = (
    	(facet_centroids[:,None,:] - tetra_centroids[None,:,:])**2
    ).sum(axis=2)
    nearest_tetras = np.argmin(facet_tetra_dists, axis=1)
    nearest_tetra_centroids = tetra_centroids[nearest_tetras]

    facet_offsets = facet_centroids - nearest_tetra_centroids
    offset_norms = np.linalg.norm(facet_offsets, axis=1)
    
    angle_cos = (facet_normals * facet_offsets).sum(axis=1) / (facet_norms * offset_norms)
    return angle_cos


def smooth_facet_values(vertices, facets, facet_values, func, order=1):
    '''
    Args:
        vertices: (N, 3) array of 3D vertex coordinates
        facets: (M, 3) array of indices of triangle vertices
        facet_values: (M, D) array of values assigned to facets
        func: Reduces sets of facet values to a single value
        order: Order of structure elements to define neighborhoods
            For order=1, use common vertices
            For order=2, use common edges
    Returns:
        new_facet_values: (M, D) array of values from applying
            func to each facet's local neighborhood of values
    '''
    # mapping from elements to facets that share that element
    common_elements = defaultdict(set)
    for i, facet_vertices in enumerate(facets):
        for element in permutations(facet_vertices, order):
            common_elements[element].add(i)

    # unordered sets of vertices for each facet
    vertex_sets = [set(v) for v in facets]
   
    new_facet_values = np.zeros_like(facet_values)
    for i, facet_vertices in enumerate(facets):
        neighbor_values = []
        for element in permutations(facet_vertices, order):
            for j in common_elements[element]:
                if vertex_sets[j] != vertex_sets[i]:
                    neighbor_values.append(facet_values[j])
        new_facet_values[i] = func(neighbor_values)
            
    return new_facet_values


def check_used_points(mesh):
    point_inds = np.arange(mesh.points.shape[0])
    point_used = np.zeros_like(point_inds, dtype=bool)
    for cells in mesh.cells:
        used_in_cells = np.unique(cells.data)
        point_used |= np.isin(point_inds, used_in_cells)
        
    return point_used


def remove_unused_points(mesh):

    points = mesh.points
    tri_cells = mesh.cells[0].data
    tet_cells = mesh.cells[1].data
    
    # identify the used points
    points_used = np.unique(np.concatenate([
        tri_cells.flatten(), tet_cells.flatten()
    ]))
    
    # mapping from old indices to new indices
    new_indices = np.zeros(points.shape[0], dtype=int)
    new_indices[points_used] = np.arange(len(points_used))
    
    # filter points to keep only the used points
    new_points = points[points_used]
    
    # update the cell indices
    new_tri_cells = new_indices[tri_cells]
    new_tet_cells = new_indices[tet_cells]
    
    return meshio.Mesh(
        points=new_points,
        cells=[('triangle', new_tri_cells), ('tetra', new_tet_cells)]
    )
