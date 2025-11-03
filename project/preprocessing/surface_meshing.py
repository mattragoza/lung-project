from typing import Dict, List, Tuple, Iterable, Optional, Any
import numpy as np
import trimesh

from ..core import utils, transforms


def repair_surface_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    '''
    Process a triangular mesh to fix several issues
    and attempt to make it a watertight surface.
    '''
    import pymeshfix
    mesh = mesh.copy()

    utils.log('Initial mesh state:')
    utils.log(utils.pprint(get_mesh_info(mesh), ret_string=True))

    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices(digits_vertex=8, merge_norm=True, merge_tex=True)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    trimesh.repair.fill_holes(mesh)

    utils.log('\nAfter trimesh repair:')
    utils.log(utils.pprint(get_mesh_info(mesh), ret_string=True))

    mesh_fixer = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
    mesh_fixer.repair(verbose=utils.VERBOSE)
    mesh = trimesh.Trimesh(vertices=mesh_fixer.v, faces=mesh_fixer.f, process=False)
    trimesh.repair.fix_normals(mesh)

    utils.log('\nAfter pymeshfix repair:')
    utils.log(utils.pprint(get_mesh_info(mesh), ret_string=True))

    return mesh


def get_mesh_info(mesh: trimesh.Trimesh) -> Dict[str, Any]:
    vertices, faces = mesh.vertices, mesh.faces
    num_components = len(mesh.split(only_watertight=False))
    angles = np.degrees(trimesh.triangles.angles(mesh.triangles))
    return dict(
        vertices=len(vertices),
        faces=len(faces),
        edges=count_edge_types(faces),
        euler_number=mesh.euler_number,
        watertight=mesh.is_watertight,
        components=num_components,
        angles=dict(
            p05=float(np.percentile(angles, 5)),
            p50=float(np.percentile(angles, 50)),
            p95=float(np.percentile(angles, 95)),
        ),

    )


def count_edge_types(faces: np.ndarray) -> Dict[str, int]:
    u, c = count_unique_edges(faces)
    n1 = int((c == 1).sum())
    n2 = int((c == 2).sum())
    nm = int((c >= 3).sum())
    return dict(boundary=n1, interior=n2, nonmanifold=nm)


def count_unique_edges(faces: np.ndarray) -> np.ndarray:
    f = faces.astype(np.int16, copy=False)
    e = np.vstack([f[:,[0,1]], f[:,[1,2]], f[:,[2,0]]])
    e.sort(axis=1)
    return np.unique(e, axis=0, return_counts=True)


# --- mesh region labels ---


def extract_face_labels(scene: trimesh.Scene):
    '''
    Args:
        scene: trimesh.Scene with multiple geometries
    Returns:
        mesh: trimesh.Trimesh from merging geometries
        labels: array of face labels indicating the
            source geometry of each face in the scene
    '''
    verts, faces, labels = [], [], []
    offset = 0

    for idx, name in enumerate(scene.graph.nodes_geometry):
        geom = scene.geometry[name]
    
        T = scene.graph[name][0]
        if not np.allclose(T, np.eye(4)):
            utils.warn(f'WARNING: Non-identity transform on {name}:\n{T}')
    
        v = trimesh.transform_points(geom.vertices, T)
        f = geom.faces

        verts.append(v.copy())
        faces.append(f.copy() + offset)
        labels.append(np.full(len(f), idx, dtype=int))
        offset += len(v)

    if not verts:
        raise ValueError('scene has no geometries')

    verts  = np.concatenate(verts, axis=0)
    faces  = np.concatenate(faces, axis=0)
    labels = np.concatenate(labels, axis=0)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    return mesh, labels


def query_face_labels(
    mesh: trimesh.Trimesh,
    labels: np.ndarray,
    points: np.ndarray,
    chunk_size=10000
) -> np.ndarray:
    '''
    Map face labels to arbitrary points by computing
    the nearest face in the mesh to each query point.

    Args:
        mesh: trimesh.Trimesh object
        labels: (N,) array of face labels
        points: (M,3) array of query points,
            in same coordinate system as mesh
    Returns:
        values: (M,) array of label values
    '''
    import sys, tqdm
    assert points.ndim == 2 and points.shape[1] == 3

    query = trimesh.proximity.ProximityQuery(mesh)
    points = np.asarray(points, dtype=np.float32)
    output = np.empty(len(points), dtype=int)

    def _most_common_value(arr):
        values, counts = np.unique(arr, return_counts=True)
        return values[np.argmax(counts)]
    
    for start in tqdm.tqdm(range(0, len(points), chunk_size), file=sys.stdout):
        end = min(start + chunk_size, len(points))
        #_, _, face_inds = query.on_surface(points[start:end])
        candidates = trimesh.proximity.nearby_faces(mesh, points[start:end])
        face_inds = np.array([_most_common_value(c) for c in candidates], dtype=int)
        output[start:end] = labels[face_inds]
    
    return output


def assign_voxel_labels(
    mask: np.ndarray,
    affine: np.ndarray,
    mesh: trimesh.Trimesh,
    labels: np.ndarray
) -> np.ndarray:
    '''
    Assign labels to nonzero voxels by computing the
    nearest face in the mesh to each voxel and mapping
    its face label to a voxel label.

    Note that voxel labels are shifted so that only the
    background is 0 in the output mask.

    Args:
        mask: (I, J, K) binary voxel mask
        affine: (4, 4) voxel -> model coordinate map
        mesh: trimesh.Trimesh in model coordinates
        labels: (N,) array of face labels
    Returns:
        labeled: (I, J, K) labeled voxel mask
    Returns:
    '''
    I, J, K = np.nonzero(mask)
    points_voxel = np.c_[I, J, K]
    points_model = transforms.voxel_to_world_coords(points_voxel, affine)

    labeled = np.zeros_like(mask, dtype=np.int16) # reserve background = 0
    labeled[I, J, K] = query_face_labels(mesh, labels, points_model) + 1
    return labeled

