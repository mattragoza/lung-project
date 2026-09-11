# preprocessing/triangular_meshing.py

from typing import Dict, Tuple, Any

import numpy as np
import trimesh

from ...common import utils, transforms


def _as_meshio(mesh: trimesh.Trimesh):
    import meshio
    return meshio.Mesh(points=mesh.vertices, cells=[('triangle', mesh.faces)])


def repair_triangular_mesh(
    mesh: trimesh.Trimesh,
    use_pymeshfix: bool = False,
    ret_meshio: bool = False
) -> trimesh.Trimesh:
    '''
    Process a triangular mesh to fix several issues
    and attempt to make it a watertight surface.
    '''
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

    if use_pymeshfix:
        import pymeshfix
        mesh_fixer = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
        mesh_fixer.repair(verbose=utils.VERBOSE)
        mesh = trimesh.Trimesh(vertices=mesh_fixer.v, faces=mesh_fixer.f, process=False)

    trimesh.repair.fix_normals(mesh)

    utils.log('\nAfter pymeshfix repair:')
    utils.log(utils.pprint(get_mesh_info(mesh), ret_string=True))

    return _as_meshio(mesh) if ret_meshio else mesh


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



