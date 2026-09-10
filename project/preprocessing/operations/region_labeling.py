# preprocessing/region_labeling.py

from typing import List, Dict, Any

import numpy as np

from ..common import utils, transforms


def label_anatomical_regions(
    roi_masks: Dict[str, np.ndarray],
    roi_order: List[str],
    filter_kws: Dict[str, Any]
) -> np.ndarray:
    from . import mask_processing

    label_masks = []
    for label, name in enumerate(roi_order, start=1): # reserve 0 for background
        label_masks.append(roi_masks[name] * label)

    raw_labels = np.max(label_arrays, axis=0) # use roi order for priority
    out_labels = np.zeros_like(raw_labels)

    for label, name in enumerate(roi_order, start=1):
        utils.log(f'Filtering region: {name}')

        if 'max_components' not in filter_kws:
            filter_kws['max_components'] = (1 if 'lobe' in name.lower() else None)

        filtered = mask_processing.filter_connected_components(
            (raw_labels == label), **filter_kws
        )
        out_labels[filtered] = label

    # reassign dropped voxels to nearest region
    dropped = (raw_map != 0) & (out_map == 0)

    if np.any(dropped):
        from scipy.ndimage import distance_transform_edt

        _, indices = distance_transform_edt(
            (out_labels == 0), return_indices=True
        )
        nearest_labels = out_labels[tuple(indices)]
        out_labels[dropped] = nearest_labels[dropped]

    return out_labels.astype(np.int16)


def label_regions_from_surface(
    mask: np.ndarray,
    affine: np.ndarray, 
    scene: 'trimesh.Scene',
    method: str,
    filter_kws: dict
) -> np.ndarray:
    from . import mask_processing

    utils.log('Extracting labels from mesh')
    mesh, labels = extract_face_labels(scene)

    utils.log('Assigning labels to voxels')
    labels = assign_voxel_labels(mask, affine, mesh, labels, method)

    utils.log('Cleaning up region mask')
    labels = mask_processing.filter_region_mask(labels, **filter_kws)

    if not len(np.unique(labels[labels > 0])) > 1:
        raise RuntimeError('Single region label')

    return labels.astype(np.int16)


def extract_face_labels(scene) -> Tuple['trimesh.Trimesh', np.ndarray]:
    '''
    Args:
        scene: trimesh.Scene with multiple geometries
    Returns:
        mesh: trimesh.Trimesh from merging geometries
        labels: array of face labels indicating the
            source geometry of each face in the scene
    '''
    import trimesh

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


def assign_voxel_labels(
    mask: np.ndarray,
    affine: np.ndarray,
    mesh: 'trimesh.Trimesh',
    labels: np.ndarray,
    method: str
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
        method: 'closest_point' | 'nearby_faces'
    Returns:
        labeled: (I, J, K) labeled voxel mask
    '''
    I, J, K = np.nonzero(mask)
    points_voxel = np.c_[I, J, K]
    points_model = transforms.voxel_to_world_coords(points_voxel, affine)

    labeled = np.zeros_like(mask, dtype=np.int16) # reserve background = 0
    labeled[I, J, K] = query_face_labels(mesh, labels, points_model, method) + 1

    return labeled


def query_face_labels(
    mesh: 'trimesh.Trimesh',
    labels: np.ndarray,
    points: np.ndarray,
    method: str,
    chunk_size: int = 10000
) -> np.ndarray:
    '''
    Map face labels to arbitrary points by computing
    the nearest face in the mesh to each query point.

    Args:
        mesh: trimesh.Trimesh object
        labels: (N,) array of face labels
        points: (M,3) array of query points,
            in same coordinate system as mesh
        method: 'closest_point' | 'nearby_faces'
    Returns:
        values: (M,) array of label values
    '''
    import sys, tqdm, trimesh

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueErrror(f'Invalid points shape: {points.shape}')

    if method not in {'closest_point', 'nearby_faces'}:
        raise ValueError(f'Invalid query method: {method:r}')

    query = trimesh.proximity.ProximityQuery(mesh)
    points = np.asarray(points, dtype=np.float32)
    output = np.empty(len(points), dtype=int)

    def _most_common_value(arr):
        values, counts = np.unique(arr, return_counts=True)
        return values[np.argmax(counts)]
    
    for start in tqdm.tqdm(range(0, len(points), chunk_size), file=sys.stdout):
        end = min(start + chunk_size, len(points))

        if method == 'closest_point':
            _, _, face_inds = query.on_surface(points[start:end])

        elif method == 'nearby_faces':
            candidates = trimesh.proximity.nearby_faces(mesh, points[start:end])
            face_inds = np.array([_most_common_value(c) for c in candidates], dtype=int)

        output[start:end] = labels[face_inds]
    
    return output

