# preprocessing/operations/region_labeling.py

from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import scipy.ndimage

from ...common import transforms, utils


def label_anatomical_regions(
    masks: Dict[str, np.ndarray],
    roi_order: List[str],
    filter_kws: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    from . import mask_processing

    if not roi_order:
        raise ValueError('roi_order is empty')

    filter_kws = filter_kws or {}

    label_masks = [] # reserve 0 for background
    for label, name in enumerate(roi_order, start=1):
        label_masks.append((masks[name] > 0) * label)

    # roi_order determines label priority
    raw_labels = np.max(label_masks, axis=0)
    out_labels = np.zeros_like(raw_labels, dtype=int)

    for label, name in enumerate(roi_order, start=1):
        utils.log(f'Filtering region: {name}')

        kwargs = filter_kws.copy()
        if 'max_components' not in kwargs:
            kwargs['max_components'] = 1 if 'lobe' in name.lower() else None

        filtered = mask_processing.filter_connected_components(
            (raw_labels == label), **kwargs
        )
        out_labels[filtered] = label

    dropped = (raw_labels != 0) & (out_labels == 0)
    if np.any(dropped):
        _, indices = scipy.ndimage.distance_transform_edt(
            (out_labels == 0), return_indices=True
        )
        nearest_labels = out_labels[tuple(indices)]
        out_labels[dropped] = nearest_labels[dropped]

    return out_labels.astype(np.int16)


def label_regions_from_surface(
    mask: np.ndarray,
    affine: np.ndarray,
    scene: 'trimesh.Scene',
    query_method: str,
    filter_kws: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    from . import mask_processing

    mesh, face_labels = extract_face_labels(scene)
    labels = assign_voxel_labels(mask, affine, mesh, face_labels, query_method)
    labels = mask_processing.filter_region_labels(labels, **(filter_kws or {}))

    region_labels = np.unique(labels[labels > 0])
    if len(region_labels) <= 1:
        raise RuntimeError(f'Single region label: {region_labels}')

    return labels.astype(np.int16)


def extract_face_labels(scene) -> Tuple['trimesh.Trimesh', np.ndarray]:
    import trimesh

    if len(scene.graph.nodes_geometry) == 0:
        raise ValueError('scene has no geometry')

    verts, faces, labels, offset = [], [], [], 0
    for idx, name in enumerate(scene.graph.nodes_geometry):

        transform = scene.graph[name][0]
        if not np.allclose(transform, np.eye(4)):
            utils.warn(f'WARNING: Non-identity transform on {name}')

        geometry = scene.geometry[name]
        v = trimesh.transform_points(geometry.vertices, transform)
        f = geometry.faces

        verts.append(v.copy())
        faces.append(f.copy() + offset)
        labels.append(np.full(len(f), idx, dtype=int))
        offset += len(v)

    mesh = trimesh.Trimesh(
        vertices=np.concatenate(verts, axis=0),
        faces=np.concatenate(faces, axis=0),
        process=False
    )

    return mesh, np.concatenate(labels, axis=0)


def assign_voxel_labels(
    mask: np.ndarray,
    affine: np.ndarray,
    mesh: 'trimesh.Trimesh',
    labels: np.ndarray,
    method: str
) -> np.ndarray:

    I, J, K = np.nonzero(mask)
    voxels = np.c_[I, J, K]
    points = transforms.voxel_to_world_coords(voxels, affine)

    labeled = np.zeros_like(mask, dtype=np.int16)
    labeled[I, J, K] = query_face_labels(mesh, labels, points, method) + 1

    return labeled


def query_face_labels(
    mesh: 'trimesh.Trimesh',
    labels: np.ndarray,
    points: np.ndarray,
    method: str,
    chunk_size: int = 10000
) -> np.ndarray:

    import sys, tqdm, trimesh

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f'Invalid points shape: {points.shape}')

    if method not in {'closest_point', 'nearby_faces'}:
        raise ValueError(f'Invalid query method: {method!r}')

    query = trimesh.proximity.ProximityQuery(mesh)
    points = np.asarray(points, dtype=np.float32)
    output = np.empty(len(points), dtype=int)

    for start in tqdm.tqdm(range(0, len(points), chunk_size), file=sys.stdout):
        end = min(start + chunk_size, len(points))

        if method == 'closest_point':
            _, _, face_inds = query.on_surface(points[start:end])
        else:
            candidates = trimesh.proximity.nearby_faces(mesh, points[start:end])
            face_inds = np.array([_most_common_value(c) for c in candidates])

        output[start:end] = labels[face_inds]

    return output


def _most_common_value(array):
    values, counts = np.unique(array, return_counts=True)
    return values[np.argmax(counts)]

