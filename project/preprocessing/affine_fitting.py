import numpy as np

from ..core import utils


def infer_binvox_affine(binvox, points):

    shape = np.asarray(binvox.dims)
    translate = np.asarray(binvox.translate)
    scale = float(binvox.scale)

    utils.log(f'Binvox shape:     {shape}')
    utils.log(f'Binvox translate: {translate}')
    utils.log(f'Binvox scale:     {scale}')

    bbox_min, bbox_extent = compute_bbox(points)

    utils.log(f'Points bbox min:    {bbox_min}')
    utils.log(f'Points bbox extent: {bbox_extent}')

    sign  = infer_sign(translate, bbox_min)
    power = infer_power(scale, bbox_extent.max())

    utils.log((sign, power))

    spacing = scale ** power / shape
    origin = translate * sign + spacing / 2
    affine = build_affine_matrix(origin, spacing)

    utils.log(affine)
    
    return affine


def build_affine_matrix(origin, spacing):
    ox, oy, oz = origin
    sx, sy, sz = spacing
    return np.array([
        [sx, 0., 0., ox],
        [0., sy, 0., oy],
        [0., 0., sz, oz],
        [0., 0., 0., 1.],
    ], dtype=float)


def compute_bbox(points):
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    bbox_extent = bbox_max - bbox_min
    return bbox_min, bbox_extent


def infer_sign(a, b, tol=1e-3):
    e_pos = relative_error(a, +b)
    e_neg = relative_error(a, -b)
    if min(e_pos, e_neg) < tol:
        return -1 if e_neg < e_pos else +1


def infer_power(a, b, tol=1e-3):
    e_eq  = relative_error(a, b)
    e_inv = relative_error(a * b, 1)
    if min(e_eq, e_inv) < tol:
        return -1 if e_inv < e_eq else +1


def relative_error(a, b, eps=1e-12):
    from numpy.linalg import norm
    return norm(a - b) / (norm(b) + eps)

