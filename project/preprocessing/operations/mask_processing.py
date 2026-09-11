# preprocessing/operations/mask_processing.py

from typing import Tuple, Optional

import numpy as np
import scipy
import skimage

from ...common import utils, transforms


def preprocess_binary_mask(
    mask: np.ndarray,
    affine: np.ndarray,
    do_filter: bool = False,
    filter_kws: dict | None = None,
    do_center: bool = False,
    pad_amount: int | float = 0,
    # backward-compatible aliases from the previous config
    foreground_filter: dict | None = None,
    background_filter: dict | None = None,
    center_mask: bool | None = None
):
    if foreground_filter is not None or background_filter is not None:
        mask = filter_binary_mask(
            mask,
            foreground_kws=foreground_filter,
            background_kws=background_filter,
        )
    elif do_filter:
        mask = filter_binary_mask(mask, **(filter_kws or {}))

    if center_mask is not None:
        do_center = center_mask

    if do_center:
        mask, affine = center_array_and_affine(mask, affine)

    if pad_amount > 0:
        mask, affine = pad_array_and_affine(mask, affine, pad_amount)

    return mask.astype(np.uint8), affine


# ----- mask / region filtering -----


def filter_binary_mask(
    mask: np.ndarray,
    foreground_kws: dict | None = None,
    background_kws: dict | None = None,
    **kwargs,
) -> np.ndarray:

    # If only one set of kwargs is supplied, use it for both passes.
    foreground_kws = kwargs if foreground_kws is None else foreground_kws
    background_kws = kwargs if background_kws is None else background_kws

    utils.log('Filtering foreground (removing blobs)')
    mask = filter_connected_components(mask != 0, **foreground_kws)

    utils.log('Filtering background (filling holes)')
    mask = ~filter_connected_components(mask == 0, **background_kws)

    return mask


def filter_region_labels(labels: np.ndarray, **kwargs):
    output = np.zeros_like(labels)

    # filter connected components in each region
    for label in np.unique(labels[labels != 0]):
        utils.log(f'Filtering region with label {label}')
        mask = filter_connected_components(labels == label, **kwargs)
        output[mask] = label

    # assign dropped voxels to nearest region
    dropped = (labels != 0) & (output == 0)

    if np.any(dropped):
        from scipy.ndimage import distance_transform_edt
        _, indices = distance_transform_edt(dropped, return_indices=True)
        nearest_labels = output[tuple(indices)]
        output[dropped] = nearest_labels[dropped]

    return output


def filter_connected_components(
    mask: np.ndarray,
    min_voxels: int = 0,
    min_percent: int = 0,
    max_components: Optional[int] = None,
    keep_largest: bool = False,
    connectivity: int = 1
):
    # label connected regions and measure their size
    labeled, input_components = skimage.measure.label(
        (mask != 0), 
        background=0,
        connectivity=connectivity,
        return_num=True
    )
    utils.log(f'Input {connectivity}-connected components: {input_components}')

    labels, counts = np.unique(labeled[labeled > 0], return_counts=True)

    total = counts.sum()
    if total == 0:
        utils.log(f'Input mask is empty')
        return np.zeros_like(mask, dtype=bool)

    percents = counts / total * 100.
    size_order = np.argsort(-counts) # largest to smallest

    utils.log(f'  Voxel counts:   {counts[size_order]} {total}')

    output_labels = []
    output_components = 0
    voxels_dropped = 0

    for rank, i in enumerate(size_order):
        l, c, p = int(labels[i]), int(counts[i]), float(percents[i])

        size_ok = (c >= min_voxels) and (p >= min_percent)
        hit_cap = max_components and (output_components >= max_components)
        keep = (size_ok and not hit_cap) or (keep_largest and rank == 0)

        if keep:
            output_labels.append(l)
            output_components += 1
        else:
            voxels_dropped += c

    output = np.isin(labeled, output_labels)

    pct_dropped = voxels_dropped / total * 100.
    utils.log(f'Output {connectivity}-connected components: {output_components}')
    utils.log(f'  Voxels dropped: {voxels_dropped} ({pct_dropped:.4f}%)')

    return output


def count_connected_components(mask, connectivity=1):
    return skimage.measure.label(
        (mask != 0),
        background=0,
        return_num=True,
        connectivity=connectivity
    )[1]


def compute_thickness_metrics(mask, p=[5, 50, 95]):
    m = mask != 0
    edt = scipy.ndimage.distance_transform_edt(m)
    dist = edt[m] # distance to nearest boundary
    return np.percentile(dist, p)


def compute_cross_section_metrics(mask, p=[5, 50, 95]):
    I, J, K = mask.shape
    m = mask != 0
    a0 = mask.mean(axis=(1,2))
    a1 = mask.mean(axis=(0,2))
    a2 = mask.mean(axis=(0,1))
    a = np.concatenate([a0, a1, a2])
    return np.percentile(a[a > 0], p)


# ----- centering and padding -----


def center_array_and_affine(
    array: np.ndarray,
    affine: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:

    if array.ndim != 3:
        raise ValueError('array must be 3D')

    center_old = scipy.ndimage.center_of_mass(array)
    center_old = np.asarray(center_old, dtype=float)
    center_new = (np.array(array.shape, dtype=float) - 1) / 2

    delta = center_new - center_old
    delta_int = np.round(delta).astype(int)
    delta_rem = delta - delta_int.astype(float)

    shifted = scipy.ndimage.shift(
        input=array,
        shift=delta_int,
        order=0,
        mode='constant',
        cval=0,
        prefilter=False
    )

    A = affine.astype(float, copy=True)
    A[:3,3] -= A[:3,:3] @ delta_int

    return shifted, A


def pad_array_and_affine(
    array: np.ndarray,
    affine: np.ndarray,
    amount: int | float = 0,
    value:  int | float = 0,
):
    if isinstance(amount, float): # interpret as shape fraction
        amount = int(np.ceil(amount * max(array.shape)))

    array = np.pad(
        array,
        amount,
        mode='constant',
        constant_values=value
    )

    origin = np.array(affine[:3,3])
    spacing = np.diag(affine[:3,:3])

    origin = origin - amount * spacing
    affine = transforms.to_affine_matrix(origin, spacing)

    return array, affine

