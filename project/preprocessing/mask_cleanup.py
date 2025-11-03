import numpy as np
import scipy
import skimage

from ..core import utils, transforms


def cleanup_binary_mask(mask):

    utils.log(f'Filtering foreground')
    mask = filter_connected_components(mask != 0, max_components=1)

    utils.log(f'Filtering background')
    mask = filter_connected_components(mask == 0) == 0

    return mask


def cleanup_region_mask(input_mask, **filter_kws):
    output_mask = np.zeros_like(input_mask)

    # filter connected components in each region
    for l in np.unique(input_mask[input_mask != 0]):
        utils.log(f'Filtering region {l}')
        m = filter_connected_components(input_mask == l, **filter_kws)
        output_mask[m] = l

    # reassign dropped voxels to nearest region
    dropped = (input_mask != 0) & (output_mask == 0)
    if np.any(dropped):
        _, indices = scipy.ndimage.distance_transform_edt(dropped, return_indices=True)
        nearest_labels = output_mask[tuple(indices)]
        output_mask[dropped] = nearest_labels[dropped]

    return output_mask


def filter_connected_components(
    mask,
    min_count=30,
    min_percent=0,
    max_components=None,
    keep_largest=True,
    connectivity=1
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

        size_ok = (c >= min_count) and (p >= min_percent)
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



def pad_array_and_affine(array, affine, pad):
    array = np.pad(array, pad, mode='constant', constant_values=0)
    origin = np.array(affine[:3,3])
    spacing = np.diag(affine[:3,:3])
    affine = transforms.build_affine_matrix(origin - pad * spacing, spacing)
    return array, affine


def center_array_and_affine(array, affine):
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

