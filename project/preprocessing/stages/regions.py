# preprocessing/stages/regions.py

from typing import List, Dict, Tuple, Any
from pathlib import Path
import numpy as np

from ...core import utils, fileio


def label_anatomical_regions(input_dir, output_path, config):
    utils.check_keys(
        config,
        valid={'roi_order', 'region_filter'},
        where='anatomical_regions'
    )
    import scipy
    from .. import mask_cleanup

    if not input_dir.is_dir():
        raise RuntimeError(f'{input_dir} is not a valid directory')

    roi_order = config['roi_order'] # required

    utils.log(f'Assigning labels to regions')
    label_arrays = []
    for label, roi in enumerate(roi_order, start=1): # reserve 0 for background
        mask_path = input_dir / f'{roi}.nii.gz'
        nifti = fileio.load_nibabel(mask_path)
        raw_mask = (nifti.get_fdata() != 0)
        label_arrays.append(raw_mask * label)

    raw_map = np.max(label_arrays, axis=0) # use roi order for priority
    out_map = np.zeros_like(raw_map)

    for label, roi in enumerate(roi_order, start=1):
        utils.log(f'Filtering region: {roi}')

        filter_kws = config.get('region_filter', {}).copy()
        if 'max_components' not in filter_kws:
            filter_kws['max_components'] = (1 if 'lobe' in roi.lower() else None)

        filtered = mask_cleanup.filter_connected_components(
            (raw_map == label), **filter_kws
        )
        out_map[filtered] = label

    # reassign dropped voxels to nearest region
    dropped = (raw_map != 0) & (out_map == 0)
    if np.any(dropped):
        _, indices = scipy.ndimage.distance_transform_edt(
            out_map == 0,
            return_indices=True
        )
        nearest_labels = out_map[tuple(indices)]
        out_map[dropped] = nearest_labels[dropped]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_nibabel(output_path, out_map.astype(np.float32), nifti.affine)


def map_regions_from_surface(mask_path, mesh_path, output_path, config):
    utils.check_keys(
        config,
        valid={'label_method', 'region_filter'},
        where='region_labels'
    )
    from .. import surface_meshing, mask_cleanup

    nifti = fileio.load_nibabel(mask_path)
    scene = fileio.load_trimesh(mesh_path)
    mask, affine = nifti.get_fdata(), nifti.affine

    utils.log('Extracting labels from mesh')
    mesh, labels = surface_meshing.extract_face_labels(scene)

    utils.log('Assigning labels to voxels')
    method = config.get('label_method')
    regions = surface_meshing.assign_voxel_labels(mask, affine, mesh, labels, method)

    utils.log('Cleaning up region mask')
    filter_kws = config.get('region_filter', {})
    regions = mask_cleanup.filter_region_mask(regions, **filter_kws)

    region_labels = np.unique(regions[regions > 0])
    assert len(region_labels) > 1, f'single region: {region_labels}'

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_nibabel(output_path, regions.astype(np.int16), affine)

