# preprocessing/stages/materials.py

from typing import List, Dict, Tuple, Any
from pathlib import Path
import numpy as np

from ...core import utils, fileio


def assign_tissue_properties(
    domain_path: Path,
    segment_dir: Path,
    output_path: Path,
    fields_dir: Path,
    config: Dict[str, Any]
):
    for key in config:
        utils.check_keys(
            config[key],
            valid={'priority', 'density', 'youngs_modulus', 'poisson_ratio'},
            where=f'tissue_properties.{key}'
        )

    nifti = fileio.load_nibabel(domain_path)
    domain = nifti.get_fdata() > 0
    affine = nifti.affine

    masks = {}
    for label in config:
        if label == 'background':
            masks[label] = ~domain
        elif label == 'normal':
            masks[label] = domain.copy()
        else:
            mask_path = segment_dir / f'{label}.nii.gz'
            nifti = fileio.load_nibabel(mask_path)

            if nifti.shape != domain.shape:
                raise ValueError('shape mismatch')
            if not np.allclose(nifti.affine, affine):
                raise ValueError('affine mismatch')
            
            masks[label] = (nifti.get_fdata() > 0) & domain

    shape = domain.shape
    labels = np.zeros(shape, dtype=np.int16)
    density = np.zeros(shape, dtype=np.float32)
    youngs  = np.zeros(shape, dtype=np.float32)
    poisson = np.zeros(shape, dtype=np.float32)

    by_priority = sorted(config.items(), key=lambda x: x[1]['priority'])

    for idx, (label, material) in enumerate(by_priority):
        mask = masks[label]
        labels[mask]  = idx
        density[mask] = material['density']
        youngs[mask]  = material['youngs_modulus']
        poisson[mask] = material['poisson_ratio']

    # write output paths
    fields_dir.mkdir(parents=True, exist_ok=True)
    fileio.save_nibabel(fields_dir / f'density.nii.gz', density, affine)
    fileio.save_nibabel(fields_dir / f'youngs_modulus.nii.gz', youngs, affine)
    fileio.save_nibabel(fields_dir / f'poisson_ratio.nii.gz', poisson, affine)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_nibabel(output_path, labels, affine)


def assign_materials_to_regions(
    mask_path,
    output_path,
    density_path,
    elastic_path,
    poisson_path,
    config,
    random_seed=0
):
    utils.check_keys(
        config,
        valid={'material_catalog', 'material_sampling'},
        where='material_labels'
    )
    from .. import materials

    nifti = fileio.load_nibabel(mask_path)
    region_mask = nifti.get_fdata().astype(np.int16)

    utils.log('Loading material catalog')
    mat_df = materials.load_material_catalog(config['material_catalog'])
    utils.log(mat_df)

    region_mats = materials.assign_materials_to_regions(
        region_mask,
        mat_df,
        sampling_kws=config.get('material_sampling'),
        random_seed=random_seed
    )

    mat_labels = np.unique(region_mats[region_mats > 0])
    assert len(mat_labels) > 1, f'single material: {mat_labels}'

    mat_mask = region_mats[region_mask]

    # NOTE we can always recover material properties from material label + catalog,
    #   we choose to save the material property masks here for supervised training
    E_mask, nu_mask, rho_mask = materials.assign_material_properties(mat_mask, mat_df)

    elastic_path.parent.mkdir(parents=True, exist_ok=True)
    poisson_path.parent.mkdir(parents=True, exist_ok=True)
    density_path.parent.mkdir(parents=True, exist_ok=True)

    fileio.save_nibabel(elastic_path, E_mask.astype(np.float32), nifti.affine)
    fileio.save_nibabel(poisson_path, nu_mask.astype(np.float32), nifti.affine)
    fileio.save_nibabel(density_path, rho_mask.astype(np.float32), nifti.affine)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_nibabel(output_path, mat_mask.astype(np.int16), nifti.affine)

