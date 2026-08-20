# preprocessing/stages/materials.py

from typing import List, Dict, Tuple, Any
from pathlib import Path
import numpy as np

from ...core import utils, fileio


def assign_material_properties(
    image_path: Path,
    domain_path: Path,
    segment_dir: Path,
    output_path: Path,
    fields_dir: Path,
    config: Dict[str, Any]
):
    utils.check_keys(
        config,
        valid={'density', 'youngs_modulus', 'poisson_ratio'},
        where='material_properties'
    )

    # load image and domain mask
    nifti = fileio.load_nibabel(image_path)
    image = nifti.get_fdata(dtype=np.float32)
    affine = nifti.affine

    domain = fileio.load_nibabel(domain_path).get_fdata() > 0

    inputs = {'image': image, 'domain': domain}

    # load additional masks referenced by the config
    referenced_labels = set()
    for prop_name, prop_config in config.items():
        utils.check_keys(
            prop_config,
            valid={'default', 'range', 'terms'},
            where=f'material_properties.{prop_name}'
        )
        terms = prop_config.get('terms', {})
        for label in terms - inputs.keys():
            referenced_labels.add(label)

    for label in referenced_labels:
        mask_path = segment_dir / f'{label}.nii.gz'
        mask = fileio.load_nibabel(mask_path).get_fdata() > 0
        inputs[label] = mask & domain

    # construct property fields by combining terms
    fields = {}

    for prop_name, prop_config in config.items():
        default = prop_config.get('default', 0.0)
        field = np.full(domain.shape, default, dtype=np.float32)

        terms = prop_config.get('terms', {})
        for term_input, term_config in terms.items():
            offset = term_config.get('offset', 0.0)
            weight = term_config.get('weight', 1.0)
            field += weight * (inputs[term_input] + offset)

        if prop_config.get('range'):
            vmin, vmax = map(float, prop_config['range'])
            field = np.clip(field, vmin, vmax)

        fields[prop_name] = field

    # write output paths
    fields_dir.mkdir(parents=True, exist_ok=True)

    for prop_name, field in fields.items():
        field_path = fields_dir / f'{prop_name}.nii.gz'
        fileio.save_nibabel(field_path, field, affine)

    # NOTE: This treats the domain as a single material "type"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_nibabel(output_path, domain.astype(np.uint8), affine)


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

