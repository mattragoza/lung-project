# TODO rename module to pipelines.py
from typing import List, Dict, Tuple, Optional, Any

from ..core import utils
from . import stages


def _ensure_output(func, *args, **kwargs) -> Tuple[bool, Any]:
    '''
    Call a function if the output path does not exist.

    Args:
        func: Function to call.
        *args, **kwargs: Passed into function call.
    Returns:
        bool: True if the function was called.
        Any: Returned from the function call.
    '''
    output_path = kwargs.get('output_path')

    run_stage = True
    if output_path is not None:
        run_stage = not output_path.exists()

    if run_stage:
        utils.log(f'INFO: {output_path} missing; Running stage {func.__name__}')
        return True, func(*args, **kwargs)

    utils.log(f'INFO: {output_path} exists; Skipping stage {func.__name__}')
    return False, None


def preprocess_example(ex, config):
    dataset = ex.dataset.lower()
    if dataset == 'shapenet':
        return preprocess_shapenet(ex, config)
    elif dataset == 'copdgene':
        return preprocess_copdgene(ex, config)
    elif dataset in {'emory4dct', 'emory-4dct'}:
        return preprocess_emory4dct(ex, config)
    elif dataset in {'bmc4dct', 'bmc-4dct'}:
        return preprocess_bmc4dct(ex, config)
    raise ValueError(f'Invalid dataset: {ex.dataset!r}')


def preprocess_shapenet(ex, config):
    utils.check_keys(
        config,
        {'binary_mask', 'surface_mesh', 'region_mask', 'volume_mesh'} |
        {'material_mask', 'material_mesh', 'displacement_simulation'} |
        {'image_generation', 'image_interpolation', 'random_seed'},
        where='preprocessing[shapenet]'
    )
    base_seed = config.pop('random_seed', 0)
    subj_seed = utils.make_seed(base_seed, ex.subject)

    _ensure_output( # binary mask
        stages.preprocess_binary_mask,
        mask_path=ex.paths['source_mask'],
        mesh_path=ex.paths['source_mesh'],
        output_path=ex.paths['binary_mask'],
        config=config.get('binary_mask', {})
    )
    _ensure_output( # surface mesh
        stages.preprocess_surface_mesh,
        input_path=ex.paths['source_mesh'],
        output_path=ex.paths['surface_mesh'],
        config=config.get('surface_mesh', {})
    )
    _ensure_output( # region mask
        stages.create_mesh_region_mask,
        mask_path=ex.paths['binary_mask'],
        mesh_path=ex.paths['source_mesh'],
        output_path=ex.paths['region_mask'],
        config=config.get('region_mask', {})
    )
    _ensure_output( # volume mesh
        stages.generate_tetrahedral_mesh,
        mask_path=ex.paths['region_mask'],
        output_path=ex.paths['volume_mesh'],
        config=config.get('volume_mesh', {}),
        random_seed=subj_seed
    )
    _ensure_output( # material mask
        stages.create_material_mask,
        mask_path=ex.paths['region_mask'],
        output_path=ex.paths['material_mask'],
        density_path=ex.paths['density_field'],
        elastic_path=ex.paths['elastic_field'],
        poisson_path=ex.paths['poisson_field'],
        config=config.get('material_mask', {}),
        random_seed=subj_seed
    )
    _ensure_output( # material mesh
        stages.create_material_fields,
        regions_path=ex.paths['region_mask'],
        materials_path=ex.paths['material_mask'],
        mesh_path=ex.paths['volume_mesh'],
        output_path=ex.paths['material_mesh'],
        config=config.get('material_mesh', {})
    )
    _ensure_output( # input image
        stages.generate_volumetric_image,
        mask_path=ex.paths['material_mask'],
        output_path=ex.paths['input_image'],
        config=config.get('image_generation', {}),
        random_seed=subj_seed
    )
    _ensure_output( # interp mesh
        stages.interpolate_image_fields,
        image_path=ex.paths['input_image'],
        mesh_path=ex.paths['material_mesh'],
        output_path=ex.paths['interp_mesh'],
        config=config.get('image_interpolation', {})
    )
    _ensure_output( # simulate mesh
        stages.simulate_displacement_field,
        mesh_path=ex.paths['interp_mesh'],
        output_path=ex.paths['simulate_mesh'],
        unit_m=ex.metadata['unit'],
        config=config.get('displacement_simulation', {}),
        random_seed=subj_seed
    )


def preprocess_copdgene(ex, config): # TODO needs update
    utils.check_keys(
        config,
        {'image_resampling', 'image_segmentation'} |
        {'region_mask', 'volume_mesh', 'image_registration'},
        where='preprocessing[copdgene]'
    )
    _ensure_output(
        stages.resample_image_on_reference,
        reference_path=ex.paths['source_ref'],
        input_path=ex.paths['source_fixed'],
        output_path=ex.paths['fixed_image'],
        config=config.get('image_resampling', {})
    )
    _ensure_output(
        stages.resample_image_on_reference,
        ref_path=ex.paths['source_ref'],
        input_path=ex.paths['source_moving'],
        output_path=ex.paths['moving_image'],
        config=config.get('image_resampling', {})
    )
    _ensure_output(
        stages.create_segmentation_masks,
        image_path=ex.paths['fixed_image'],
        output_path=ex.paths['binary_mask'],
        config=config.get('image_segmentation', {})
    )
    _ensure_output(
        stages.create_lung_region_mask,
        mask_dir=ex.paths['binary_mask'].parent,
        output_path=ex.paths['region_mask'],
        config=config.get('region_mask', {})
    )
    _ensure_output(
        stages.generate_tetrahedral_mesh,
        mask_path=ex.paths['region_mask'],
        output_path=ex.paths['volume_mesh'],
        config=config.get('volume_mesh', {})
    )
    _ensure_output(
        stages.register_displacement_field,
        fixed_path=ex.paths['fixed_image'],
        moving_path=ex.paths['moving_image'],
        mask_path=ex.paths['region_mask'],
        output_path=ex.paths['disp_field'],
        config=config.get('deformable_registration', {})
    )


def preprocess_emory4dct(ex, config):
    utils.check_keys(
        config,
        {'image_resampling', 'image_segmentation', 'region_labeling'} |
        {'image_registration', 'mesh_generation', 'mesh_interpolation'},
        where='preprocessing[emory4dct]'
    )
    
    # ----- image conversion -----

    _ensure_output( # ref_nifti
        stages.convert_image_to_nifti,
        input_path=ex.paths['ref_source'],
        output_path=ex.paths['ref_nifti'],
        **ex.metadata['image_params']
    )
    _ensure_output( # init_nifti
        stages.convert_image_to_nifti,
        input_path=ex.paths['init_source'],
        output_path=ex.paths['init_nifti'],
        **ex.metadata['image_params']
    )
    _ensure_output( # curr_nifti
        stages.convert_image_to_nifti,
        input_path=ex.paths['curr_source'],
        output_path=ex.paths['curr_nifti'],
        **ex.metadata['image_params']
    )

    # ----- image resampling -----

    _ensure_output( # init_resample
        stages.resample_image_spacing,
        ref_path=ex.paths['ref_nifti'],
        input_path=ex.paths['init_nifti'],
        output_path=ex.paths['init_resample'],
        config=config.get('image_resampling', {})
    )
    _ensure_output( # curr_resample
        stages.resample_image_spacing,
        ref_path=ex.paths['ref_nifti'],
        input_path=ex.paths['curr_nifti'],
        output_path=ex.paths['curr_resample'],
        config=config.get('image_resampling', {})
    )

    # ----- image segmentation -----

    _ensure_output( # segment_dir
        stages.create_segmentation_masks,
        input_path=ex.paths['init_resample'],
        segment_dir=ex.paths['segment_dir'],
        output_path=ex.paths['segment_mask'],
        config=config.get('image_segmentation', {})
    )
    _ensure_output( # region_map
        stages.create_multi_region_map,
        input_dir=ex.paths['segment_dir'],
        output_path=ex.paths['region_map'],
        config=config.get('region_labeling', {})
    )

    # ----- image registration -----

    _ensure_output( # disp_field
        stages.register_displacement_field,
        fixed_path=ex.paths['init_resample'],
        moving_path=ex.paths['curr_resample'],
        mask_path=ex.paths['segment_mask'],
        output_path=ex.paths['disp_field'],
        config=config.get('image_registration', {})
    )

    # ----- mesh construction -----

    _ensure_output( # region_mesh
        stages.generate_tetrahedral_mesh,
        mask_path=ex.paths['region_map'],
        output_path=ex.paths['region_mesh'],
        config=config.get('mesh_generation', {})
    )

    _ensure_output( # interp_mesh
        stages.interpolate_mesh_fields,
        mesh_path=ex.paths['region_mesh'],
        image_path=ex.paths['input_image'],
        disp_path=ex.paths['disp_field'],
        output_path=ex.paths['interp_mesh'],
        config=config.get('mesh_interpolation', {})
    )


def preprocess_bmc4dct(ex, config):
    raise NotImplementedError

