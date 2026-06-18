from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

from ..core import utils

from . import stages


def run_stage(
    func,
    *args,
    output_path: Path = None,
    force: bool = False,
    **kwargs
) -> Tuple[bool, Any]:
    '''
    Call function if output path does not exist.

    Args:
        func: Function to call.
        *args, **kwargs: Passed into function call.
    Returns:
        bool: True if the function was called.
        Any: Return value from function call.
    '''
    if output_path is None:
        raise ValueError(f'{func.__name__} requires output_path')

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if force or not output_path.exists():
        log(f'INFO: {output_path} missing; Running stage {func.__name__}')
        return True, func(*args, **kwargs)

    log(f'INFO: {output_path} exists; Skipping stage {func.__name__}')
    return False, None


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

    run_stage( # binary mask
        stages.preprocess_binary_mask,
        mask_path=ex.paths['source_mask'],
        mesh_path=ex.paths['source_mesh'],
        output_path=ex.paths['binary_mask'],
        config=config.get('binary_mask', {})
    )
    run_stage( # surface mesh
        stages.preprocess_surface_mesh,
        input_path=ex.paths['source_mesh'],
        output_path=ex.paths['surface_mesh'],
        config=config.get('surface_mesh', {})
    )
    run_stage( # region mask
        stages.create_mesh_region_mask,
        mask_path=ex.paths['binary_mask'],
        mesh_path=ex.paths['source_mesh'],
        output_path=ex.paths['region_mask'],
        config=config.get('region_mask', {})
    )
    run_stage( # volume mesh
        stages.generate_tetrahedral_mesh,
        mask_path=ex.paths['region_mask'],
        output_path=ex.paths['volume_mesh'],
        config=config.get('volume_mesh', {}),
        random_seed=subj_seed
    )
    run_stage( # material mask
        stages.create_material_mask,
        mask_path=ex.paths['region_mask'],
        output_path=ex.paths['material_mask'],
        density_path=ex.paths['density_field'],
        elastic_path=ex.paths['elastic_field'],
        poisson_path=ex.paths['poisson_field'],
        config=config.get('material_mask', {}),
        random_seed=subj_seed
    )
    run_stage( # material mesh
        stages.create_material_fields,
        regions_path=ex.paths['region_mask'],
        materials_path=ex.paths['material_mask'],
        mesh_path=ex.paths['volume_mesh'],
        output_path=ex.paths['material_mesh'],
        config=config.get('material_mesh', {})
    )
    run_stage( # input image
        stages.generate_volumetric_image,
        mask_path=ex.paths['material_mask'],
        output_path=ex.paths['input_image'],
        config=config.get('image_generation', {}),
        random_seed=subj_seed
    )
    run_stage( # interp mesh
        stages.interpolate_image_fields,
        image_path=ex.paths['input_image'],
        mesh_path=ex.paths['material_mesh'],
        output_path=ex.paths['interp_mesh'],
        config=config.get('image_interpolation', {})
    )
    run_stage( # simulate mesh
        stages.simulate_displacement_field,
        mesh_path=ex.paths['interp_mesh'],
        output_path=ex.paths['simulate_mesh'],
        unit_m=ex.metadata['unit'],
        config=config.get('displacement_simulation', {}),
        random_seed=subj_seed
    )


def preprocess_copdgene(ex, config):
    utils.check_keys(
        config,
        {'image_resampling', 'image_segmentation', 'region_labeling'} |
        {'image_registration', 'mesh_generation', 'mesh_interpolation'},
        where='preprocessing[copdgene]'
    )

    # ----- image resampling -----

    for state in ['init_state', 'curr_state']:

        run_stage( # resampled_image
            stages.resample_image_spacing,
            ref_path=ex.paths['ref_state']['source_image'],
            input_path=ex.paths[state]['source_image'],
            output_path=ex.paths[state]['resampled_image'],
            config=config.get('image_resampling', {})
        )

    # ----- image segmentation -----

    for state in ['init_state', 'curr_state']:

        run_stage( # combined_mask
            stages.create_segmentation_masks,
            input_path=ex.paths[state]['resampled_image'],
            segment_dir=ex.paths[state]['segment_dir'],
            output_path=ex.paths[state]['combined_mask'],
            config=config.get('image_segmentation', {})
        )

        run_stage( # region_map
            stages.create_multi_region_map,
            input_dir=ex.paths[state]['segment_dir'],
            output_path=ex.paths[state]['region_map'],
            config=config.get('region_labeling', {})
        )

    # ----- image registration -----

    run_stage( # disp_field
        stages.register_displacement_field,
        fixed_image=ex.paths['init_state']['resampled_image'],
        moving_image=ex.paths['curr_state']['resampled_image'],
        fixed_mask=ex.paths['init_state']['combined_mask'],
        moving_mask=ex.paths['curr_state']['combined_mask'],
        output_path=ex.paths['disp_field'],
        config=config.get('image_registration', {})
    )

    # ----- mesh construction -----

    run_stage( # region_mesh
        stages.generate_tetrahedral_mesh,
        mask_path=ex.paths['region_map'],
        output_path=ex.paths['region_mesh'],
        config=config.get('mesh_generation', {})
    )

    run_stage( # interp_mesh
        stages.interpolate_mesh_fields,
        mesh_path=ex.paths['region_mesh'],
        image_path=ex.paths['input_image'],
        disp_path=ex.paths['disp_field'],
        output_path=ex.paths['interp_mesh'],
        config=config.get('mesh_interpolation', {})
    )


def preprocess_emory4dct(ex, config):
    utils.check_keys(
        config,
        {'image_resampling', 'image_segmentation', 'region_labeling'} |
        {'image_registration', 'mesh_generation', 'mesh_interpolation'},
        where='preprocessing[emory4dct]'
    )
    
    # ----- image conversion -----

    for state in ['ref_state', 'init_state', 'curr_state']:

        run_stage( # converted_image
            stages.convert_image_to_nifti,
            input_path=ex.paths[state]['source_image'],
            output_path=ex.paths[state]['converted_image'],
            **ex.metadata['image_params']
        )

    # ----- image resampling -----

    for state in ['init_state', 'curr_state']:

        run_stage( # resampled_image
            stages.resample_image_spacing,
            ref_path=ex.paths['ref_state']['converted_image'],
            input_path=ex.paths[state]['converted_image'],
            output_path=ex.paths[state]['resampled_image'],
            config=config.get('image_resampling', {})
        )

    # ----- image segmentation -----

    for state in ['init_state', 'curr_state']:

        run_stage( # combined_mask
            stages.create_segmentation_masks,
            input_path=ex.paths[state]['resampled_image'],
            segment_dir=ex.paths[state]['segment_dir'],
            output_path=ex.paths[state]['combined_mask'],
            config=config.get('image_segmentation', {})
        )

        run_stage( # region_map
            stages.create_multi_region_map,
            input_dir=ex.paths[state]['segment_dir'],
            output_path=ex.paths[state]['region_map'],
            config=config.get('region_labeling', {})
        )

    # ----- image registration -----

    run_stage( # disp_field
        stages.register_displacement_field,
        fixed_image=ex.paths['init_state']['resampled_image'],
        moving_image=ex.paths['curr_state']['resampled_image'],
        fixed_mask=ex.paths['init_state']['combined_mask'],
        moving_mask=ex.paths['curr_state']['combined_mask'],
        output_path=ex.paths['disp_field'],
        config=config.get('image_registration', {})
    )

    # ----- mesh construction -----

    run_stage( # region_mesh
        stages.generate_tetrahedral_mesh,
        mask_path=ex.paths['region_map'],
        output_path=ex.paths['region_mesh'],
        config=config.get('mesh_generation', {})
    )

    run_stage( # interp_mesh
        stages.interpolate_mesh_fields,
        mesh_path=ex.paths['region_mesh'],
        image_path=ex.paths['input_image'],
        disp_path=ex.paths['disp_field'],
        output_path=ex.paths['interp_mesh'],
        config=config.get('mesh_interpolation', {})
    )



def preprocess_phantom(ex, config):
    raise NotImplementedError


def preprocess_bmc4dct(ex, config):
    raise NotImplementedError

