# preprocessing/pipelines/emory4dct.py

from ...core import utils
from ..runner import run_stage
from .. import stages


def preprocess(ex, config):
    utils.check_keys(
        config,
        {'image_resampling', 'image_segmentation', 'region_labeling'} |
        {'image_registration', 'mesh_generation', 'mesh_interpolation'},
        where='preprocessing[emory4dct]'
    )
    
    for state in ['ref_state', 'init_state', 'curr_state']:
        run_stage(
            stages.images.convert_image_to_nifti,
            input_path=ex.paths[state]['source_image'],
            output_path=ex.paths[state]['converted_image'],
            **ex.metadata['image_params']
        )

    for state in ['init_state', 'curr_state']:
        run_stage(
            stages.images.resample_image_spacing,
            ref_path=ex.paths['ref_state']['converted_image'],
            input_path=ex.paths[state]['converted_image'],
            output_path=ex.paths[state]['resampled_image'],
            config=config.get('image_resampling', {})
        )

    for state in ['init_state', 'curr_state']:
        run_stage(
            stages.masks.create_segmentation_masks,
            input_path=ex.paths[state]['resampled_image'],
            segment_dir=ex.paths[state]['segment_dir'],
            output_path=ex.paths[state]['combined_mask'],
            config=config.get('image_segmentation', {})
        )
        run_stage(
            stages.regions.map_regions_from_masks,
            input_dir=ex.paths[state]['segment_dir'],
            output_path=ex.paths[state]['region_map'],
            config=config.get('region_labeling', {})
        )

    run_stage(
        stages.registration.estimate_displacement,
        fixed_image=ex.paths['init_state']['resampled_image'],
        moving_image=ex.paths['curr_state']['resampled_image'],
        fixed_mask=ex.paths['init_state']['combined_mask'],
        moving_mask=ex.paths['curr_state']['combined_mask'],
        output_path=ex.paths['disp_field'],
        config=config.get('image_registration', {})
    )
    run_stage(
        stages.meshes.generate_volume_mesh,
        mask_path=ex.paths['region_map'],
        output_path=ex.paths['region_mesh'],
        config=config.get('mesh_generation', {})
    )
    run_stage(
        stages.fields.interpolate_mesh_fields,
        mesh_path=ex.paths['region_mesh'],
        image_path=ex.paths['input_image'],
        disp_path=ex.paths['disp_field'],
        output_path=ex.paths['interp_mesh'],
        config=config.get('mesh_interpolation', {})
    )

