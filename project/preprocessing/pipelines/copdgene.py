# preprocessing/pipelines/copdgene.py

from ...common import utils
from .. import stages
from ..runner import run_stage


def preprocess(ex, config):
    utils.check_keys(
        config,
        {'image_resampling', 'image_segmentation', 'image_registration'} |
        {'anatomical_regions', 'material_properties', 'mesh_generation'} |
        {'field_interpolation', 'displacement_simulation'},
        where='preprocessing[copdgene]'
    )

    for state in ['init_state', 'curr_state']:
        run_stage(
            stages.resample_image_spacing,
            input_path=ex.paths[state]['source_image'],
            output_path=ex.paths[state]['resampled_image'],
            reference_path=ex.paths['ref_state']['source_image'],
            config=config.get('image_resampling', {})
        )
        run_stage(
            stages.create_segmentation_masks,
            image_path=ex.paths[state]['resampled_image'],
            segment_dir=ex.paths[state]['segment_dir'],
            output_path=ex.paths[state]['domain_mask'],
            config=config.get('image_segmentation', {})
        )

    run_stage(
        stages.estimate_displacement_field,
        fixed_image=ex.paths['init_state']['resampled_image'],
        fixed_mask=ex.paths['init_state']['domain_mask'],
        moving_image=ex.paths['curr_state']['resampled_image'],
        moving_mask=ex.paths['curr_state']['domain_mask'],
        output_path=ex.paths['disp_field'],
        config=config.get('image_registration', {})
    )
    run_stage(
        stages.label_anatomical_regions,
        input_dir=ex.paths['init_state']['segment_dir'],
        output_path=ex.paths['anatomical_map'],
        config=config.get('anatomical_regions', {})
    )
    run_stage(
        stages.assign_material_properties,
        image_path=ex.paths['init_state']['resampled_image'],
        domain_path=ex.paths['init_state']['domain_mask'],
        segment_dir=ex.paths['init_state']['segment_dir'],
        output_path=ex.paths['material_map'],
        fields_dir=ex.paths['material_dir'],
        config=config.get('material_properties', {})
    )
    run_stage(
        stages.generate_tetrahedral_mesh,
        mask_path=ex.paths['anatomical_map'],
        output_path=ex.paths['anatomical_mesh'],
        config=config.get('mesh_generation', {})
    )
    run_stage(
        stages.interpolate_mesh_fields,
        mesh_path=ex.paths['anatomical_mesh'],
        image_path=ex.paths['input_image'],
        disp_path=ex.paths['disp_field'],
        fields_dir=ex.paths['material_dir'],
        output_path=ex.paths['interp_mesh'],
        config=config.get('field_interpolation', {})
    )
    run_stage(
        stages.simulate_displacement_field,
        mesh_path=ex.paths['interp_mesh'],
        output_path=ex.paths['forward_mesh'],
        unit_m=ex.metadata['unit'],
        config=config.get('displacement_simulation', {})
    )

