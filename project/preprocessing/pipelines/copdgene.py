# preprocessing/pipelines/copdgene.py

from ...core import utils
from ..runner import run_stage
from .. import stages


def preprocess(ex, config):
    utils.check_keys(
        config,
        {'image_resampling', 'image_segmentation', 'image_registration'} |
        {'anatomical_regions', 'material_properties', 'mesh_generation'} |
        {'mesh_interpolation', 'forward_simulation'},
        where='preprocessing[copdgene]'
    )

    for state in ['init_state', 'curr_state']:
        run_stage(
            stages.images.resample_image_spacing,
            ref_path=ex.paths['ref_state']['source_image'],
            input_path=ex.paths[state]['source_image'],
            output_path=ex.paths[state]['resampled_image'],
            config=config.get('image_resampling', {})
        )

        run_stage(
            stages.masks.create_segmentation_masks,
            image_path=ex.paths[state]['resampled_image'],
            segment_dir=ex.paths[state]['segment_dir'],
            output_path=ex.paths[state]['domain_mask'],
            config=config.get('image_segmentation', {})
        )

    run_stage(
        stages.registration.estimate_displacement_field,
        fixed_image=ex.paths['init_state']['resampled_image'],
        fixed_mask=ex.paths['init_state']['domain_mask'],
        moving_image=ex.paths['curr_state']['resampled_image'],
        moving_mask=ex.paths['curr_state']['domain_mask'],
        output_path=ex.paths['disp_field'],
        config=config.get('image_registration', {})
    )

    run_stage(
        stages.regions.label_anatomical_regions,
        input_dir=ex.paths['init_state']['segment_dir'],
        output_path=ex.paths['anatomical_map'],
        config=config.get('anatomical_regions', {})
    )

    run_stage(
        stages.materials.assign_material_properties,
        domain_path=ex.paths['init_state']['domain_mask'],
        segment_dir=ex.paths['init_state']['segment_dir'],
        output_path=ex.paths['material_map'],
        fields_dir=ex.paths['material_dir'],
        config=config.get('material_properties', {})
    )

    run_stage(
        stages.meshes.generate_tetrahedral_mesh,
        mask_path=ex.paths['anatomical_map'],
        output_path=ex.paths['anatomical_mesh'],
        config=config.get('mesh_generation', {})
    )

    run_stage(
        stages.fields.interpolate_mesh_fields,
        mesh_path=ex.paths['anatomical_mesh'],
        image_path=ex.paths['input_image'],
        disp_path=ex.paths['disp_field'],
        fields_dir=ex.paths['material_dir'],
        output_path=ex.paths['interp_mesh'],
        config=config.get('mesh_interpolation', {})
    )

    run_stage(
        stages.simulation.simulate_displacement_field,
        mesh_path=ex.paths['interp_mesh'],
        output_path=ex.paths['forward_mesh'],
        unit_m=ex.metadata['unit'],
        config=config.get('forward_simulation', {})
    )

