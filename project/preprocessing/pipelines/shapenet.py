# preprocessing/pipelines/shapenet.py

from ...core import utils
from ..runner import run_stage
from .. import stages


def preprocess(ex, config):
    utils.check_keys(
        config,
        {'binary_mask', 'surface_mesh', 'region_map', 'volume_mesh'} |
        {'material_map', 'material_mesh', 'displacement_simulation'} |
        {'image_generation', 'image_interpolation', 'random_seed'},
        where='preprocessing[shapenet]'
    )
    base_seed = config.get('random_seed', 0)
    subj_seed = utils.make_seed(base_seed, ex.subject)

    run_stage(
        stages.masks.convert_binvox_mask,
        mask_path=ex.paths['source_mask'],
        mesh_path=ex.paths['source_mesh'],
        output_path=ex.paths['binary_mask'],
        config=config.get('binary_mask', {})
    )
    run_stage(
        stages.meshes.repair_surface_mesh,
        input_path=ex.paths['source_mesh'],
        output_path=ex.paths['surface_mesh'],
        config=config.get('surface_mesh', {})
    )
    run_stage(
        stages.regions.map_regions_from_surface,
        mask_path=ex.paths['binary_mask'],
        mesh_path=ex.paths['source_mesh'],
        output_path=ex.paths['region_map'],
        config=config.get('region_map', {})
    )
    run_stage(
        stages.meshes.generate_volume_mesh,
        mask_path=ex.paths['region_map'],
        output_path=ex.paths['volume_mesh'],
        config=config.get('volume_mesh', {}),
        random_seed=subj_seed
    )
    run_stage(
        stages.materials.assign_materials_to_regions,
        mask_path=ex.paths['region_map'],
        output_path=ex.paths['material_map'],
        density_path=ex.paths['density_field'],
        elastic_path=ex.paths['elastic_field'],
        poisson_path=ex.paths['poisson_field'],
        config=config.get('material_map', {}),
        random_seed=subj_seed
    )
    run_stage(
        stages.fields.interpolate_materials,
        regions_path=ex.paths['region_map'],
        materials_path=ex.paths['material_map'],
        mesh_path=ex.paths['volume_mesh'],
        output_path=ex.paths['material_mesh'],
        config=config.get('material_mesh', {})
    )
    run_stage(
        stages.synthetic_images.generate_image,
        mask_path=ex.paths['material_map'],
        output_path=ex.paths['input_image'],
        config=config.get('image_generation', {}),
        random_seed=subj_seed
    )
    run_stage( # interp mesh
        stages.fields.interpolate_image,
        image_path=ex.paths['input_image'],
        mesh_path=ex.paths['material_mesh'],
        output_path=ex.paths['interp_mesh'],
        config=config.get('image_interpolation', {})
    )
    run_stage( # simulate mesh
        stages.simulation.simulate_displacement_field,
        mesh_path=ex.paths['interp_mesh'],
        output_path=ex.paths['simulate_mesh'],
        unit_m=ex.metadata['unit'],
        config=config.get('displacement_simulation', {}),
        random_seed=subj_seed
    )

