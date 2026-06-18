# preprocessing/pipelines/shapenet.py

from ...core import utils
from ..runner import run_stage
from .. import stages


def preprocess(ex, config):
    utils.check_keys(
        config,
        {'binary_mask', 'surface_mesh', 'region_mask', 'volume_mesh'} |
        {'material_mask', 'material_mesh', 'displacement_simulation'} |
        {'image_generation', 'image_interpolation', 'random_seed'},
        where='preprocessing[shapenet]'
    )
    base_seed = config.get('random_seed', 0)
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

