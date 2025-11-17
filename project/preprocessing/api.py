from ..core import utils
from ..api import _check_keys
from . import stages


def _ensure_output(func, *args, force=False, **kwargs):
    '''
    Call a function if its output path does not already exist.
    '''
    output_path = kwargs.get('output_path')
    output_exists = False
    if output_path is not None:
        output_exists = output_path.is_file()
    if force or not output_exists:
        return func(*args, **kwargs)
    utils.log(f'INFO: {output_path} exists; Skipping stage {func.__name__}')


def preprocess_example(ex, config):
    dataset = ex.dataset.lower()
    if dataset == 'shapenet':
        return preprocess_shapenet(ex, config)
    elif dataset == 'copdgene':
        return preprocess_copdgene(ex, config)
    elif dataset == 'emory4dct':
        return preprocess_emory4dct(ex, config)
    raise ValueError(f'Invalid dataset: {ex.dataset:r}')
    

def preprocess_shapenet(ex, config):
    _check_keys(
        config,
        {'binary_mask', 'surface_mesh', 'region_mask', 'volume_mesh'} |
        {'material_mask', 'material_mesh', 'displacement_simulation'} |
        {'image_generation', 'image_interpolation'},
        where='preprocessing[shapenet]'
    )
    _ensure_output(
        stages.preprocess_binary_mask,
        mask_path=ex.paths['source_mask'],
        mesh_path=ex.paths['source_mesh'],
        output_path=ex.paths['binary_mask'],
        config=config.get('binary_mask', {})
    )
    _ensure_output(
        stages.preprocess_surface_mesh,
        input_path=ex.paths['source_mesh'],
        output_path=ex.paths['surface_mesh'],
        config=config.get('surface_mesh', {})
    )
    _ensure_output(
        stages.create_mesh_region_mask,
        mask_path=ex.paths['binary_mask'],
        mesh_path=ex.paths['source_mesh'],
        output_path=ex.paths['region_mask'],
        config=config.get('region_mask', {})
    )
    _ensure_output(
        stages.create_volume_mesh_from_mask,
        mask_path=ex.paths['region_mask'],
        output_path=ex.paths['volume_mesh'],
        config=config.get('volume_mesh', {})
    )
    _ensure_output(
        stages.create_material_mask,
        mask_path=ex.paths['region_mask'],
        output_path=ex.paths['material_mask'],
        density_path=ex.paths['density_field'],
        elastic_path=ex.paths['elastic_field'],
        config=config.get('material_mask', {})
    )
    _ensure_output(
        stages.create_material_fields,
        regions_path=ex.paths['region_mask'],
        materials_path=ex.paths['material_mask'],
        mesh_path=ex.paths['volume_mesh'],
        output_path=ex.paths['material_mesh'],
        config=config.get('material_mesh', {})
    )
    _ensure_output(
        stages.generate_volumetric_image,
        mask_path=ex.paths['material_mask'],
        output_path=ex.paths['input_image'],
        config=config.get('image_generation', {})
    )
    _ensure_output(
        stages.interpolate_image_fields,
        image_path=ex.paths['input_image'],
        mesh_path=ex.paths['material_mesh'],
        output_path=ex.paths['interp_mesh'],
        config=config.get('image_interpolation', {})
    )
    _ensure_output(
        stages.simulate_displacement_field,
        mesh_path=ex.paths['interp_mesh'],
        output_path=ex.paths['simulate_mesh'],
        unit_m=ex.metadata['unit'],
        config=config.get('displacement_simulation', {})
    )


def preprocess_copdgene(ex, config):
    _ensure_output(
        config,
        {'image_resampling', 'image_segmentation'} |
        {'region_mask', 'volume_mesh', 'deformable_registration'},
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
        reference_path=ex.paths['source_ref'],
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
        stages.create_volume_mesh_from_mask,
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

