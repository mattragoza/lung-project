# preprocessing/stages.py

from typing import Dict, Optional, Any
from pathlib import Path

import numpy as np

from ..common import fileio, utils


# ----- conversion to NIFTI -----


def convert_image_to_nifti(
    input_path: Path,
    output_path: Path,
    config: Dict[str, Any]
):
    utils.check_keys(
        config,
        valid={'shape', 'dtype', 'spacing', 'axcodes', 'slope', 'intercept'},
        where='nifti_conversion'
    )
    from .operations import nifti_conversion

    kwargs = config.copy()
    shape = kwargs.pop('shape')
    dtype = kwargs.pop('dtype')

    array = fileio.load_binary_image(input_path, shape, dtype)

    utils.log('Converting binary image to NIFTI')
    nifti = nifti_conversion.convert_array_to_nifti(array, **kwargs)

    fileio.save_nibabel(output_path, nifti)


def convert_binvox_to_nifti(
    mask_path: Path,
    mesh_path: Path,
    output_path: Path,
    config: Dict[str, Any]
):
    from .operations import nifti_conversion

    binvox = fileio.load_binvox(mask_path)
    mesh = fileio.load_meshio(mesh_path)

    utils.log('Converting binvox mask to NIFTI')
    nifti = nifti_conversion.convert_binvox_to_nifti(
        binvox, mesh.points, **config
    )

    fileio.save_nibabel(output_path, nifti)


# ----- image resampling -----


def resample_image_spacing(
    input_path: Path,
    output_path: Path,
    reference_path: Path,
    config: Dict[str, Any]
):
    utils.check_keys(
        config,
        valid={'spacing', 'interpolator', 'default_value'},
        where='image_resampling'
    )
    from .operations import image_resampling

    src_image = fileio.load_simpleitk(input_path)
    ref_image = fileio.load_simpleitk(reference_path)

    utils.log('Resampling image on reference grid')
    image = image_resampling.resample_image_spacing(
        src_image, ref_image, **config
    )

    fileio.save_simpleitk(output_path, image)


# ----- image segmentation -----


def create_segmentation_masks(
    image_path: Path,
    segment_dir: Path,
    output_path: Path,
    config: Dict[str, Any]
):
    '''
    Run segmentation tasks and write individual + combined domain masks.
    '''
    utils.check_keys(
        config, valid={'tasks'}, where='image_segmentation'
    )
    from .operations import image_segmentation

    fileio.make_dir_exist(segment_dir)

    for task_config in config.get('tasks', []):
        image_segmentation.run_segmentation_task(
            image_path=image_path,
            output_dir=segment_dir,
            **task_config
        )

    utils.log('Combining segmentation masks')
    nifti = image_segmentation.combine_segmentation_masks(
        segment_dir, class_type='lung'
    )

    fileio.save_nibabel(output_path, nifti)


# ----- image registration -----


def estimate_displacement_field(
    fixed_image: Path,
    fixed_mask: Path,
    moving_image: Path,
    moving_mask: Path,
    output_path: Path,
    config: Dict[str, Any]
):
    utils.check_keys(
        config,
        valid={'method', 'kwargs'},
        where='image_registration'
    )
    from .operations import image_registration

    image_registration.run_image_registration(
        fixed_image=fixed_image,
        fixed_mask=fixed_mask,
        moving_image=moving_image,
        moving_mask=moving_mask,
        output_path=output_path,
        **config
    )


# ----- region labeling -----


def label_anatomical_regions(
    input_dir: Path,
    output_path: Path,
    config: Dict[str, Any]
):
    utils.check_keys(
        config,
        valid={'roi_order', 'filter_kws'},
        where='anatomical_regions'
    )
    from .operations import region_labeling

    roi_order = config['roi_order']
    filter_kws = config.get('filter_kws', {})

    if not input_dir.is_dir():
        raise RuntimeError(f'{input_dir} is not a valid directory')

    masks, affine = {}, None
    for name in roi_order:
        nifti = fileio.load_nibabel(input_dir / f'{name}.nii.gz')
        masks[name], affine = nifti.get_fdata(), nifti.affine

    utils.log('Assigning labels to anatomical regions')
    labels = region_labeling.label_anatomical_regions(
        masks, roi_order=roi_order, filter_kws=filter_kws
    )

    fileio.save_nibabel(output_path, labels, affine)


def label_regions_from_surface(
    mask_path: Path,
    mesh_path: Path,
    output_path: Path,
    config: Dict[str, Any],
):
    from .operations import region_labeling

    nifti = fileio.load_nibabel(mask_path)
    scene = fileio.load_trimesh(mesh_path)

    utils.log('Assigning labels using surface regions')
    labels = region_labeling.label_regions_from_surface(
        mask=nifti.get_fdata(),
        affine=nifti.affine,
        scene=scene,
        **config
    )

    fileio.save_nibabel(output_path, labels, nifti.affine)


# ----- material properties -----


def assign_material_properties(
    image_path: Path,
    domain_path: Path,
    segment_dir: Path,
    output_path: Path,
    fields_dir: Path,
    config: Dict[str, Any]
):
    from .operations import material_properties

    nifti = fileio.load_nibabel(image_path)
    domain = fileio.load_nibabel(domain_path).get_fdata() > 0

    inputs = {'image': nifti.get_fdata(), 'domain': domain}
    for name in material_properties.get_referenced_names(config) - inputs.keys():
        mask = fileio.load_nibabel(segment_dir / f'{name}.nii.gz').get_fdata() > 0
        inputs[name] = mask & domain

    utils.log('Assigning material property fields')
    fields = material_properties.compute_property_fields(
        inputs, nifti.affine, config
    )

    fileio.make_dir_exist(fields_dir)
    for name, array in fields.items():
        fileio.save_nibabel(fields_dir / f'{name}.nii.gz', array, nifti.affine)

    # The current patient-specific phantom uses one material label while
    #   storing the actual physical parameters in separate dense fields.
    fileio.save_nibabel(output_path, domain.astype(np.uint8), nifti.affine)


def assign_materials_to_regions( # deprecate
    mask_path: Path,
    output_path: Path,
    density_path: Path,
    elastic_path: Path,
    poisson_path: Path,
    config: Dict[str, Any],
    random_seed: int = 0
):
    utils.check_keys(
        config,
        valid={'material_catalog', 'material_sampling'},
        where='material_labels'
    )
    from .operations import material_properties

    nifti = fileio.load_nibabel(mask_path)
    region_labels = nifti.get_fdata().astype(np.int16)

    utils.log('Loading material catalog')
    mat_df = material_properties.load_material_catalog(config['material_catalog'])
    utils.log(mat_df)

    material_labels = material_properties.assign_materials_to_region_mask(
        region_labels,
        mat_df,
        sampling_kws=config.get('material_sampling'),
        random_seed=random_seed,
    )
    E, nu, rho = material_properties.assign_material_properties(
        material_labels,
        mat_df,
    )

    for path in (density_path, elastic_path, poisson_path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    fileio.save_nibabel(elastic_path, E.astype(np.float32), nifti.affine)
    fileio.save_nibabel(poisson_path, nu.astype(np.float32), nifti.affine)
    fileio.save_nibabel(density_path, rho.astype(np.float32), nifti.affine)
    fileio.save_nibabel(output_path, material_labels.astype(np.int16), nifti.affine)


# ----- mesh generation / repair -----


def generate_tetrahedral_mesh(
    mask_path: Path,
    output_path: Path,
    config: Dict[str, Any],
    random_seed: int = 0
):
    utils.check_keys(
        config,
        valid={'use_affine', 'pygalmesh_kws'},
        where='mesh_generation'
    )
    from .operations import tetrahedral_meshing

    nifti = fileio.load_nibabel(mask_path)

    utils.log('Generating tetrahedral mesh')
    mesh = tetrahedral_meshing.generate_mesh_from_mask(
        mask=nifti.get_fdata(),
        affine=nifti.affine,
        random_seed=random_seed,
        **config
    )

    fileio.save_meshio(output_path, mesh)


def repair_triangular_mesh(
    input_path: Path,
    output_path: Path,
    config: Dict[str, Any],
):
    utils.check_keys(
        config,
        valid={'run_pymeshfix'},
        where='surface_mesh'
    )
    from .operations import triangular_meshing

    mesh = fileio.load_trimesh(input_path).to_mesh()

    utils.log('Repairing triangular mesh')
    mesh = triangular_meshing.repair_triangular_mesh(
        mesh, ret_meshio=True, **config
    )

    fileio.save_meshio(output_path, mesh)


# ----- mesh field interpolation -----


def interpolate_mesh_fields(
    mesh_path: Path,
    image_path: Path,
    disp_path: Path,
    output_path: Path,
    config: Dict[str, Any],
    fields_dir: Optional[Path] = None
):
    utils.check_keys(
        config,
        valid={'displacement_key', 'interpolate_kws'},
        where='field_interpolation'
    )
    from .operations import field_interpolation

    disp_key = config.get('displacement_key', 'u')
    interp_kws = config.get('interpolate_kws', {})

    mesh = fileio.load_meshio(mesh_path)
    nifti = fileio.load_nibabel(image_path)

    fields = {
        'image': nifti.get_fdata(),
        disp_key: fileio.load_nibabel(disp_path).get_fdata()
    }

    if fields_dir is not None:
        fields.update({
            'E': fileio.load_nibabel(fields_dir / 'youngs_modulus.nii.gz').get_fdata(),
            'nu': fileio.load_nibabel(fields_dir / 'poisson_ratio.nii.gz').get_fdata(),
            'rho': fileio.load_nibabel(fields_dir / 'density.nii.gz').get_fdata(),
        })

    utils.log('Interpolating voxel fields onto mesh')
    mesh = field_interpolation.interpolate_mesh_fields(
        mesh, fields, nifti.affine, interp_kws,
    )

    fileio.save_meshio(output_path, mesh)


def interpolate_materials(
    mesh_path: Path,
    regions_path: Path,
    materials_path: Path,
    output_path: Path,
    config: Dict[str, Any],
):
    utils.check_keys(config, valid={'material_catalog'}, where='material_mesh')
    from .operations import field_interpolation, material_properties

    mesh = fileio.load_meshio(mesh_path)
    region_labels = fileio.load_nibabel(regions_path).get_fdata().astype(int)
    material_labels = fileio.load_nibabel(materials_path).get_fdata().astype(int)

    utils.log('Loading material catalog')
    mat_df = material_properties.load_material_catalog(config['material_catalog'])
    utils.log(mat_df)

    mesh = field_interpolation.interpolate_materials(
        mesh,
        region_labels,
        material_labels,
        mat_df,
    )

    fileio.save_meshio(output_path, mesh)


def interpolate_image(
    image_path: Path,
    mesh_path: Path,
    output_path: Path,
    config: Dict[str, Any]
):
    from .operations import field_interpolation

    nifti = fileio.load_nibabel(image_path)
    image = nifti.get_fdata(dtype=np.float32)
    mesh = fileio.load_meshio(mesh_path)

    utils.log('Interpolating image onto mesh')
    mesh = field_interpolation.interpolate_mesh_fields(
        mesh, {'image': image}, nifti.affine, **config
    )

    fileio.save_meshio(output_path, mesh)


# ----- displacement simulation -----


def simulate_displacement_field(
    mesh_path: Path,
    output_path: Path,
    unit_m: float,
    config: Dict[str, Any]
):
    utils.check_keys(
        config,
        valid={'physics_adapter', 'pde_solver', 'boundary_condition', 'output_key'},
        where='displacement_simulation'
    )
    from .. import physics

    mesh = fileio.load_meshio(mesh_path)

    cell_blocks = [block.type for block in mesh.cells]
    if cell_blocks != ['tetra']:
        raise ValueError(f'Expected one tetra cell block: {cell_blocks!r}')

    adapter = physics.api.get_adapter(config)
    bc_spec = physics.api.get_bc_spec(config)
    u_sim = adapter.simulate_displacement(mesh, unit_m, bc_spec)  # meters

    def _to_numpy(t):
        return t.detach().cpu().numpy()

    output_key = config.get('output_key', 'u')
    mesh.cell_data[output_key] = [_to_numpy(u_sim.cell_values) / unit_m]
    mesh.point_data[output_key] = _to_numpy(u_sim.node_values) / unit_m

    fileio.save_meshio(output_path, mesh)


# ----- synthetic image generation -----


def generate_synthetic_image(
    mask_path: Path,
    output_path: Path,
    config: Dict[str, Any],
    random_seed: int = 0
):
    utils.check_keys(
        config,
        valid={
            'material_catalog',
            'texture_source',
            'intensity_model',
            'noise_model',
            'use_simple',
        },
        where='image_generation'
    )
    from .operations import image_synthesis

    nifti = fileio.load_nibabel(mask_path)
    mask = nifti.get_fdata().astype(int)

    image = image_synthesis.generate_synthetic_image(
        mask, nifti.affine, config, random_seed=random_seed
    )

    fileio.save_nibabel(output_path, image, nifti.affine)

