# preprocessing/stages.py

from typing import List, Dict, Tuple, Any

from pathlib import Path
import numpy as np

from ..common import utils, fileio, transforms


# ----- conversion to NIFTI -----


def convert_image_to_nifti(
    input_path: Path,
    output_path: Path,
    shape: Tuple[int, int, int],
    dtype: str,
    config: Dict[str, Any]
):
    from .operations import nifti_conversion

    array = fileio.load_binary_image(input_path, shape, dtype)

    utils.log('Converting binary image to NIFTI')
    nifti = nifti_conversion.convert_array_to_nifti(array, **config)

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

    fileio.save_nibabel(output_path, mask.astype(np.uint8), affine)


def preprocess_binary_mask(
    input_path: Path,
    output_path: Path,
    config: Dict[str, Any]
):
    from .operations import mask_processing

    nifti = fileio.load_nibabel(input_path)

    utils.log('Preprocessing binary mask')
    mask, affine = mask_processing.preprocess_binary_mask(
        mask=nifti.get_fdata(), affine=nifti.affine, **config
    )

    fileio.save_nibabel(output_path, mask, affine)


# ----- image resampling -----


def resample_image_spacing(
    input_path: Path,
    output_path: Path,
    ref_path: Path,
    config: Dict[str, Any]
):
    from .operations import image_resampling

    src_image = fileio.load_simpleitk(input_path)
    ref_image = fileio.load_simpleitk(ref_path)

    utils.log('Resampling image on reference grid')
    image = image_resampling.resample_image(src_image, ref_image, **config)

    fileio.save_simpleitk(output_path, image)


# ----- image segmentation -----


def create_segmentation_masks(
    image_path: Path,
    segment_dir: Path, # individual masks for each class
    output_path: Path, # combined mask for entire domain
    config: Dict[str, Any]
):
    from .operations import image_segmentation

    image_segmentation.run_segmentation_tasks(
        image_path=image_path,
        output_dir=segment_dir,
        output_path=output_path,
        **config
    )


# ----- image registration -----


def estimate_displacement_field(
    fixed_image: Path,
    fixed_mask: Path,
    moving_image: Path,
    moving_mask: Path,
    output_path: Path,
    config: Dict[str, Any]
):
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

    masks = {}
    for name in config['roi_order']:
        nifti = fileio.load_nibabel(input_dir / f'{name}.nii.gz')
        masks[name] = (nifti.get_fdata() != 0)
        affine = nifti.affine

    labels = region_labeling.label_anatomical_regions(masks, **config)

    fileio.save_nibabel(output_path, labels, affine)


def label_regions_from_surface(
    mask_path: Path,
    mesh_path: Path,
    output_path: Path,
    config: Dict[str, Any]
):
    from .operations import region_labeling

    nifti = fileio.load_nibabel(mask_path)
    scene = fileio.load_trimesh(mesh_path)

    labels = region_labeling.label_regions_from_surface(
        nifti.get_fdata(),
        nifti.affine,
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
    image = nifti.get_fdata(dtype=np.float32)
    domain = fileio.load_nibabel(domain_path).get_fdata() > 0
    affine = nifti.affine

    inputs = {'image': image, 'domain': domain}

    # load additional masks referenced by the config
    referenced = set()
    for prop_config in config.values():
        referenced.update(prop_config.get('terms', {}))

    for name in referenced - inputs.keys():
        nifti = fileio.load_nibabel(segment_dir / f'{label}.nii.gz')
        inputs[label] = (nifti.get_fdata() > 0) & domain

    fields = material_properties.compute_propertyfields(
        inputs, affine, config
    )

    for name, field in fields.items():
        fileio.save_nibabel(fields_dir / f'{name}.nii.gz', field, affine)

    # NOTE: This treats the domain as a single material "type"
    fileio.save_nibabel(output_path, domain.astype(np.uint8), affine)


def assign_materials_to_regions(
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
    from . import material_properties

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

    fileio.save_nibabel(elastic_path, E_mask.astype(np.float32), nifti.affine)
    fileio.save_nibabel(poisson_path, nu_mask.astype(np.float32), nifti.affine)
    fileio.save_nibabel(density_path, rho_mask.astype(np.float32), nifti.affine)
    fileio.save_nibabel(output_path, mat_mask.astype(np.int16), nifti.affine)


# ----- mesh generation / repair -----


def generate_tetrahedral_mesh(mask_path, output_path, config, random_seed=0):
    utils.check_keys(
        config,
        valid={'use_affine', 'pygalmesh_kws'},
        where='mesh_generation'
    )
    from .. import tetrahedral_meshing

    nifti = fileio.load_nibabel(mask_path)

    utils.log('Generating tetrahedral mesh')

    mesh = tetrahedral_meshing.generate_mesh_from_mask(
        mask=nifti.get_fdata(),
        affine=nifti.affine,
        use_affine=config.get('use_affine', True),
        random_seed=random_seed,
        pygalmesh_kws=config.get('pygalmesh_kws', {})
    )

    fileio.save_meshio(output_path, mesh)


def repair_triangular_mesh(input_path, output_path, config):
    utils.check_keys(
        config,
        valid={'run_pymeshfix'},
        where='surface_mesh'
    )
    from .. import triangular_meshing

    mesh = fileio.load_trimesh(input_path).to_mesh()

    utils.log('Repairing triangular mesh')

    mesh = triangular_meshing.repair_triangular_mesh(
        mesh, config.get('run_pymeshfix'), ret_meshio=True
    )

    fileio.save_meshio(output_path, mesh)


# ----- mesh field interpolation -----


def interpolate_mesh_fields(
    mesh_path: Path,
    image_path: Path,
    disp_path: Path,
    fields_dir: Path,
    output_path: Path,
    config: Dict[str, Any]
):
    from .operations import field_interpolation

    mesh = fileio.load_meshio(mesh_path)
    nifti = fileio.load_nibabel(image_path)
    image = nifti.get_fdata(dtype=np.float32)
    affine = nifti.affine

    u = fileio.load_nibabel(disp_path).get_fdata(dtype=np.float32)
    E = fileio.load_nibabel(fields_dir / 'youngs_modulus.nii.gz').get_fdata(dtype=np.float32)
    nu = fileio.load_nibabel(fields_dir / 'poisson_ratio.nii.gz').get_fdata(dtype=np.float32)
    rho = fileio.load_nibabel(fields_dir / 'density.nii.gz').get_fdata(dtype=np.float32)

    u_key = config.get('displacement_key', 'u')
    kwargs = config.get('interpolate_args', {})

    fields = {'image': image, u_key: disp, 'E': E, 'nu': nu, 'rho': rho}

    utils.log(f'Interpolating voxel fields onto mesh')

    mesh = field_interpolation.interpolate_mesh_fields(
        mesh, fields, affine, **kwargs
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

    mesh = fileio.load_meshio(mesh_path) # world coordinates

    if len(mesh.cells) != 1 or mesh.cells[0].type != 'tetra':
        block_types = [block.type for block in mesh.cells]
        raise ValueError(f'Expected exactly one tetra cell block: {block_types}')

    adapter = physics.api.get_adapter(config)
    bc_spec = physics.api.get_bc_spec(config)

    u_sim = adapter.simulate_displacement(mesh, unit_m, bc_spec) # meters

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
        valid={'material_catalog', 'texture_source', 'intensity_model', 'noise_model', 'use_simple'},
        where='image_generation'
    )
    from . import material_properties, textures, image_synthesis

    nifti = fileio.load_nibabel(mask_path)
    mask = nifti.get_fdata().astype(int)

    mat_df = materials.load_material_catalog(config['material_catalog'])

    tex_path = config['texture_source']['annotations']
    tex_df = textures.load_texture_annotations(tex_path)

    use_solid = config['texture_source']['use_solid']
    tex_cache = textures.TextureCache(tex_df)

    proc_kws = config['texture_source']['preprocessing']
    proc_spec = textures.PreprocessSpec(**proc_kws)

    def texture_map(label: int):
        tid = mat_df.loc[label].texture_id
        return tex_cache.get(tid, use_solid, proc_spec)

    utils.log('Computing intensity model')
    intensity_kws = config.get('intensity_model', {})
    outputs = materials.compute_intensity_model(
        mat_df['density_val'], mat_df['elastic_val'], **intensity_kws
    )
    mat_df['density_feat'] = outputs['density_feat']
    mat_df['elastic_feat'] = outputs['elastic_feat']
    mat_df['intensity_bias'] = outputs['intensity_bias']
    mat_df['intensity_range'] = outputs['intensity_range']
    utils.log(mat_df)

    utils.log('Generating volumetric image')
    if config.get('use_simple', False):
        rgb = not proc_spec.grayscale
        image = image_synthesis.generate_simple_image(
            mask, texture_map, seed=random_seed, rgb=rgb
        )
    else:
        noise_kws = config.get('noise_model', {})
        image = image_synthesis.generate_volumetric_image(
            mask, nifti.affine, mat_df, tex_cache, **noise_kws, random_seed=random_seed
        )

    fileio.save_nibabel(output_path, image, nifti.affine)

