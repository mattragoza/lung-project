from typing import List, Dict, Tuple
from pathlib import Path
import numpy as np
from ..core import fileio, utils


def preprocess_binary_mask(mask_path, mesh_path, output_path, config):
    utils.check_keys(
        config,
        valid={'foreground_filter', 'background_filter', 'center_mask', 'pad_amount'},
        where='binary_mask'
    )
    from . import affine_fitting, mask_cleanup

    mesh   = fileio.load_meshio(mesh_path)
    binvox = fileio.load_binvox(mask_path)
    affine = affine_fitting.infer_binvox_affine(binvox, mesh.points)

    foreground_kws = config.get('foreground_filter', {})
    background_kws = config.get('background_filter', {})
    mask = mask_cleanup.filter_binary_mask(binvox.numpy(), foreground_kws, background_kws)

    center = config.get('center_mask')
    if center:
        mask, affine = mask_cleanup.center_array_and_affine(mask, affine)

    pad = config.get('pad_amount')
    if pad > 0:
        mask, affine = mask_cleanup.pad_array_and_affine(mask, affine, pad)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_nibabel(output_path, mask.astype(np.uint8), affine)


def preprocess_surface_mesh(input_path, output_path, config):
    utils.check_keys(
        config,
        valid={'run_pymeshfix'},
        where='surface_mesh'
    )
    from . import surface_meshing
    import meshio

    mesh = fileio.load_trimesh(input_path).to_mesh()

    utils.log('Repairing surface mesh')
    use_pymeshfix = config.get('run_pymeshfix')
    mesh = surface_meshing.repair_surface_mesh(mesh, use_pymeshfix)
    mesh = meshio.Mesh(points=mesh.vertices, cells=[('triangle', mesh.faces)])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_meshio(output_path, mesh)


def create_mesh_region_mask(mask_path, mesh_path, output_path, config):
    utils.check_keys(
        config,
        valid={'label_method', 'region_filter'},
        where='region_mask'
    )
    from . import surface_meshing, mask_cleanup

    nifti = fileio.load_nibabel(mask_path)
    scene = fileio.load_trimesh(mesh_path)
    mask, affine = nifti.get_fdata(), nifti.affine

    utils.log('Extracting labels from mesh')
    mesh, labels = surface_meshing.extract_face_labels(scene)

    utils.log('Assigning labels to voxels')
    method = config.get('label_method')
    regions = surface_meshing.assign_voxel_labels(mask, affine, mesh, labels, method)

    utils.log('Cleaning up region mask')
    filter_kws = config.get('region_filter', {})
    regions = mask_cleanup.filter_region_mask(regions, **filter_kws)

    region_labels = np.unique(regions[regions > 0])
    assert len(region_labels) > 1, f'single region: {region_labels}'

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_nibabel(output_path, regions.astype(np.int16), affine)


def create_material_mask(
    mask_path,
    output_path,
    density_path,
    elastic_path,
    poisson_path,
    config,
    random_seed=0
):
    utils.check_keys(
        config,
        valid={'material_catalog', 'material_sampling'},
        where='material_mask'
    )
    from . import materials

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

    elastic_path.parent.mkdir(parents=True, exist_ok=True)
    poisson_path.parent.mkdir(parents=True, exist_ok=True)
    density_path.parent.mkdir(parents=True, exist_ok=True)

    fileio.save_nibabel(elastic_path, E_mask.astype(np.float32), nifti.affine)
    fileio.save_nibabel(poisson_path, nu_mask.astype(np.float32), nifti.affine)
    fileio.save_nibabel(density_path, rho_mask.astype(np.float32), nifti.affine)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_nibabel(output_path, mat_mask.astype(np.int16), nifti.affine)


def create_material_fields(regions_path, materials_path, mesh_path, output_path, config):
    utils.check_keys(config, valid={'material_catalog'}, where='material_mesh')
    from ..core import transforms, interpolation
    from . import materials

    reg_mask = fileio.load_nibabel(regions_path).get_fdata().astype(int)
    mat_mask = fileio.load_nibabel(materials_path).get_fdata().astype(int)
    mesh = fileio.load_meshio(mesh_path)

    # infer the region -> material label map from masks
    region_to_material = materials.infer_material_by_region(reg_mask, mat_mask)

    # assign material labels to cells using region cell labels
    reg_cells = mesh.cell_data_dict['region']['tetra'].astype(int)
    mat_cells = region_to_material[reg_cells]
    mesh.cell_data['material'] = [mat_cells]

    utils.log('Loading material catalog')
    mat_df = materials.load_material_catalog(config['material_catalog'])
    utils.log(mat_df)

    # get material properties from material labels and assign to cells
    rho_cells, E_cells = materials.assign_material_properties(mat_cells, mat_df)
    mesh.cell_data['rho'] = [rho_cells]
    mesh.cell_data['E']   = [E_cells]

    # map material properties from cells to nodes
    verts, cells = mesh.points, mesh.cells_dict['tetra']
    volume = transforms.compute_cell_volume(verts, cells)
    cells_to_nodes = transforms.compute_node_adjacency(verts, cells, volume)
    mesh.point_data['mat'] = transforms.cell_to_node_labels(verts, cells, mat_cells)
    mesh.point_data['rho'] = transforms.cell_to_node_values(cells_to_nodes, rho_cells)
    mesh.point_data['E']   = transforms.cell_to_node_values(cells_to_nodes, E_cells)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_meshio(output_path, mesh)


def generate_volumetric_image(mask_path, output_path, config, random_seed=0):
    utils.check_keys(
        config,
        valid={'material_catalog', 'texture_source', 'intensity_model', 'noise_model', 'use_simple'},
        where='image_generation'
    )
    from . import materials, textures, image_generation

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
        image = image_generation.generate_simple_image(
            mask, texture_map, seed=random_seed, rgb=rgb
        )
    else:
        noise_kws = config.get('noise_model', {})
        image = image_generation.generate_volumetric_image(
            mask, nifti.affine, mat_df, tex_cache, **noise_kws, random_seed=random_seed
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_nibabel(output_path, image, nifti.affine)


def interpolate_image_fields(image_path, mesh_path, output_path, config):
    utils.check_keys(config, valid={'order', 'mode'}, where='image_interpolation')
    from ..core import transforms
    from . import image_generation
    import scipy.ndimage

    nifti = fileio.load_nibabel(image_path)
    image = nifti.get_fdata().astype(float)
    affine = nifti.affine

    mesh = fileio.load_meshio(mesh_path)

    utils.log('Interpolating image at mesh vertices')
    pts_world = mesh.points
    pts_voxel = transforms.world_to_voxel_coords(pts_world, affine)
    values = image_generation.interpolate_volume(image, pts_voxel, **config)
    mesh.point_data['image'] = values.astype(np.float32)

    utils.log('Interpolating image at tetra cell centroids')
    pts_world = mesh.points[mesh.cells_dict['tetra']].mean(axis=1)
    pts_voxel = transforms.world_to_voxel_coords(pts_world, affine)
    values = image_generation.interpolate_volume(image, pts_voxel, **config)
    mesh.cell_data['image'] = [values.astype(np.float32)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_meshio(output_path, mesh)


def simulate_displacement_field(
    mesh_path, output_path, unit_m, config, random_seed=0
):
    utils.check_keys(
        config,
        valid={'physics_adapter', 'pde_solver'},
        where='displacement_simulation'
    )
    from .. import physics

    mesh = fileio.load_meshio(mesh_path)
    utils.log(mesh)

    physics_adapter_kws = config.get('physics_adapter', {})
    pde_solver_kws = config.get('pde_solver', {}).copy()

    physics_adapter = physics.PhysicsAdapter(
        pde_solver_cls=pde_solver_kws.pop('_class'),
        pde_solver_kws=pde_solver_kws,
        **physics_adapter_kws
    )
    bc_spec = None #physics_adapter.get_bc_spec(random_seed)
    outputs = physics_adapter.simulate(mesh, unit_m, bc_spec)

    for k, v in outputs.items():
        utils.log((k, v.shape, v.dtype, v.mean()))
        if v.shape[0] == mesh.points.shape[0]:
            mesh.point_data[k] = v.astype(np.float32)
        elif v.shape[0] == mesh.cells.shape[0]:
            mesh.cell_data_dict[k] = [v.astype(np.float32)]
        else:
            raise ValueError(f'Invalid mesh field shape: {v.shape}')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_meshio(output_path, mesh)


# ----- CT image processing -----


def convert_image_to_nifti(
    input_path,
    output_path,
    shape: Tuple[int, int, int],
    dtype: str,
    system: str,
    spacing: Tuple[float, float, float],
    slope: float = 1.,
    intercept: float = 0.
):
    def _interpret_coord_system(code):
        cx, cy, cz = code.upper()
        assert cx in 'LR', cx
        assert cy in 'PA', cy
        assert cz in 'IS', cz
        return  (
            1 if cx == 'R' else -1,
            1 if cy == 'A' else -1,
            1 if cz == 'S' else -1
        )

    signs = _interpret_coord_system(system)

    array = fileio.load_binary_image(input_path, shape, dtype)
    array = array.astype(np.float32) * slope + intercept

    affine = np.diag([
        signs[0] * spacing[0],
        signs[1] * spacing[1],
        signs[2] * spacing[2],
        1.0
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_nibabel(output_path, array, affine)


def resample_image_spacing(input_path, output_path, ref_path, config):
    utils.check_keys(
        config,
        valid={'spacing', 'interpolation', 'default_value'},
        where='image_resampling'
    )
    from . import resampling

    src_image = fileio.load_simpleitk(input_path)
    ref_image = fileio.load_simpleitk(ref_path)

    utils.log('Resampling image using reference domain')
    output_image = resampling.resample_image(src_image, ref_image, **config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_simpleitk(output_path, output_image)


def create_segmentation_masks(input_path, segment_dir, output_path, config):
    utils.check_keys(
        config,
        valid={'tasks', 'combine'},
        where='image_segmentation'
    )
    from . import segmentation

    segment_dir.mkdir(parents=True, exist_ok=True)

    for task_kws in config.get('tasks', []):
        task_name = task_kws['task_name']
        utils.log(f'Running TotalSegmentator task: {task_name}')
        segmentation.run_segmentation_task(input_path, segment_dir, **task_kws)

    if config.get('combine'):
        utils.log('Combining segmentation masks: lung')
        nifti = segmentation.combine_segmentation_masks(segment_dir, class_type='lung')

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fileio.save_nibabel(output_path, nifti.get_fdata(), nifti.affine)


def create_multi_region_map(input_dir, output_path, config):
    utils.check_keys(
        config,
        valid={'roi_order', 'region_filter'},
        where='region_mask'
    )
    from . import mask_cleanup
    import scipy, skimage

    if not input_dir.is_dir():
        raise RuntimeError(f'{input_dir} is not a valid directory')

    roi_order = config['roi_order'] # required

    utils.log(f'Assigning labels to regions')
    label_arrays = []
    for label, roi in enumerate(roi_order, start=1): # reserve 0 for background
        mask_path = input_dir / f'{roi}.nii.gz'
        nifti = fileio.load_nibabel(mask_path)
        raw_mask = (nifti.get_fdata() != 0)
        label_arrays.append(raw_mask * label)

    raw_map = np.max(label_arrays, axis=0) # use roi order for priority
    out_map = np.zeros_like(raw_map)

    for label, roi in enumerate(roi_order, start=1):
        utils.log(f'Filtering region: {roi}')

        filter_kws = config.get('region_filter', {})
        if 'max_components' not in filter_kws:
            filter_kws['max_components'] = (1 if 'lobe' in roi.lower() else None)

        filtered = mask_cleanup.filter_connected_components(
            (raw_map == label), **filter_kws
        )
        out_map[filtered] = label

    # reassign dropped voxels to nearest region
    dropped = (raw_map != 0) & (out_map == 0)
    if np.any(dropped):
        _, indices = scipy.ndimage.distance_transform_edt(dropped, return_indices=True)
        nearest_labels = out_map[tuple(indices)]
        out_map[dropped] = nearest_labels[dropped]

    fileio.save_nibabel(
        output_path, out_map.astype(np.float32), nifti.affine
    )


def register_displacement_field(
    fixed_path, moving_path, mask_path, output_path, config
):
    utils.check_keys(
        config,
        valid={},
        where='image_registration'
    )
    from . import registration
    device = 'cuda'

    fixed_nifti  = fileio.load_nibabel(fixed_path)
    moving_nifti = fileio.load_nibabel(moving_path)
    mask_nifti   = fileio.load_nibabel(mask_path)

    fixed_array  = fixed_nifti.get_fdata()
    moving_array = moving_nifti.get_fdata()
    mask_array   = mask_nifti.get_fdata() > 0 # ensure binary

    utils.log('Estimating displacement field by registration')
    disp_voxel, warped_array = registration.register_corrfield(
        moving_image=moving_array,
        fixed_image=fixed_array,
        fixed_mask=mask_array,
        device=device
    )

    utils.log('Mapping displacement field to world coordinates')
    affine = fixed_nifti.affine # apply linear transform only
    disp_world = np.einsum('wv,ijkv->ijkw', affine[:3,:3], disp_voxel)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_nibabel(output_path, disp_world.astype(np.float32), affine)


# ----- mesh processing -----


def generate_tetrahedral_mesh(mask_path, output_path, config, random_seed=0):
    utils.check_keys(
        config,
        valid={'use_affine_spacing', 'mesh_parameters'},
        where='mesh_generation'
    )
    from . import volume_meshing

    nifti = fileio.load_nibabel(mask_path)

    use_affine = config.get('use_affine_spacing', False)
    pygalmesh_kws = config.get('mesh_parameters', {})

    utils.log('Generating tetrahedral mesh')
    mesh = volume_meshing.generate_mesh_from_mask(
        mask=nifti.get_fdata(),
        affine=nifti.affine,
        use_affine=use_affine,
        random_seed=random_seed,
        pygalmesh_kws=pygalmesh_kws
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_meshio(output_path, mesh)


def interpolate_mesh_fields(
    mesh_path, image_path, disp_path, output_path, config
):
    utils.check_keys(config, valid={}, where='mesh_interpolation')

    from ..core import transforms
    from . import image_generation
    import scipy.ndimage

    mesh = fileio.load_meshio(mesh_path)

    nifti = fileio.load_nibabel(image_path)
    image = nifti.get_fdata().astype(float)
    affine = nifti.affine

    disp = fileio.load_nibabel(disp_path).get_fdata()

    utils.log('Interpolating fields onto mesh vertices')
    pts_world = mesh.points
    pts_voxel = transforms.world_to_voxel_coords(pts_world, affine)

    img_values = image_generation.interpolate_volume(image, pts_voxel, **config)
    mesh.point_data['image'] = img_values.astype(np.float32)

    disp_values = image_generation.interpolate_volume(disp, pts_voxel, **config)
    mesh.point_data['u_true'] = disp_values.astype(np.float32)

    utils.log('Interpolating fields onto tet cell centers')
    pts_world = mesh.points[mesh.cells_dict['tetra']].mean(axis=1)
    pts_voxel = transforms.world_to_voxel_coords(pts_world, affine)

    img_values = image_generation.interpolate_volume(image, pts_voxel, **config)
    mesh.cell_data['image'] = [img_values.astype(np.float32)]

    disp_values = image_generation.interpolate_volume(disp, pts_voxel, **config)
    mesh.cell_data['u_true'] = [disp_values.astype(np.float32)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_meshio(output_path, mesh)
