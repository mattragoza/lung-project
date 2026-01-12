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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_nibabel(output_path, regions.astype(np.int16), affine)


def create_volume_mesh_from_mask(mask_path, output_path, config):
    utils.check_keys(
        config,
        valid={'use_affine_spacing', 'meshing_parameters'},
        where='volume_mesh'
    )
    from . import volume_meshing
    nifti = fileio.load_nibabel(mask_path)

    utils.log('Generating volume mesh from mask')
    mesh = volume_meshing.generate_mesh_from_mask(
        mask=nifti.get_fdata(),
        affine=nifti.affine,
        use_affine_spacing=config.get('use_affine_spacing'),
        pygalmesh_kws=config.get('meshing_parameters', {}),
        label_key='region'
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_meshio(output_path, mesh)


def create_material_mask(mask_path, output_path, density_path, elastic_path, config):
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
        region_mask, mat_df, sampling_kws=config.get('material_sampling')
    )
    mat_mask = region_mats[region_mask]

    # NOTE we can always recover material properties from material label + catalog,
    #   we choose to save the material property masks here for supervised training
    rho_mask, E_mask = materials.assign_material_properties(mat_mask, mat_df)

    density_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_nibabel(density_path, rho_mask.astype(np.float32), nifti.affine)

    elastic_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_nibabel(elastic_path, E_mask.astype(np.float32), nifti.affine)

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


def generate_volumetric_image(mask_path, output_path, sid, config):
    utils.check_keys(
        config,
        valid={'material_catalog', 'intensity_model', 'texture_source', 'noise_model'},
        where='image_generation'
    )
    from . import materials, textures, image_generation

    nifti = fileio.load_nibabel(mask_path)
    mask = nifti.get_fdata().astype(int)

    utils.log('Loading material catalog')
    mat_df = materials.load_material_catalog(config['material_catalog'])

    utils.log('Computing intensity model')
    density = mat_df['density_val']
    elastic = mat_df['elastic_val']    

    intensity_kws = config.get('intensity_model', {})
    outputs = materials.compute_intensity_model(density, elastic, **intensity_kws)

    mat_df['density_feat'] = outputs['density_feat']
    mat_df['elastic_feat'] = outputs['elastic_feat']
    mat_df['intensity_bias'] = outputs['intensity_bias']
    mat_df['intensity_range'] = outputs['intensity_range']
    utils.log(mat_df)

    utils.log('Initializing texture cache')
    tex_kws = config.get('texture_source', {})
    tex_cache = textures.TextureCache(**tex_kws)

    utils.log('Generating volumetric image')
    noise_kws = config.get('noise_model', {})

    def make_seed(sid, seed):
        import hashlib
        h = hashlib.sha256(f'{sid}|{seed}'.encode('utf-8')).digest()
        return int.from_bytes(h[:8], byteorder='little', signed=False)

    seed = make_seed(sid, noise_kws.pop('seed', 0))
    image = image_generation.generate_volumetric_image(
        mask, nifti.affine, mat_df, tex_cache, seed=seed, **noise_kws
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_nibabel(output_path, image, nifti.affine)


def interpolate_image_fields(image_path, mesh_path, output_path, config):
    utils.check_keys(config, valid={'order', 'mode'}, where='image_interpolation')
    from ..core import transforms
    import scipy.ndimage

    nifti = fileio.load_nibabel(image_path)
    image = nifti.get_fdata().astype(float)
    affine = nifti.affine

    mesh = fileio.load_meshio(mesh_path)

    utils.log('Interpolating image at mesh vertices')
    pts_world = mesh.points
    pts_voxel = transforms.world_to_voxel_coords(pts_world, affine)
    values = scipy.ndimage.map_coordinates(image, pts_voxel.T, **config)
    mesh.point_data['image'] = values.astype(np.float32)

    utils.log('Interpolating image at tetra cell centroids')
    pts_world = mesh.points[mesh.cells_dict['tetra']].mean(axis=1)
    pts_voxel = transforms.world_to_voxel_coords(pts_world, affine)
    values = scipy.ndimage.map_coordinates(image, pts_voxel.T, **config)
    mesh.cell_data['image'] = [values.astype(np.float32)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_meshio(output_path, mesh)


def simulate_displacement_field(mesh_path, output_path, unit_m, config):
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
    outputs = physics_adapter.simulate(mesh, unit_m, bc_spec=None)

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


# ---- CT image preprocessing -----


def resample_image_on_reference(
    input_path,
    output_path,
    reference_path,
    spacing=(1.0, 1.0, 1.0),
    interp='bspline',
    default=-1000.
):
    from . import resampling
    input_image = fileio.load_simpleitk(input_path)
    ref_image = fileio.load_simpleitk(reference_path)

    utils.log('Creating reference grid')
    grid = resampling.create_reference_grid(ref_image, spacing, anchor='center')

    utils.log('Resampling image on grid')
    output_image = resampling.resample_image(input_image, grid, interp, default)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_simpleitk(output_path, output_image)


def create_segmentation_masks(image_path, output_path):
    from . import segmentation

    output_dir = output_path.parent
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    utils.log('Running TotalSegmentator task: total')
    ret = segmentation.run_total_segmentator(image_path, output_dir)

    utils.log('Running TotalSegmentator task: lung_vessels')
    ret = segmentation.run_vessel_segmentation(image_path, output_dir)

    utils.log('Combining segmentation masks: lung')
    nifti = segmentation.combine_segmentation_masks(output_dir, class_type='lung')
    fileio.save_nibabel(output_path, nifti.get_fdata(), nifti.affine)


def create_lung_region_mask(mask_dir, output_path, min_count=30):
    from . import segmentation, mask_cleanup
    all_rois = list(segmentation.ALL_TASK_ROIS)
    lobe_rois = set(segmentation.TOTAL_TASK_ROIS)

    if not mask_dir.is_dir():
        raise RuntimeError(f'{mask_dir} is not a valid directory')

    label_arrays = []
    for label, roi in enumerate(all_rois, start=1): # reserve 0 for background

        mask_path = mask_dir / f'{roi}.nii.gz'
        nifti = fileio.load_nibabel(mask_path)
        mask_array, affine = nifti.get_fdata(), nifti.affine

        utils.log(f'Filtering segmentation mask: {roi}')
        max_comps = 1 if roi in lobe_rois else None
        filt_array = mask_cleanup.filter_connected_components(
            (mask_array != 0),
            min_count=min_count,
            max_components=max_comps
        ).astype(np.uint8)
        label_arrays.append(filt_array * label)

    utils.log('Combining anatomical region masks')
    multi_array = np.max(label_arrays, axis=0)

    utils.log('Cleaning up anatomical region mask')
    multi_array *= mask_cleanup.cleanup_binary_mask(multi_array > 0)

    fileio.save_nibabel(output_path, multi_array.astype(np.float32), affine)


def register_displacement_field(
    fixed_path, moving_path, mask_path, output_path, device='cuda'
):
    from . import registration

    fixed_nifti  = fileio.load_nibabel(fixed_path)
    moving_nifti = fileio.load_nibabel(moving_path)
    mask_nifti   = fileio.load_nibabel(mask_path)

    fixed_array  = fixed_nifti.get_fdata()
    moving_array = moving_nifti.get_fdata()
    mask_array   = mask_nifti.get_fdata() > 0 # ensure binary

    utils.log('Estimating displacement field by registration')
    disp_voxel, warped_array = registration.register_corrfield(
        image_mov=moving_array,
        image_fix=fixed_array,
        mask_fix=mask_array,
        device=device
    )

    utils.log('Mapping displacement field to world coordinates')
    affine = fixed_nifti.affine # apply linear transform only
    disp_world = np.einsum('wv,ijkv->ijkw', affine[:3,:3], disp_voxel)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_nibabel(output_path, disp_world.astype(np.float32), affine)

