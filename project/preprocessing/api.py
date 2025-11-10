import numpy as np

from ..core import fileio, utils, metrics


def _check_output(func, *args, force=False, **kwargs):
    '''
    Wrapper to call a function only if its output path
    does not already exist.
    '''
    output_path = kwargs.get('output_path')
    output_exists = False

    if output_path is not None:
        output_exists = output_path.is_file()

    if not output_exists or force:
        return func(*args, **kwargs)
    
    utils.log(f'Skipping {func.__name__}: Output {output_path} exists')
    

# ----- full pipelines -----


def preprocess_copdgene(ex, config):
    _check_output(
        resample_image_on_reference,
        reference_path=ex.paths['source_ref'],
        input_path=ex.paths['source_fixed'],
        output_path=ex.paths['fixed_image'],
        **(config.get('resample_image_on_reference') or {})
    )
    _check_output(
        resample_image_on_reference,
        reference_path=ex.paths['source_ref'],
        input_path=ex.paths['source_moving'],
        output_path=ex.paths['moving_image'],
        **(config.get('resample_image_on_reference') or {})
    )
    _check_output(
        create_segmentation_masks,
        image_path=ex.paths['fixed_image'],
        output_path=ex.paths['binary_mask'],
        **(config.get('create_segmentation_masks') or {})
    )
    _check_output(
        create_lung_region_mask,
        mask_dir=ex.paths['binary_mask'].parent,
        output_path=ex.paths['region_mask'],
        **(config.get('create_lung_region_mask') or {})
    )
    _check_output(
        create_volume_mesh_from_mask,
        mask_path=ex.paths['region_mask'],
        output_path=ex.paths['volume_mesh'],
        **(config.get('create_volume_mesh_from_mask') or {})
    )
    _check_output(
        register_displacement_field,
        fixed_path=ex.paths['fixed_image'],
        moving_path=ex.paths['moving_image'],
        mask_path=ex.paths['region_mask'],
        output_path=ex.paths['disp_field'],
        **(config.get('register_displacement_field') or {}),
    )


def preprocess_shapenet(ex, config):
    _check_output(
        preprocess_binary_mask,
        mask_path=ex.paths['source_mask'],
        mesh_path=ex.paths['source_mesh'],
        output_path=ex.paths['binary_mask'],
        **(config.get('preprocess_binary_mask') or {})
    )
    _check_output(
        preprocess_surface_mesh,
        input_path=ex.paths['source_mesh'],
        output_path=ex.paths['surface_mesh'],
        **(config.get('preprocess_surface_mesh') or {})
    )
    _check_output(
        create_mesh_region_mask,
        mask_path=ex.paths['binary_mask'],
        mesh_path=ex.paths['source_mesh'],
        output_path=ex.paths['region_mask'],
        **(config.get('create_mesh_region_mask') or {})
    )
    _check_output(
        create_volume_mesh_from_mask,
        mask_path=ex.paths['region_mask'],
        output_path=ex.paths['volume_mesh'],
        **(config.get('create_volume_mesh_from_mask') or {})
    )
    _check_output(
        create_material_mask,
        mask_path=ex.paths['region_mask'],
        output_path=ex.paths['material_mask'],
        density_path=ex.paths['density_field'],
        elastic_path=ex.paths['elastic_field'],
        **(config.get('create_material_mask') or {})
    )
    _check_output(
        create_material_fields,
        regions_path=ex.paths['region_mask'],
        materials_path=ex.paths['material_mask'],
        mesh_path=ex.paths['volume_mesh'],
        output_path=ex.paths['mat_fields'],
        force=True,
        **(config.get('create_material_fields') or {})
    )
    _check_output(
        simulate_displacement_field,
        mesh_path=ex.paths['mat_fields'],
        output_path=ex.paths['sim_fields'],
        unit_m=ex.metadata['unit'],
        force=True,
        **(config.get('simulate_displacement_field') or {})
    )
    _check_output(
        generate_volumetric_image,
        mask_path=ex.paths['material_mask'],
        output_path=ex.paths['input_image'],
        force=True,
        **(config.get('generate_volumetric_image') or {})
    )
    _check_output(
        interpolate_image_fields,
        image_path=ex.paths['input_image'],
        mesh_path=ex.paths['sim_fields'],
        output_path=ex.paths['img_fields'],
        force=True,
        **(config.get('interpolate_image_fields') or {})
    )


# ----- discrete steps -----


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
    utils.log('Done')


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

    utils.log('Done')


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
    utils.log('Done')


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
    utils.log('Done')


# --- shapenet processing ---


def preprocess_binary_mask(mask_path, mesh_path, output_path, pad_factor=0.37):
    from . import affine_fitting, mask_cleanup

    binvox = fileio.load_binvox(mask_path)
    mesh = fileio.load_meshio(mesh_path)

    utils.log('Inferring affine from mesh bounding box')
    affine = affine_fitting.infer_binvox_affine(binvox, mesh.points)

    utils.log('Cleaning up binary mask')
    mask = mask_cleanup.cleanup_binary_mask(binvox.numpy())

    utils.log('Centering binary mask')
    mask, affine = mask_cleanup.center_array_and_affine(mask, affine)

    if pad_factor > 0:
        pad = int(np.ceil(pad_factor * max(mask.shape)))
        utils.log(f'Padding binary mask by {pad:d} ({pad_factor*100}%)')
        mask, affine = mask_cleanup.pad_array_and_affine(mask, affine, pad)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_nibabel(output_path, mask.astype(np.uint8), affine)
    utils.log('Done')


def preprocess_surface_mesh(input_path, output_path):
    from . import surface_meshing
    import meshio
    mesh = fileio.load_trimesh(input_path).to_mesh()

    utils.log('Repairing surface mesh')
    mesh = surface_meshing.repair_surface_mesh(mesh)
    mesh = meshio.Mesh(points=mesh.vertices, cells=[('triangle', mesh.faces)])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_meshio(output_path, mesh)
    utils.log('Done')


def create_mesh_region_mask(mask_path, mesh_path, output_path, clean=True):
    from . import surface_meshing, mask_cleanup

    nifti = fileio.load_nibabel(mask_path)
    scene = fileio.load_trimesh(mesh_path)
    mask, affine = nifti.get_fdata(), nifti.affine

    utils.log('Extracting labels from mesh')
    mesh, labels = surface_meshing.extract_face_labels(scene)

    utils.log('Assigning labels to voxels')
    regions = surface_meshing.assign_voxel_labels(mask, affine, mesh, labels)
    if clean:
        utils.log('Cleaning up region mask')
        regions = mask_cleanup.cleanup_region_mask(
            regions, min_count=1000, keep_largest=False
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_nibabel(output_path, regions.astype(np.int16), affine)
    utils.log('Done')


def create_volume_mesh_from_mask(
    mask_path, output_path, use_affine_spacing=True, pygalmesh_kws=None
):
    from . import volume_meshing
    nifti = fileio.load_nibabel(mask_path)

    utils.log('Generating volume mesh from mask')
    mesh = volume_meshing.generate_mesh_from_mask(
        mask=nifti.get_fdata(),
        affine=nifti.affine,
        use_affine_spacing=use_affine_spacing,
        pygalmesh_kws=pygalmesh_kws,
        label_key='region'
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_meshio(output_path, mesh)
    utils.log('Done')


def create_material_mask(
    mask_path, output_path, density_path=None, elastic_path=None, sample_kws=None
):
    from . import materials

    nifti = fileio.load_nibabel(mask_path)
    region_mask = nifti.get_fdata().astype(np.int16)

    region_mats = materials.assign_materials_to_regions(region_mask, sample_kws)

    mat_mask = region_mats[region_mask]
    rho_mask, E_mask = materials.assign_material_properties(mat_mask)

    if density_path:
        density_path.parent.mkdir(parents=True, exist_ok=True)
        fileio.save_nibabel(density_path, rho_mask.astype(np.float32), nifti.affine)

    if elastic_path:
        elastic_path.parent.mkdir(parents=True, exist_ok=True)
        fileio.save_nibabel(elastic_path, E_mask.astype(np.float32), nifti.affine)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_nibabel(output_path, mat_mask.astype(np.int16), nifti.affine)


def create_material_fields(regions_path, materials_path, mesh_path, output_path, device='cuda'):
    from ..core import transforms, interpolation
    from . import materials

    reg_mask = fileio.load_nibabel(regions_path).get_fdata().astype(int)
    mat_mask = fileio.load_nibabel(materials_path).get_fdata().astype(int)
    mesh = fileio.load_meshio(mesh_path)

    region_mats = materials.compute_region_materials(reg_mask, mat_mask)

    cell_region = mesh.cell_data_dict['region']['tetra'].astype(int)
    cell_material = region_mats[cell_region]
    mesh.cell_data['material'] = [cell_material]

    rho_cells, E_cells = materials.assign_material_properties(cell_material)
    mesh.cell_data['rho'] = [rho_cells]
    mesh.cell_data['E']   = [E_cells]

    verts = mesh.points
    cells = mesh.cells_dict['tetra']

    mesh.point_data['rho'] = transforms.cell_to_node_values(verts, cells, rho_cells)
    mesh.point_data['E']   = transforms.cell_to_node_values(verts, cells, E_cells)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_meshio(output_path, mesh)


def simulate_displacement_field(
    mesh_path,
    output_path,
    nu_value=0.4,
    unit_m=1e-3, # meters per world unit
    scalar_degree=0,
    vector_degree=1,
    solver_kws=None,
    device='cuda'
):
    from . import simulation

    mesh = fileio.load_meshio(mesh_path)
    utils.log(mesh)

    verts = mesh.points # world coords
    cells = mesh.cells_dict['tetra']

    if scalar_degree == 0:
        rho_values = mesh.cell_data_dict['rho']['tetra']
        E_values   = mesh.cell_data_dict['E']['tetra']

    elif scalar_degree == 1:
        rho_values = mesh.point_data['rho']
        E_values   = mesh.point_data['E']

    utils.log('Simulating displacement using material properties')
    out_values = simulation.simulate_displacement(
        verts=verts,
        cells=cells,
        rho_values=rho_values,
        E_values=E_values,
        nu_value=nu_value,
        unit_m=unit_m,
        scalar_degree=scalar_degree,
        vector_degree=vector_degree,
        solver_kws=solver_kws,
        device=device
    )

    utils.log('Assigning simulation fields to mesh')
    for k, v in out_values.items():
        utils.log((k, v.shape, v.dtype, v.mean()))
        if v.shape[0] == len(verts):
            mesh.point_data[k] = v.astype(np.float32)
        elif v.shape[0] == len(cells):
            mesh.cell_data_dict[k] = [v.astype(np.float32)]
        else:
            raise ValueError(f'Invalid mesh field shape: {v.shape}')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_meshio(output_path, mesh)


def generate_volumetric_image(mask_path, annot_path, output_path, imp_kws=None, gen_kws=None):
    from . import materials, texturing

    nifti = fileio.load_nibabel(mask_path)
    mask = nifti.get_fdata().astype(int)

    utils.log('Building material catalog')
    mat_df = materials.build_material_catalog()
    mat_df = materials.assign_image_parameters(mat_df, **(imp_kws or {}))

    print(mat_df[['material_key', 'density_feat', 'elastic_feat', 'image_bias', 'image_range']])

    utils.log('Loading texture annotations')
    tex_df = texturing.load_texture_annotations(annot_path)

    utils.log('Generating volumetric image')
    image = texturing.generate_volumetric_image(
        mask, nifti.affine, mat_df, tex_df, **(gen_kws or {})
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_nibabel(output_path, image, nifti.affine)


def interpolate_image_fields(image_path, mesh_path, output_path, interp_kws=None, mesh_key='image'):
    from ..core import transforms
    import scipy.ndimage

    interp_kws = utils.update_defaults(interp_kws, order=1, mode='nearest')

    nifti = fileio.load_nibabel(image_path)
    image = nifti.get_fdata().astype(float)
    affine = nifti.affine

    mesh = fileio.load_meshio(mesh_path)

    utils.log('Interpolating image at mesh vertices')
    pts_world = mesh.points
    pts_voxel = transforms.world_to_voxel_coords(pts_world, affine)
    values = scipy.ndimage.map_coordinates(image, pts_voxel.T, **(interp_kws or {}))
    mesh.point_data[mesh_key] = values.astype(np.float32)

    utils.log('Interpolating image at tet cell barycenters')
    pts_world = mesh.points[mesh.cells_dict['tetra']].mean(axis=1)
    pts_voxel = transforms.world_to_voxel_coords(pts_world, affine)
    values = scipy.ndimage.map_coordinates(image, pts_voxel.T, **(interp_kws or {}))
    mesh.cell_data[mesh_key] = [values.astype(np.float32)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_meshio(output_path, mesh)


def rasterize_elasticity_field(
    image_path,
    mesh_path,
    output_path,
    E_key,
    nu_value=0.4,
    unit_m=1e-3,
    scalar_degree=1,
    device='cuda'
):
    from . import simulation

    nifti = fileio.load_nibabel(image_path)
    shape = nifti.shape
    affine = nifti.affine

    mesh = fileio.load_meshio(mesh_path)

    verts = mesh.points # world coords
    cells = mesh.cells_dict['tetra']

    if scalar_degree == 0:
        E_values = mesh.cell_data_dict[E_key]['tetra']
    elif scalar_degree == 1:
        E_values = mesh.point_data[E_key]

    utils.log('Rasterizing elasticity field on voxel grid')
    elast = simulation.rasterize_elasticity(
        shape=shape,
        affine=affine,
        verts=verts,
        cells=cells,
        E_values=E_values,
        nu_value=nu_value,
        unit_m=unit_m,
        scalar_degree=scalar_degree,
        device=device
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_nibabel(output_path, elast, affine)
