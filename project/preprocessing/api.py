import numpy as np

from ..core import fileio, utils, metrics


def _check_output(func, *args, **kwargs):
    '''
    Wrapper to call a function only if its output path
    does not already exist.
    '''
    output_path = kwargs.get('output_path')
    output_exists = False

    if output_path is not None:
        output_exists = output_path.is_file()

    if not output_exists:
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
        create_material_fields,
        mask_path=ex.paths['region_mask'],
        density_path=ex.paths['density_field'],
        elastic_path=ex.paths['elastic_field'],
        output_path=ex.paths['material_mask'],
        **(config.get('create_material_fields') or {})
    )
    _check_output(
        simulate_displacement_field,
        mesh_path=ex.paths['volume_mesh'],
        density_path=ex.paths['density_field'],
        elastic_path=ex.paths['elastic_field'],
        nodes_path=ex.paths['node_values'],
        output_path=ex.paths['disp_field'],
        unit_m=ex.metadata['unit'],
        **(config.get('simulate_displacement_field') or {})
    )
    _check_output(
        generate_volumetric_image,
        mask_path=ex.paths['material_mask'],
        output_path=ex.paths['input_image'],
        **(config.get('generate_volumetric_image') or {})
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


def create_volume_mesh_from_mask(
    mask_path,
    output_path,
    use_affine_spacing=True,
    pygalmesh_kws=None
):
    from . import volume_meshing
    nifti = fileio.load_nibabel(mask_path)

    utils.log('Generating volume mesh from mask')
    mesh = volume_meshing.generate_mesh_from_mask(
        mask=nifti.get_fdata(), 
        affine=nifti.affine,
        use_affine_spacing=use_affine_spacing,
        pygalmesh_kws=pygalmesh_kws
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_meshio(output_path, mesh)
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


def create_material_fields(mask_path, density_path, elastic_path, output_path):
    from . import materials

    nifti = fileio.load_nibabel(mask_path)
    region_mask, affine = nifti.get_fdata(), nifti.affine

    utils.log('Assigning material properties to regions')
    m_mask, d_mask, e_mask = materials.assign_material_properties(region_mask)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    density_path.parent.mkdir(parents=True, exist_ok=True)
    elastic_path.parent.mkdir(parents=True, exist_ok=True)

    fileio.save_nibabel(density_path, d_mask.astype(np.float32), affine)
    fileio.save_nibabel(elastic_path, e_mask.astype(np.float32), affine)
    fileio.save_nibabel(output_path, m_mask.astype(np.int16), affine)


def simulate_displacement_field(
    mesh_path,
    density_path,
    elastic_path,
    nodes_path,
    output_path,
    nu_value=0.4,
    unit_m=1e-3,
    solver_kws=None,
    rasterize=True
):
    from . import simulation

    mesh = fileio.load_meshio(mesh_path)
    density_nifti = fileio.load_nibabel(density_path)
    elastic_nifti = fileio.load_nibabel(elastic_path)

    affine_d = density_nifti.affine
    affine_e = elastic_nifti.affine
    assert np.allclose(affine_d, affine_e)

    rho_field = density_nifti.get_fdata().astype(np.float32)
    E_field   = elastic_nifti.get_fdata().astype(np.float32)
    assert rho_field.shape == E_field.shape

    utils.log('Simulating displacement using material fields')
    disp_field, node_values = simulation.simulate_displacement(
        mesh, affine_d, rho_field, E_field, nu_value, unit_m, solver_kws, rasterize=rasterize
    )
    for k, v in node_values.items():
        print(k, v.shape, v.dtype, v.mean())
        mesh.point_data[k] = v.astype(np.float32)

    nodes_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_meshio(nodes_path, mesh)

    if rasterize:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fileio.save_nibabel(output_path, disp_field, affine_d)


def optimize_elasticity_field(
    input_nodes_path,
    input_mask_path,
    output_nodes_path,
    output_path,
    nu_value=0.4,
    unit_m=1e-3,
    solver_kws=None,
    global_kws=None,
    local_kws=None,
    rasterize=True
):
    from . import simulation

    mesh = fileio.load_meshio(input_nodes_path)
    rho_nodes = mesh.point_data['rho']
    u_true_nodes = mesh.point_data['u'] # world units
    E_true_nodes = mesh.point_data['E'] # Pa

    nifti = fileio.load_nibabel(input_mask_path)

    utils.log('Optimizing elasticity to match observed displacement')
    E_field, node_values = simulation.optimize_elasticity(
        mesh=mesh,
        shape=nifti.get_fdata().shape,
        affine=nifti.affine,
        rho_nodes=rho_nodes,
        E_nodes=E_true_nodes,
        u_obs_nodes=u_true_nodes,
        nu_value=nu_value,
        unit_m=unit_m,
        solver_kws=solver_kws,
        global_kws=global_kws,
        local_kws=local_kws,
        rasterize=rasterize,
    )
    # save new keys in mesh file
    for k, v in node_values.items():
        print(k, v.shape, v.dtype)
        mesh.point_data[k] = v.astype(np.float32)

    # compute evaluation metrics
    u_metrics = metrics.evaluate_metrics(mesh.point_data['u_opt'], mesh.point_data['u'], which='u')
    E_metrics = metrics.evaluate_metrics(mesh.point_data['E_opt'], mesh.point_data['E'], which='E')
    r_metrics = metrics.evaluate_metrics(mesh.point_data['r_opt'], which='res')
    ret_metrics = (
        utils.namespace(u_metrics, 'u') |
        utils.namespace(E_metrics, 'E') |
        utils.namespace(r_metrics, 'res')
    )
    output_nodes_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_meshio(output_nodes_path, mesh)

    if rasterize:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fileio.save_nibabel(output_path, E_field, nifti.affine)

    return ret_metrics


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

