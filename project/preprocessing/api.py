# preprocessing.api

def resample_image_on_reference(
    input_path,
    output_path,
    ref_path,
    spacing=[1.0, 1.0, 1.0],
    anchor='center',
    interp='bspline',
    default=-1000
):
    import SimpleITK as sitk
    from . import registration

    print('Creating reference grid')
    ref_image = sitk.ReadImage(ref_path)
    ref_grid = registration.create_reference_grid(ref_image, spacing, anchor)

    print('Resampling image on grid')
    input_image = sitk.ReadImage(input_path)
    output_image = registration.resample_image_on_grid(input_image, ref_grid, interp, default)
    sitk.WriteImage(output_image, output_path)

    print('Done')


def create_segmentation_masks(image_path, output_dir, combined_path=None):
    import nibabel as nib
    import totalsegmentator as ts
    import totalsegmentator.python_api
    import totalsegmentator.libs
    from . import segmentation
    output_dir.mkdir(parents=True, exist_ok=True)

    print('Running TotalSegmentator task: total')
    ts.python_api.totalsegmentator(
        image_path, output_dir, task='total', roi_subset=segmentation.TOTAL_TASK_ROIS
    )

    print('Running TotalSegmentator task: lung_vessels')
    ts.python_api.totalsegmentator(image_path, output_dir, task='lung_vessels')

    if combined_path:
        print('Combining segmentation masks: lung')
        combined_mask = ts.libs.combine_masks(output_dir, class_type='lung')
        nib.save(combined_mask, combined_path)

    print('Done')


def create_multi_region_mask(mask_dir, output_path, connectivity=1, min_count=10):
    import numpy as np
    import nibabel as nib
    from . import segmentation

    region_arrays = []
    for i, roi_name in enumerate(segmentation.ALL_TASK_ROIS):

        print(f'Processing segmentation mask: {roi_name}')
        mask_path = mask_dir / f'{roi_name}.nii.gz'
        mask_nifti = nib.load(mask_path)
        mask_array = mask_nifti.get_fdata().astype(int)
        filtered_array = segmentation.filter_connected_components(
            mask_array,
            connectivity=connectivity,
            min_count=min_count,
            max_components=(1 if roi_name in segmentation.TOTAL_TASK_ROIS else None)
        )
        region_arrays.append(filtered_array * (i + 1))

    multi_array = np.max(region_arrays, axis=0).astype(np.float64)
    multi_nifti = nib.nifti1.Nifti1Image(multi_array, mask_nifti.affine)
    nib.save(multi_nifti, output_path)

    print('Done')


def create_anatomical_mesh(
    mask_path,
    output_path,
    max_facet_distance=1.0,
    max_cell_circumradius=20.0

):
    import nibabel as nib
    import meshio
    from . import meshing

    mask_nifti = nib.load(mask_path)
    mask_array = mask_nifti.get_fdata() #.astype(int)
    resolution = mask_nifti.header.get_zooms()[:3]

    mesh_dict = meshing.generate_mesh_from_mask(
        mask_array,
        resolution,
        pygalmesh_kws={
            'max_facet_distance': max_facet_distance,
            'max_cell_circumradius': max_cell_circumradius,
            'seed': 0
        }
    )
    mesh = meshing.apply_affine_to_mesh(mesh_dict['tetra'], resolution, mask_nifti.affine)
    meshio.xdmf.write(output_path, mesh)

    print('Done')


def create_corrfield_displacement(fixed_path, moving_path, mask_path, output_path):
    import nibabel as nib
    import torch
    import torch.nn.functional as F
    import corrfield

    fixed_nifti = nib.load(fixed_path)
    moving_nifti = nib.load(moving_path)
    mask_nifti = nib.load(mask_path)

    fixed_tensor = torch.as_tensor(fixed_nifti.get_fdata(), dtype=torch.float, device='cuda')
    moving_tensor = torch.as_tensor(moving_nifti.get_fdata(), dtype=torch.float, device='cuda')
    mask_tensor = torch.as_tensor(mask_nifti.get_fdata(), dtype=torch.int, device='cuda')

    disp_tensor_ijk = corrfield.corrfield.corrfield(
        img_mov=moving_tensor[None,None,...],
        img_fix=fixed_tensor[None,None,...],
        mask_fix=mask_tensor[None,None,...]
    )[0][0]

    # check registration error
    X,Y,Z = fixed_tensor.shape
    disp_tensor_pt = corrfield.utils.flow_pt(
        disp_tensor_ijk[None,...],
        shape=(X,Y,Z),
        align_corners=True
    )
    base = F.affine_grid(
        torch.eye(3, 4, dtype=torch.float, device='cuda')[None,...],
        size=(1,1,X,Y,Z),
        align_corners=True
    )
    warped_tensor = F.grid_sample(
        input=moving_tensor[None,None,...],
        grid=base + disp_tensor_pt,
        align_corners=True
    )
    def compute_error(t):
        m = (mask_tensor[None,...] > 0)
        return torch.norm((t - fixed_tensor)*m) / torch.norm(fixed_tensor*m)

    error_before = compute_error(moving_tensor)
    error_after  = compute_error(warped_tensor)
    print(error_before, error_after)
    assert error_after < error_before, 'registration error did not decrease'

    # convert from voxel to world coordinates (mm)
    R = torch.as_tensor(fixed_nifti.affine[:3,:3], dtype=torch.float, device='cuda')
    disp_tensor_xyz = torch.einsum('ij,dhwj->dhwi', R, disp_tensor_ijk)

    disp_array = disp_tensor_xyz.detach().cpu().numpy()
    disp_nifti = nib.nifti1.Nifti1Image(disp_array, fixed_nifti.affine)
    disp_nifti.header.set_intent('vector', (), name='displacement_mm')
    nib.save(disp_nifti, output_path)

    print('Done')
