from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F

from ..core import utils, fileio


def _as_tensor(a, device):
    return torch.as_tensor(a, dtype=torch.float, device=device)


def _as_array(t):
    return t.detach().cpu().numpy()


def _relative_error(a, b, m):
    from numpy.linalg import norm
    a, b, m = a.flatten(), b.flatten(), m.flatten()
    return norm((a - b) * m) / norm(b * m)


def _correlation_r2(a, b, m):
    mu_a = np.mean(a * m)
    mu_b = np.mean(b * m)
    dev_a = (a - mu_a)
    dev_b = (b - mu_b)
    std_a = np.sqrt(np.mean(dev_a**2 * m))
    std_b = np.sqrt(np.mean(dev_b**2 * m))
    numer = dev_a * dev_b
    denom = std_a * std_b
    return (numer / denom * m).mean()**2


# ----- unigradicon backend -----


def run_unigradicon_registration(
    fixed_image: Path,
    moving_image: Path,
    fixed_mask: Path,
    moving_mask: Path,
    output_path: Path,
    **kwargs
):
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        transform_path = Path(tmpdir) / 'transform.hdf5'
        raw_disp_path = Path(tmpdir) / 'raw_disp.nii.gz'

        run_unigradicon_command(
            fixed_image=fixed_image,
            fixed_mask=fixed_mask,
            moving_image=moving_image,
            moving_mask=moving_mask,
            output_path=transform_path,
            **kwargs
        )
        convert_itk_transform(
            input_path=transform_path,
            output_path=raw_disp_path,
            ref_path=fixed_image
        )
        canonicalize_itk_disp(
            input_path=raw_disp_path,
            output_path=output_path
        )


def run_unigradicon_command(
    fixed_image: Path,
    fixed_mask: Path,
    moving_image: Path,
    moving_mask: Path,
    output_path: Path,
    model: str = 'unigradicon',
    io_sim: str = 'lncc2',
    io_iterations: int = 50,
    io_sigma: int = 1,
    loss_function_masking: bool = False,
    intensity_conservation: bool = False,
    executable: str = 'unigradicon-register'
):
    import subprocess

    if model not in {'unigradicon', 'multigradicon', 'gradiconlung'}:
        raise ValueError(f'Invalid model: {model!r}')
    if io_sim not in {'lncc', 'lncc2', 'mind'}:
        raise ValueError(f'Invalid io_sim: {io_sim!r}')

    cmd = [
        str(executable),
        f'--fixed={fixed_image}',
        f'--fixed_modality=ct',
        f'--fixed_segmentation={fixed_mask}',
        f'--moving={moving_image}',
        f'--moving_segmentation={moving_mask}',
        f'--moving_modality=ct',
        f'--transform_out={output_path}',
        f'--model={model}',
        f'--io_sim={io_sim}',
        f'--io_iterations={io_iterations}',
        f'--sigma={io_sigma}'
    ]
    if loss_function_masking:
        cmd.append('--loss_function_masking')
    if intensity_conservation:
        cmd.append('--intensity_conservation_loss')

    subprocess.run(cmd, check=True)


def convert_itk_transform(
    input_path: Path, output_path: Path, ref_path: Path
):
    import itk

    transform = itk.transformread(str(input_path))[0]
    ref_image = itk.imread(str(ref_path))

    disp_image = itk.transform_to_displacement_field_filter(
        transform,
        reference_image=ref_image,
        use_reference_image=True
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    itk.imwrite(disp_image, str(output_path))


def canonicalize_itk_disp(input_path: Path, output_path: Path):

    nifti = fileio.load_nibabel(input_path)
    array = nifti.get_fdata()

    if array.ndim == 5 and array.shape[-2] == 1:
        array = array[...,0,:]

    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f'Invalid DVF shape: {nifti.shape}')

    # ITK displacement vectors use LPS world coordinate system.
    # Convert vector components to RAS basis expected by NIFTI.
    #   Reference: itk::NiftiImageIO::ConvertRASVectorsOn()
    array[...,0] *= -1
    array[...,1] *= -1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_nibabel(output_path, array.astype(np.float32), nifti.affine)


# ----- corrfield backend -----


def run_corrfield_registration(
    fixed_image: Path,
    moving_image: Path,
    fixed_mask: Path,
    output_path: Path,
    device: str='cuda'
):
    fixed_nifti  = fileio.load_nibabel(fixed_image)
    moving_nifti = fileio.load_nibabel(moving_image)
    mask_nifti   = fileio.load_nibabel(fixed_mask)

    fixed_array  = fixed_nifti.get_fdata()
    moving_array = moving_nifti.get_fdata()
    mask_array   = mask_nifti.get_fdata() > 0 # ensure binary

    disp_voxel, warped_array = register_corrfield(
        fixed_image=fixed_array,
        moving_image=moving_array,
        fixed_mask=mask_array,
        device=device
    )

    affine = fixed_nifti.affine # apply linear transform only
    disp_world = np.einsum('wv,ijkv->ijkw', affine[:3,:3], disp_voxel)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_nibabel(output_path, disp_world.astype(np.float32), affine)


def register_corrfield(
    fixed_image: np.ndarray,
    moving_image: np.ndarray,
    fixed_mask: np.ndarray,
    evaluate: bool = True,
    device: str = 'cuda'
) -> np.ndarray:
    '''
    Use CorrField deformable registration to estimate
    a *voxel* displacement field between two 3D images.

    Args:
        moving_image: (I, J, K) array
        fixed_image: (I, J, K) array
        fixed_mask:  (I, J, K) array
    Returns:
        disp_field: (I, J, K, 3) array
        warped_image: (I, J, K) array
    '''
    import corrfield
    assert moving_image.shape == fixed_image.shape == fixed_mask.shape

    moving_tensor = _as_tensor(moving_image, device)
    fixed_tensor = _as_tensor(fixed_image, device)
    mask_tensor = _as_tensor(fixed_mask, device)

    utils.log('Registering images using CorrField')

    disp_tensor, _, _ = corrfield.corrfield.corrfield(
        img_mov=moving_tensor[None,None,...], # (1, 1, I, J, K)
        img_fix=fixed_tensor[None,None,...],  # (1, 1, I, J, K)
        mask_fix=mask_tensor[None,None,...],  # (1, 1, I, J, K)
    )
    disp_tensor = disp_tensor[0] # (1, I, J, K, 3) -> (I, J, K, 3)

    utils.log('Applying deformation to moving image')

    warped_tensor = deform_image(moving_tensor, disp_tensor)

    disp_field = _as_array(disp_tensor)
    warped_image = _as_array(warped_tensor)

    if evaluate:
        utils.log('Evaluating registration metrics')

        e1 = _relative_error(moving_image, fixed_image, fixed_mask)
        e2 = _relative_error(warped_image, fixed_image, fixed_mask)

        utils.log(f'Rel. error:   {e1 * 100:.2f}% -> {e2 * 100:.2f}%')
        if e2 >= e1:
            utils.warn('WARNING: registration did not decrease error')

        r1 = _correlation_r2(moving_image, fixed_image, fixed_mask)
        r2 = _correlation_r2(warped_image, fixed_image, fixed_mask)

        utils.log(f'Correlation:  {r1:.4f} -> {r2:.4f}')
        if r2 <= r1:
            utils.warn('WARNING: registration did not increase correlation')

    return disp_field, warped_image


def deform_image(image: torch.Tensor, disp: torch.Tensor):
    '''
    Args:
        image: (I, J, K)
        disp: (I, J, K, 3)
    Returns:
        warped: (I, J, K)
    '''
    import corrfield
    I, J, K = disp.shape[:3]

    grid = F.affine_grid(
        torch.eye(3, 4, dtype=torch.float, device=disp.device)[None,...],
        size=(1,1,I,J,K),
        align_corners=True
    )

    disp = corrfield.utils.flow_pt(
        disp[None,...], # (1, I, J, K, 3)
        shape=(I,J,K),
        align_corners=True
    )

    warped = F.grid_sample(
        input=image[None,None,...], # (1, 1, I, J, K)
        grid=grid + disp,           # (1, I, J, K, 3)
        align_corners=True
    )[0,0] # (1, 1, I, J, K) -> (I, J, K)

    return warped


# ----- simpleitk backend -----


def register_simpleitk(
    image_mov: sitk.Image,
    image_fix: sitk.Image,
    transform: str='similarity',
    center: str='geometry',
    metric: str='MI',
    scale_init: float=1.0,
    num_scale_steps: int=0,
    scale_step_size: float=0.1,
    learning_rate: float=1.0,
    num_iterations: int=0,
    print_every: int=10,
):
    '''
    Perform rigid image registration using SITK.

    This is intended for intra-patient registration as a
    preprocessing step to generate images with the same
    shape and similar FOV for input to deep learning model.

    Args:
        image_mov: SITK moving image
        image_fix: SITK fixed image
        transform: Type of transformation model
            Options are 'affine', 'similarity', or 'rigid'
        center: Type of center for initialization
            Options are 'geometric' or 'moments'
        metric: Metric for image registration
            Options are 'MSE' or 'MI'
    Returns:
        transform: SITK transform object
    '''
    import SimpleITK as sitk

    transform_type = transform
    if transform_type == 'affine':
        transform = sitk.AffineTransform(3)
    elif transform_type == 'similarity':
        transform = sitk.Similarity3DTransform()
        transform.SetScale(scale_init)
    elif transform_type == 'rigid':
        transform = sitk.Euler3DTransform()

    if center == 'geometry':
        center = sitk.CenteredTransformInitializerFilter.GEOMETRY
    elif center == 'moments':
        center = sitk.CenteredTransformInitializerFilter.MOMENTS

    # initialize center of rotation parameters
    transform = sitk.CenteredTransformInitializer(
        image_fix, image_mov, transform, center
    )

    reg_method = sitk.ImageRegistrationMethod()
    reg_method.SetInterpolator(sitk.sitkLinear)

    if metric == 'MSE':
        reg_method.SetMetricAsMeanSquares()
    elif metric == 'MI':
        reg_method.SetMetricAsMattesMutualInformation()

    # initialize scale parameter by grid search
    if num_scale_steps > 0: 
        print('Start exhaustive search...')
        num_angle_steps = 0   # very slow if nonzero
        angle_step_size = 0.1 # _angle_to_versor(10)
        reg_method.SetInitialTransform(transform)
        reg_method.SetOptimizerAsExhaustive(
            numberOfSteps=[
                num_angle_steps,
                num_angle_steps,
                num_angle_steps,
                0, 0, 0,
                num_scale_steps
            ]
        )
        reg_method.SetOptimizerScales([
            angle_step_size,
            angle_step_size,
            angle_step_size,
            0.0, 0.0, 0.0,
            scale_step_size
        ])

        def print_iteration():
            position = reg_method.GetOptimizerPosition()
            metric = reg_method.GetMetricValue()
            print(f'{position} metric = {metric:.4f}')

        reg_method.AddCommand(sitk.sitkIterationEvent, print_iteration)
        transform = reg_method.Execute(image_fix, image_mov)
   
    # then perform iterative optimization
    print('Start iterative refinement...')
    reg_method.SetInitialTransform(transform)
    reg_method.SetOptimizerAsGradientDescent(
        learningRate=learning_rate,
        numberOfIterations=num_iterations,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    reg_method.SetOptimizerScalesFromPhysicalShift()
    reg_method.SetShrinkFactorsPerLevel([4, 2, 1])
    reg_method.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    def print_iteration():
        level = reg_method.GetCurrentLevel() + 1
        iteration = reg_method.GetOptimizerIteration()
        metric = reg_method.GetMetricValue()
        if iteration % print_every == 0:
            print(f'[level {level}|iteration {iteration}] metric = {metric:.4f}')

    reg_method.RemoveAllCommands()
    reg_method.AddCommand(sitk.sitkIterationEvent, print_iteration)
    transform = reg_method.Execute(image_fix, image_mov)

    return transform


def transform_simpleitk(
    image_mov: sitk.Image,
    image_fix: sitk.Image,
    transform: sitk.Transform,
    default=0
):
    '''
    Apply transformation to image using SITK.

    Args:
        image_mov: SITK moving image
            Input image to transform/resample.
        image_fix: SITK fixed image
            Determines output sampling grid.
        transform: SITK transform object
        default: Value for out-of-domain sampling
    Returns:
        image_warp: SITK moving image after applying
            transform and resampling on fixed image grid.
    '''
    import SimpleITK as sitk
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image_fix)
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetDefaultPixelValue(default)
    if transform is not None:
        resampler.SetTransform(transform)
    image_warp = resampler.Execute(image_mov)
    return image_warp


def _angle_to_versor(angle_degrees):
    angle_radians = np.deg2rad(angle_degrees)
    return np.sin(angle_radians / 2.0)

