from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F

from ..core import utils


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


def register_corrfield(
    image_mov: np.ndarray,
    image_fix: np.ndarray,
    mask_fix: np.ndarray,
    device: str='cuda'
) -> np.ndarray:
    '''
    Args:
        image_mov: (I, J, K) array
        image_fix: (I, J, K) array
        mask_fix:  (I, J, K) array
    Returns:
        disp_voxel: (L, M, N, 3) array
        image_warp: (I, J, L, C) array
    '''
    import corrfield
    assert image_mov.shape == image_fix.shape == mask_fix.shape

    moving_tensor = _as_tensor(image_mov, device)
    fixed_tensor = _as_tensor(image_fix, device)
    mask_tensor = _as_tensor(mask_fix, device)

    utils.log('Registering images using CorrField')

    disp_voxel, kpts_fix, kpts_warp = corrfield.corrfield.corrfield(
        img_mov=moving_tensor[None,None,...], # (1, 1, I, J, K)
        img_fix=fixed_tensor[None,None,...],  # (1, 1, I, J, K)
        mask_fix=mask_tensor[None,None,...],  # (1, 1, I, J, K)
    )
    disp_voxel = disp_voxel[0] # (1, I, J, K, 3) -> (I, J, K, 3)

    utils.log('Applying deformation to moving image')
    warp_tensor = deform_image(moving_tensor, disp_voxel)

    disp_voxel = _as_array(disp_voxel)
    image_warp = _as_array(warp_tensor)

    utils.log('Evaluating registration metrics')

    e1 = _relative_error(image_mov, image_fix, mask_fix)
    e2 = _relative_error(image_warp, image_fix, mask_fix)

    utils.log(f'Rel. error:   {e1 * 100:.2f}% -> {e2 * 100:.2f}%')
    if e2 >= e1:
        utils.warn('WARNING: registration did not decrease error')

    r1 = _correlation_r2(image_mov, image_fix, mask_fix)
    r2 = _correlation_r2(image_warp, image_fix, mask_fix)

    utils.log(f'Correlation:  {r1:.4f} -> {r2:.4f}')
    if r2 <= r1:
        utils.warn('WARNING: registration did not increase correlation')

    return disp_voxel, image_warp


def deform_image(image: torch.Tensor, disp_voxel: torch.Tensor):
    '''
    Args:
        image: (I, J, K)
        disp_voxel: (I, J, K, 3)
    Returns:
        warped: (I, J, K)
    '''
    import corrfield
    I, J, K = disp_voxel.shape[:3]

    grid = F.affine_grid(
        torch.eye(3, 4, dtype=torch.float, device=disp_voxel.device)[None,...],
        size=(1,1,I,J,K),
        align_corners=True
    )

    disp = corrfield.utils.flow_pt(
        disp_voxel[None,...], # (1, I, J, K, 3)
        shape=(I,J,K),
        align_corners=True
    )

    warped = F.grid_sample(
        input=image[None,None,...], # (1, 1, I, J, K)
        grid=grid + disp,           # (1, I, J, K, 3)
        align_corners=True
    )[0,0] # (1, 1, I, J, K) -> (I, J, K)

    return warped


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
    Perform non-deformable image registration using SITK.

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

