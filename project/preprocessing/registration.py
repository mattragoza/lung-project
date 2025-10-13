import numpy as np
import SimpleITK as sitk


def create_reference_grid(image, spacing, anchor='center'):
    assert anchor in {'origin', 'center'}
    
    old_size = np.array(image.GetSize(), dtype=float)
    old_spacing = np.array(image.GetSpacing(), dtype=float)
    old_origin = np.array(image.GetOrigin(), dtype=float)

    new_spacing = np.array(spacing, dtype=float)
    new_size = np.round((old_size - 1.0) * (old_spacing / new_spacing)).astype(int) + 1
    new_size = np.maximum(new_size, 1)

    if anchor == 'origin':
        new_origin = old_origin

    elif anchor == 'center':
        D = np.array(image.GetDirection(), dtype=float).reshape(3, 3)
        center = old_origin + D @ (old_spacing * (old_size - 1) / 2)
        new_origin = center - D @ (new_spacing * (new_size - 1) / 2)

    grid = sitk.Image(tuple(int(n) for n in new_size), image.GetPixelID())
    grid.SetSpacing(tuple(new_spacing))
    grid.SetOrigin(tuple(new_origin))
    grid.SetDirection(image.GetDirection())
    return grid


def resample_image_on_grid(image, grid, interp, default):
    assert interp in {'bspline', 'linear', 'nearest'}

    if interp == 'bspline':
        interp = sitk.sitkBSpline
    elif interp == 'linear':
        interp = sitk.sitkLinear
    elif interp == 'nearest':
        interp = sitk.sitkNearestNeighbor
    
    return sitk.Resample(
        image, grid, sitk.Transform(), interp, default, image.GetPixelID()
    )


def register_image(
    image_mov,
    image_fix,
    transform='similarity',
    center='geometry',
    metric='MI',
    scale_init=1.0,
    num_scale_steps=0,
    scale_step_size=0.1,
    learning_rate=1.0,
    num_iterations=0,
    print_every=10,
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
        num_angle_steps = 0 # very slow if nonzero
        angle_step_size = 0.1 #angle_to_versor(10)
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


def transform_image(image_mov, image_fix, transform, default=0):
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
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image_fix)
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetDefaultPixelValue(default)
    if transform is not None:
        resampler.SetTransform(transform)
    image_warp = resampler.Execute(image_mov)
    return image_warp


def angle_to_versor(angle_degrees):
    angle_radians = np.deg2rad(angle_degrees)
    return np.sin(angle_radians / 2.0)

