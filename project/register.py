import numpy as np
import xarray as xr
import SimpleITK as sitk


def image_from_xarray(array):
    '''
    Convert a 3D xarray to an SITK image.
    '''
    assert array.dims == ('x', 'y', 'z')
    data_T = array.transpose('z', 'y', 'x').data
    image = sitk.GetImageFromArray(data_T)
    x_res = float(array.x[1] - array.x[0])
    y_res = float(array.y[1] - array.y[0])
    z_res = float(array.z[1] - array.z[0])
    image.SetSpacing((x_res, y_res, z_res))
    x0 = float(array.x[0])
    y0 = float(array.y[0])
    z0 = float(array.z[0])
    image.SetOrigin((x0, y0, z0))
    return image


def xarray_from_image(image):
    '''
    Convert a 3D SITK image to an xarray.
    '''
    assert image.GetDimension() == 3
    assert image.GetNumberOfComponentsPerPixel() == 1
    data = sitk.GetArrayFromImage(image).T
    shape = image.GetSize()
    origin = image.GetOrigin()
    resolution = image.GetSpacing()
    array = xr.DataArray(
        data=data,
        dims=['x', 'y', 'z'],
        coords={
            'x': origin[0] + np.arange(shape[0]) * resolution[0],
            'y': origin[1] + np.arange(shape[1]) * resolution[1],
            'z': origin[2] + np.arange(shape[2]) * resolution[2]
        }
    )
    return array


def angle_to_versor(angle_degrees):
    angle_radians = np.deg2rad(angle_degrees)
    return np.sin(angle_radians / 2.0)


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
    resampler.SetTransform(transform)
    image_warp = resampler.Execute(image_mov)
    return image_warp


def register_array(array_mov, array_fix, *args, **kwargs):
    image_mov = image_from_xarray(array_mov.astype(float))
    image_fix = image_from_xarray(array_fix.astype(float))
    transform = register_image(image_mov, image_fix, *args, **kwargs)
    return transform


def transform_array(array_mov, array_fix, *args, **kwargs):
    image_mov = image_from_xarray(array_mov.astype(float))
    image_fix = image_from_xarray(array_fix.astype(float))
    image_warp = transform_image(image_mov, image_fix, *args, **kwargs)
    array_warp = xarray_from_image(image_warp).astype(array_mov.dtype)
    array_warp.name = array_mov.name
    return array_warp
