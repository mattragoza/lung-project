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
    xres = float(array.x[1] - array.x[0])
    yres = float(array.y[1] - array.y[0])
    zres = float(array.z[1] - array.z[0])
    image.SetSpacing((xres, yres, zres))
    x0 = float(array.x[0])
    y0 = float(array.y[0])
    z0 = float(array.z[0])
    image.SetOrigin((x0, y0, z0))
    return image


def xarray_from_image(image):
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
        },
        name='result'
    )
    return array


def register_image(
    image_mov,
    image_fix,
    learning_rate=1.0,
    num_iterations=100,
    print_every=10,
    transform='similarity'
):
    if transform == 'affine':
        transform = sitk.AffineTransform(3)
    elif transform == 'similarity':
        transform = sitk.Similarity3DTransform()
    elif transform == 'rigid':
        transform = sitk.Euler3DTransform()

    transform = sitk.CenteredTransformInitializer(
        image_fix, image_mov, transform,
        sitk.CenteredTransformInitializerFilter.MOMENTS
    )
    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMeanSquares()
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsGradientDescent(
        learningRate=learning_rate,
        numberOfIterations=num_iterations,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel([4, 2, 1])
    method.SetSmoothingSigmasPerLevel([2, 1, 0])
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    method.SetInitialTransform(transform)

    def print_iteration():
        level = method.GetCurrentLevel() + 1
        iteration = method.GetOptimizerIteration()
        error = method.GetMetricValue()
        if iteration % print_every == 0:
            print(f'[level {level}|iteration {iteration}] error = {error:.4f}')

    method.AddCommand(sitk.sitkIterationEvent, print_iteration)

    transform = method.Execute(image_fix, image_mov)
    return transform


def transform_image(image_mov, image_fix, transform, default=0):
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
    return array_warp
