import numpy as np
import xarray as xr
import SimpleITK as sitk


class Registration(object):
    '''
    Intensity-based registration using SimpleElastix.
    '''
    def __init__(self, learning_rate=1.0, n_iterations=100):
        self.method = sitk.ImageRegistrationMethod()
        self.method.SetMetricAsMeanSquares()
        self.method.SetInterpolator(sitk.sitkLinear)
        self.method.SetOptimizerAsRegularStepGradientDescent(
            learningRate=learning_rate,
            numberOfIterations=n_iterations,
            gradientMagnitudeTolerance=1e-8,
            minStep=1e-4
        )
        self.method.SetOptimizerScalesFromPhysicalShift()
        self.method.SetShrinkFactorsPerLevel([4, 2, 1])
        self.method.SetSmoothingSigmasPerLevel([2, 1, 0])
        self.method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        self.method.AddCommand(sitk.sitkIterationEvent, self.print_metric)

    @property
    def iteration(self):
        return self.method.GetOptimizerIteration()

    @property
    def metric(self):
        return self.method.GetMetricValue()

    @property
    def optimizer_position(self):
        return self.method.GetOptimizerPosition()

    @property
    def optimizer_scale(self):
        return self.method.GetOptimizerScales()

    def print_metric(self):
        if self.iteration == 0:
            print('Estimated scales: ', self.optimizer_scale)
        print(f'[{self.iteration:4}] {self.metric}')

    def execute(self, fixed: xr.DataArray, moving: xr.DataArray, type='rigid'):
        fixed_image = image_from_xarray(fixed)
        moving_image = image_from_xarray(moving)
        initial = getattr(Deformation, type)(fixed_image, moving_image)
        self.method.SetInitialTransform(initial.transform)
        final_transform = self.method.Execute(fixed_image, moving_image)
        return Deformation(final_transform)


class Deformation(object):

    def __init__(self, transform):
        self.transform = transform

    @classmethod
    def rigid(cls, fixed, moving):
        transform = sitk.CenteredTransformInitializer(
            fixed_image,
            moving_image,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        return cls(transform)

    @classmethod
    def affine(cls, fixed, moving):
        transform = sitk.CenteredTransformInitializer(
            fixed_image,
            moving_image,
            sitk.AffineTransform(3),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        return cls(transform)

    @classmethod
    def bspline(cls, fixed, moving):
        transform = sitk.BSplineTransform(3, 3)
        return cls(transform)

    def invert(self):
        inverse = self.transform.GetInverse()
        return Deformation(transform=inverse)

    def compose(self, other):
        composite = sitk.CompositeTransform(3)
        composite.AddTransform(self.transform)
        composite.AddTransform(other.transform)
        return Deformation(transform=composite)

    def apply(self, points):
        return np.array([
            self.transform.TransformPoint(p) for p in points
        ])


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
