import numpy as np
import xarray as xr
import SimpleITK as sitk


class Registration(object):
    '''
    A registration problem using SimpleElastix.
    '''
    def __init__(self, fixed, moving, transform='rigid'):
        self.params = sitk.GetDefaultParameterMap(transform)
        self.filter = sitk.ElastixImageFilter()
        self.filter.SetParameterMap(self.params)
        self.fixed = fixed
        self.moving = moving

    def get_result(self):
        return xarray_from_image(self.filter.GetResultImage())

    def execute(self, verbose=False):
        self.filter.SetFixedImage(image_from_xarray(self.fixed))
        self.filter.SetMovingImage(image_from_xarray(self.moving))
        self.filter.Execute()


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
