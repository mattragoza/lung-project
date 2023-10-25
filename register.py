import numpy as np
import xarray as xr
import SimpleITK as sitk


class Registration(object):
    '''
    A registration problem using SimpleElastix.
    '''
    def __init__(self, fixed_array, moving_array, transform='rigid'):
        self.params = sitk.GetDefaultParameterMap(transform)
        self.filter = sitk.ElastixImageFilter()
        self.filter.SetParameterMap(self.params)
        self.filter.SetFixedImage(fixed_image)
        self.filter.SetMovingImage(moving_image)

    def execute(self, verbose=False):
        self.filter.Execute()


def image_from_xarray(array):
    image = sitk.Image()
