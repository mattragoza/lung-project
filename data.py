import sys, os, re, yaml
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import xarray as xr
import hvplot.xarray
from tqdm import tqdm

ALL_CASES = [
    'Case1Pack',
    'Case2Pack',
    'Case3Pack',
    'Case4Pack',
    'Case5Pack',
    'Case6Pack',
    'Case7Pack',
    'Case8Deploy',
    'Case9Pack',
    'Case10Pack'
]
ALL_PHASES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
CASE_NAME_RE = re.compile(r'Case(\d+)(Pack|Deploy)')


class Emory4DCTDataset(object):

    def __init__(self, data_root, case_names=ALL_CASES, phases=ALL_PHASES):
        self.data_root = Path(data_root)
        self.cases = []
        for case_name in as_list(case_names):
            case = Emory4DCTCase(data_root, case_name, phases)
            self.cases.append(case)

    def __repr__(self):
        class_name = type(self).__name__
        case_reprs = [repr(case) for case in self.cases]
        return f'{class_name}([\n  ' + ',\n  '.join(case_reprs) + '\n])'

    def __getitem__(self, idx):
        return self.cases[idx]

    @property
    def metadata_file(self):
        return self.data_root / 'metadata.tsv'

    def load_metadata(self):
        return pd.read_csv(self.metadata_file, sep='\t')

    def load_images(self):
        metadata = self.load_metadata()
        metadata.set_index('case_id', inplace=True)
        for case in self.cases:
            mdata = metadata.loc[case.case_id]
            shape = (mdata.n_x, mdata.n_y, mdata.n_z)
            resolution = (mdata.xres, mdata.yres, mdata.zres)
            case.load_images(shape, resolution)

    def describe(self):
        stats = []
        for case in tqdm(self.cases):
            case_stats = case.describe()
            stats.append(case_stats)
        return pd.concat(stats)

    def normalize(self, loc, scale):
        for case in self.cases:
            case.normalize(loc, scale)

    def save_niftis(self):
        for case in self.cases:
            case.save_niftis()

    def load_niftis(self):
        for case in self.cases:
            case.load_niftis()


class Emory4DCTCase(object):
    
    def __init__(self, data_root, case_name, phases):
        self.data_root = Path(data_root)
        self.case_name = str(case_name)
        self.phases = as_list(phases)

    def __repr__(self):
        class_name = type(self).__name__
        data_root = repr(str(self.data_root))
        case_name = repr(self.case_name)
        n_phases = len(self.phases)
        return f'{class_name}({data_root}, {case_name}, {n_phases} phases)'

    @property
    def case_id(self):
        return int(CASE_NAME_RE.match(self.case_name).group(1))
        
    @property
    def case_dir(self):
        return self.data_root / self.case_name

    @property
    def image_dir(self):
        return self.case_dir / 'Images'

    @property
    def nifti_dir(self):
        return self.case_dir / 'NIFTI'

    @property
    def mask_dir(self):
        return self.case_dir / 'TotalSegment'
        
    def load_images(self, shape, resolution):
        images = []
        for phase in self.phases:
            img_glob = self.image_dir.glob(f'case{self.case_id}_T{phase:02d}*.img')
            img_file = next(img_glob) # assumes exactly one match
            image = load_img_file(img_file, shape, dtype=np.int16)
            images.append(image)

        self.shape = shape
        self.resolution = resolution
        self.array = xr.DataArray(
            data=np.stack(images)[...,::-1], # flip z orientation
            dims=['phase', 'x', 'y', 'z'],
            coords={
                'phase': self.phases,
                'x': np.arange(shape[0]) * resolution[0],
                'y': np.arange(shape[1]) * resolution[1],
                'z': np.arange(shape[2]) * resolution[2]
            },
            name=f'case{self.case_id}'
        )

    def load_niftis(self):
        nifti_data = []
        for phase in self.phases:
            nifti_file = self.nifti_dir / f'case{self.case_id}_T{phase:02d}.nii.gz'
            print(f'Loading {nifti_file}')
            nifti = nib.load(nifti_file)
            if nifti_data:
                assert nifti.header.get_data_shape() == shape
                assert nifti.header.get_zooms() == resolution
            else:
                shape = nifti.header.get_data_shape()
                resolution = nifti.header.get_zooms()
            nifti_data.append(nifti.get_fdata())

        self.shape = shape
        self.resolution = resolution
        self.array = xr.DataArray(
            data=np.stack(nifti_data),
            dims=['phase', 'x', 'y', 'z'],
            coords={
                'phase': self.phases,
                'x': np.arange(shape[0]) * resolution[0],
                'y': np.arange(shape[1]) * resolution[1],
                'z': np.arange(shape[2]) * resolution[2]
            },
            name=f'CT'
        )

    def load_masks(self, roi='lung_combined_mask'):
        mask_data = []
        for phase in self.phases:
            mask_file = self.mask_dir / f'case{self.case_id}_T{phase:02d}/{roi}.nii.gz'
            print(f'Loading {mask_file}')
            mask = nib.load(mask_file)
            if mask_data:
                assert mask.header.get_data_shape() == shape
                assert mask.header.get_zooms() == resolution
            else:
                shape = mask.header.get_data_shape()
                resolution = mask.header.get_zooms()
            mask_data.append(mask.get_fdata())

        assert shape == self.shape
        assert resolution == self.resolution

        self.mask = xr.DataArray(
            data=np.stack(mask_data),
            dims=['phase', 'x', 'y', 'z'],
            coords={
                'phase': self.phases,
                'x': np.arange(shape[0]) * resolution[0],
                'y': np.arange(shape[1]) * resolution[1],
                'z': np.arange(shape[2]) * resolution[2]
            },
            name=f'mask'
        )

    def save_niftis(self):
        for phase in self.phases:
            data = self.array.sel(phase=phase).data
            affine = np.diag(list(self.resolution) + [1])
            nifti = nib.nifti1.Nifti1Image(data, affine)
            nifti_file = self.nifti_dir / f'case{self.case_id}_T{phase:02d}.nii.gz'
            self.nifti_dir.mkdir(exist_ok=True)
            print(f'Saving {nifti_file}')
            nib.save(nifti, nifti_file)

    def normalize(self, loc, scale):
        self.array = (self.array - loc) / scale

    def copy(self):
        copy = type(self)(self.data_root, self.case_name, self.phases)
        copy.array = self.array
        copy.shape = self.shape
        copy.resolution = self.resolution
        return copy
        
    def describe(self):
        return self.array.to_dataframe().describe().T
    
    def select(self, *args, **kwargs):
        selection = self.copy()
        selection.array = self.array.sel(*args, **kwargs, method='nearest')
        return selection
        
    def view(self, *args, **kwargs):
        return view_array(self.array, *args, **kwargs)


def view_array(array, *args, **kwargs):
    if ('x' in kwargs and 'y' in kwargs): # view image
        median = array.quantile(0.5)
        IQR = array.quantile(0.75) - array.quantile(0.25)
        image_kws = {
            'cmap': 'greys_r',
            'clim': (0, median + 1.5 * IQR),
            'frame_width': 500,
            'data_aspect': 1
        }
        image_kws.update(**kwargs)
        kwargs = image_kws

    return array.hvplot(*args, **kwargs)


def as_list(obj):
    return obj if isinstance(obj, list) else [obj]


def load_yaml_file(yaml_file):
    '''
    Read a YAML configuration file.
    '''
    print(f'Loading {yaml_file}')
    with open(yaml_file) as f:
        return yaml.safe_load(f)


def load_xyz_file(xyz_file):
    '''
    Read landmark xyz coordinates from text file.
    '''
    with open(xyz_file) as f:
        data = [line.strip().split() for line in f]
    return np.array(data, dtype=np.uint8)


def load_img_file(img_file, shape, dtype, verbose=True):
    '''
    Read CT image from file in Analyze 7.5 format.
    
    https://stackoverflow.com/questions/27507928/loading-analyze-7-5-format-images-in-python
    '''
    if verbose:
        print(f'Loading {img_file}')
    data = np.fromfile(img_file, dtype)
    data = data.reshape(shape)
    itemsize = data.dtype.itemsize
    data.strides = (
        itemsize,
        itemsize * shape[0],
        itemsize * shape[0] * shape[1]
    )
    return data.copy()
