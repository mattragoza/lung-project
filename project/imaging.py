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
        for case_name in as_iterable(case_names):
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

    def load_masks(self, roi):
        for case in self.cases:
            case.load_masks(roi)

    def load_displacements(self, fixed_phase):
        for case in self.cases:
            case.load_displacements(fixed_phase)

    def load_landmarks(self):
        for case in self.cases:
            case.load_landmarks()

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
        self.phases = as_iterable(phases)
        self.phase_index = {p: i for i, p in enumerate(self.phases)}

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
    def landmark_dir(self):
        return self.case_dir / 'Sampled4D'

    @property
    def nifti_dir(self):
        return self.case_dir / 'NIFTI'

    @property
    def mask_dir(self):
        return self.case_dir / 'TotalSegment'

    @property
    def disp_dir(self):
        return self.case_dir / 'CorrField'
        
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
            phase_mask_data = []
            for r in as_iterable(roi):
                mask_file = self.mask_dir / f'case{self.case_id}_T{phase:02d}/{r}.nii.gz'
                print(f'Loading {mask_file}')
                mask = nib.load(mask_file)
                if mask_data:
                    assert mask.header.get_data_shape() == shape
                    assert mask.header.get_zooms() == resolution
                else:
                    shape = mask.header.get_data_shape()
                    resolution = mask.header.get_zooms()
                phase_mask_data.append(mask.get_fdata())
            mask_data.append(np.stack(phase_mask_data, axis=-1))

        assert shape == self.shape, f'{shape} vs. {self.shape}'
        assert np.allclose(resolution, self.resolution), \
            f'{resolution} vs {self.resolution}'

        self.mask = xr.DataArray(
            data=np.stack(mask_data),
            dims=['phase', 'x', 'y', 'z', 'roi'],
            coords={
                'phase': self.phases,
                'x': np.arange(shape[0]) * resolution[0],
                'y': np.arange(shape[1]) * resolution[1],
                'z': np.arange(shape[2]) * resolution[2],
                'roi': as_iterable(roi)
            },
            name='mask'
        )

    def load_displacements(self, fixed_phase):

        disp_data = []
        for moving_phase in self.phases:
            disp_file = self.disp_dir / f'case{self.case_id}_T{moving_phase:02d}_T{fixed_phase:02d}.nii.gz'
            print(f'Loading {disp_file}')
            disp = nib.load(disp_file)
            if disp_data:
                assert disp.header.get_data_shape() == shape
                assert disp.header.get_zooms() == resolution
            else:
                shape = disp.header.get_data_shape()
                resolution = disp.header.get_zooms()
            disp_data.append(disp.get_fdata())

        expected_shape = (1,) + self.shape + (3,)
        assert shape == expected_shape, f'{shape} vs. {expected_shape}'
        assert np.allclose(resolution, 1), resolution

        self.disp = xr.DataArray(
            data=np.concatenate(disp_data),
            dims=['phase', 'x', 'y', 'z', 'component'],
            coords={
                'phase': self.phases,
                'x': np.arange(self.shape[0]) * self.resolution[0],
                'y': np.arange(self.shape[1]) * self.resolution[1],
                'z': np.arange(self.shape[2]) * self.resolution[2],
                'component': ['x', 'y', 'z']
            },
            name='displacement'
        )

    def load_landmarks(self):
        self.landmarks = []
        for phase in self.phases:
            if phase > 50:
                self.landmarks.append(None)
                continue
            try:
                txt_pattern = f'case{self.case_id}_*_T{phase:02d}*.txt'
                txt_glob = self.landmark_dir.glob(txt_pattern)
                txt_file = next(txt_glob)
            except StopIteration:
                txt_pattern = f'Case{self.case_id}_*_T{phase:02d}*.txt'
                txt_glob = self.landmark_dir.glob(txt_pattern)
                txt_file = next(txt_glob)

            points = load_xyz_file(txt_file)
            points[:,2] = (self.shape[2] - 1 - points[:,2]) # flip z index
            self.landmarks.append(points)

    def get_landmark_points(self, phase):
        index = self.phase_index[phase]
        return self.landmarks[index] * self.resolution

    def save_niftis(self):
        for phase in self.phases:
            data = self.array.sel(phase=phase).data
            affine = np.diag(list(self.resolution) + [1])
            nifti = nib.nifti1.Nifti1Image(data, affine)
            nifti_file = self.nifti_dir / f'case{self.case_id}_T{phase:02d}.nii.gz'
            self.nifti_dir.mkdir(exist_ok=True)
            print(f'Saving {nifti_file}')
            nib.save(nifti, nifti_file)

    def register(self, fixed_phase):
        import torch, corrfield
        import torch.nn.functional as F

        disp_data = []
        for moving_phase in self.phases:

            img_fix = torch.from_numpy(
                self.array.sel(phase=fixed_phase).data.copy()
            ).unsqueeze(0).unsqueeze(0).to('cuda', torch.float32)
            img_mov = torch.from_numpy(
                self.array.sel(phase=moving_phase).data.copy()
            ).unsqueeze(0).unsqueeze(0).to('cuda', torch.float32)
            mask_fix = torch.from_numpy(
                self.mask.sel(phase=fixed_phase).data.copy()
            ).unsqueeze(0).unsqueeze(0).to('cuda', torch.float32)

            disp, kpts_fix, kpts_mov = corrfield.corrfield.corrfield(
                img_fix, mask_fix, img_mov
            )
            if disp_data:
                assert disp.shape == shape
            else:
                shape = disp.shape

            disp_data.append(disp.cpu().numpy())

        expected_shape = (1,) + self.shape + (3,)
        assert shape == expected_shape, f'{shape} vs. {expected_shape}'

        self.disp = xr.DataArray(
            data=np.concatenate(disp_data),
            dims=['phase', 'x', 'y', 'z', 'component'],
            coords={
                'phase': self.phases,
                'x': np.arange(self.shape[0]) * self.resolution[0],
                'y': np.arange(self.shape[1]) * self.resolution[1],
                'z': np.arange(self.shape[2]) * self.resolution[2],
                'component': ['x', 'y', 'z']
            },
            name='displacement'
        )

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


def is_iterable(obj):
    return hasattr(obj, '__iter__') and not isinstance(obj, str)


def as_iterable(obj):
    return obj if is_iterable(obj) else [obj]


def load_yaml_file(yaml_file):
    '''
    Read a YAML configuration file.
    '''
    print(f'Loading {yaml_file}')
    with open(yaml_file) as f:
        return yaml.safe_load(f)


def load_xyz_file(xyz_file, dtype=float):
    '''
    Read landmark xyz coordinates from text file.
    '''
    print(f'Loading {xyz_file}')
    with open(xyz_file) as f:
        data = [line.strip().split() for line in f]
    return np.array(data, dtype=dtype)


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
