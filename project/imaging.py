import sys, os, re, yaml
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import xarray as xr
import hvplot.xarray
from tqdm import tqdm
import torch

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

CASE_METADATA = [ # shape, resolution, num_features
    ((256, 256,  94), (0.97, 0.97, 2.50), 1280),
    ((256, 256, 112), (1.16, 1.16, 2.50), 1487),
    ((256, 256, 104), (1.15, 1.15, 2.50), 1561),
    ((256, 256,  99), (1.13, 1.13, 2.50), 1166),
    ((256, 256, 106), (1.10, 1.10, 2.50), 1268),
    ((512, 512, 128), (0.97, 0.97, 2.50),  419),
    ((512, 512, 136), (0.97, 0.97, 2.50),  398),
    ((512, 512, 128), (0.97, 0.97, 2.50),  476),
    ((512, 512, 128), (0.97, 0.97, 2.50),  342),
    ((512, 512, 120), (0.97, 0.97, 2.50),  435),
]


class Emory4DCT(object):

    def __init__(self, data_root, case_names=ALL_CASES, phases=ALL_PHASES):
        self.data_root = Path(data_root)
        self.case_names = as_iterable(case_names)
        self.phases = as_iterable(phases)
        self.cases = []
        self.case_index = {c: i for i, c in enumerate(self.case_names)}

        for case_name in as_iterable(case_names):
            case = Emory4DCTCase(data_root, case_name, phases)
            self.cases.append(case)

    def __repr__(self):
        class_name = type(self).__name__
        data_root = repr(str(self.data_root))
        n_cases = len(self.cases)
        return f'{class_name}({data_root}, {n_cases} cases)'

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        return self.cases[idx]

    def load_images(self, *args, **kwargs):
        for i, case in enumerate(self.cases):
            case_idx = ALL_CASES.index(case.case_name)
            shape, resolution, _ = CASE_METADATA[case_idx]
            case.load_images(shape, resolution, *args, **kwargs)

    def load_masks(self, roi):
        for case in self.cases:
            case.load_masks(roi)

    def save_masks(self, roi):
        for case in self.cases:
            case.save_masks(roi)

    def load_displacements(self, fixed_phase, relative):
        for case in self.cases:
            case.load_displacements(fixed_phase, relative)

    def load_landmarks(self):
        for case in self.cases:
            case.load_landmarks()

    def describe(self):
        stats = []
        for case in tqdm(self.cases):
            case_stats = case.describe()
            stats.append(case_stats)
        return pd.concat(stats)

    def save_niftis(self):
        for case in self.cases:
            case.save_niftis()

    def load_niftis(self):
        for case in self.cases:
            case.load_niftis()

    def register_all(self, fixed_case=0):
        fixed_case = self.cases[fixed_case]
        for case in self.cases:
            case.register_cases(fixed_case)

    def get_examples(self, mask_roi='lung_combined_mask', mesh_radius=20):
        examples = []
        for case in self.cases:
            for fixed_phase in self.phases:
                moving_phase = (fixed_phase + 10) % 100
                examples.append({
                    'name': case.nifti_file(fixed_phase).stem,
                    'anat_file': case.nifti_file(fixed_phase),
                    'disp_file': case.disp_file(moving_phase, fixed_phase),
                    'mask_file': case.mask_file(fixed_phase, mask_roi),
                    'mesh_file': case.mesh_file(fixed_phase, mask_roi, mesh_radius),
                    'mesh_radius': mesh_radius
                })
        return examples


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

    @property
    def mesh_dir(self):
        return self.case_dir / 'pygalmesh'

    def image_file(self, phase):
        img_glob = self.image_dir.glob(f'case{self.case_id}_T{phase:02d}*.img')
        return next(img_glob) # assume exactly one match

    def nifti_file(self, phase):
        return self.nifti_dir / f'case{self.case_id}_T{phase:02d}.nii.gz'

    def mask_file(self, phase, roi):
        return self.mask_dir  /f'case{self.case_id}_T{phase:02d}/{roi}.nii.gz'

    def disp_file(self, moving_phase, fixed_phase):
        return self.disp_dir / \
            f'case{self.case_id}_T{moving_phase:02d}_T{fixed_phase:02d}.nii.gz' 

    def mesh_file(self, phase, roi, radius):
        return self.mesh_dir / \
            f'case{self.case_id}_T{phase:02d}_{roi}_{radius}.xdmf'
        
    def load_images(self, shape, resolution, shift=-1000, flip_z=True):

        images = []
        for phase in self.phases:
            img_file = self.image_file(phase)
            image = load_img_file(img_file, shape, dtype=np.int16)
            images.append(image)

        # stack images and apply shift
        images = np.stack(images) + shift

        if flip_z: # flip z orientation
            images = images[...,::-1] 

        self.shape = shape
        self.resolution = resolution
        self.anat = xr.DataArray(
            data=images,
            dims=['phase', 'x', 'y', 'z'],
            coords={
                'phase': self.phases,
                'x': np.arange(shape[0]) * resolution[0],
                'y': np.arange(shape[1]) * resolution[1],
                'z': np.arange(shape[2]) * resolution[2]
            },
            name=f'CT'
        )

    def load_niftis(self):

        all_data = []
        for phase in self.phases:
            nifti_file = self.nifti_file(phase)
            print(f'Loading {nifti_file}')
            nifti = nib.load(nifti_file)
            if all_data:
                assert nifti.header.get_data_shape() == shape
                assert nifti.header.get_zooms() == resolution
            else:
                shape = nifti.header.get_data_shape()
                resolution = nifti.header.get_zooms()
            all_data.append(nifti.get_fdata())

        self.shape = shape
        self.resolution = resolution
        self.anat = xr.DataArray(
            data=np.stack(all_data),
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
        all_data = []
        for phase in self.phases:
            phase_data = []
            for r in as_iterable(roi):
                mask_file = self.mask_file(phase, roi=r)
                print(f'Loading {mask_file}')
                mask = nib.load(mask_file)
                if all_data:
                    assert mask.header.get_data_shape() == shape
                    assert mask.header.get_zooms() == resolution
                else:
                    shape = mask.header.get_data_shape()
                    resolution = mask.header.get_zooms()

                phase_data.append(mask.get_fdata())
            all_data.append(np.stack(phase_data))

        assert shape == self.shape, f'{shape} vs. {self.shape}'
        assert np.allclose(resolution, self.resolution), \
            f'{resolution} vs {self.resolution}'

        self.mask = xr.DataArray(
            data=np.stack(all_data),
            dims=['phase', 'roi', 'x', 'y', 'z'],
            coords={
                'phase': self.phases,
                'x': np.arange(shape[0]) * resolution[0],
                'y': np.arange(shape[1]) * resolution[1],
                'z': np.arange(shape[2]) * resolution[2],
                'roi': as_iterable(roi)
            },
            name='mask'
        )

    def load_displacements(self, moving_phase, relative):
        all_data = []
        fixed_phase_coords = self.phases
        for f in fixed_phase_coords:
            fixed_phase_data = []
            moving_phase_coords = []
            for m in as_iterable(moving_phase):
                if relative:
                    m = (f + m) % 100
                disp_file = self.disp_file(moving_phase=m, fixed_phase=f)
                print(f'Loading {disp_file}')
                disp = nib.load(disp_file)
                if all_data:
                    assert disp.header.get_data_shape() == shape
                    assert disp.header.get_zooms() == resolution
                else:
                    shape = disp.header.get_data_shape()
                    resolution = disp.header.get_zooms()

                fixed_phase_data.append(disp.get_fdata())
                moving_phase_coords.append(m)
            all_data.append(np.stack(fixed_phase_data))

        expected_shape = self.shape + (3,)
        expected_resolution = self.resolution + (1.0,)

        assert shape == expected_shape, f'{shape} vs. {expected_shape}'
        assert np.allclose(resolution, expected_resolution), \
            f'{resolution} vs. {expected_resolution}'

        print(np.stack(all_data, axis=0).shape)

        self.disp = xr.DataArray(
            data=np.stack(all_data, axis=0),
            dims=['fixed_phase', 'moving_phase', 'x', 'y', 'z', 'component'],
            coords={
                'fixed_phase': fixed_phase_coords,
                'moving_phase': moving_phase_coords,
                'x': np.arange(self.shape[0]) * self.resolution[0],
                'y': np.arange(self.shape[1]) * self.resolution[1],
                'z': np.arange(self.shape[2]) * self.resolution[2],
                'component': ['x', 'y', 'z'],
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
            data = self.anat.sel(phase=phase).data
            affine = np.diag(list(self.resolution) + [1])
            nifti = nib.nifti1.Nifti1Image(data, affine)
            nifti_file = self.nifti_file(phase)
            self.nifti_dir.mkdir(exist_ok=True)
            print(f'Saving {nifti_file}')
            nib.save(nifti, nifti_file)

    def save_masks(self, roi):
        for phase in self.phases:
            data = self.mask.sel(phase=phase).data
            affine = np.diag(list(self.resolution) + [1])
            nifti = nib.nifti1.Nifti1Image(data, affine)
            mask_file = self.mask_file(phase, roi)
            self.mask_dir.mkdir(exist_ok=True)
            print(f'Saving {mask_file}')
            nib.save(nifti, mask_file)

    def copy(self):
        copy = type(self)(self.data_root, self.case_name, self.phases)
        copy.anat = self.anat
        copy.shape = self.shape
        copy.resolution = self.resolution
        return copy
        
    def describe(self):
        return self.anat.to_dataframe().describe().T
    
    def select(self, *args, **kwargs):
        selection = self.copy()
        selection.anat = self.anat.sel(*args, **kwargs, method='nearest')
        return selection


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
