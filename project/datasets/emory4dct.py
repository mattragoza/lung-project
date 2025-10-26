from typing import Optional, Any, Dict, List, Tuple, Iterable
from pathlib import Path
import numpy as np

from . import base


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


class Emory4DCTDataset(base.BaseDataset):
    '''
    <data_root>/<subject>/
        Images/
            <image_name>.img
        <variant>/
            <image_name>.nii.gz
            TotalSeg/
                <image_name>/
                    <mask_name>.nii.gz
            pygalmesh/
                <image_name>_<mask_name>_<mesh_tag>.xdmf
            Corrfield/
                case<sid>_<fixed_state>_<moving_state>.nii.gz

    <image_name> = case<sid>_<state>.nii.gz
    '''
    VALID_STATES = [f'T{i:02d}' for i in range(0, 100, 10)]

    def __init__(self, data_root: str|Path):
        self.root = Path(data_root)

    def subjects(self) -> List[str]:
        if self.root.is_dir():
            return sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        return []

    def variants(self, subject: str, visit: str) -> List[str]:
        subject_root = self.root / subject
        if subject_root.is_dir():
            found = []
            for d in sorted(p for p in subject_root.iterdir() if p.is_dir()):
                if d not in {'Sampled4D', 'ExtremePhases'}:
                    found.append(d)
            return found
        return []

    def visits(self, subject: str) -> List[str]:
        return []

    def states(self, subject: str, visit: str) -> List[str]:
        return list(self.VALID_STATES)

    def get_path(
        self,
        subject: str,
        variant: str,
        visit: Optional[str], 
        state: Optional[str] = None,
        asset_type: str = 'image',
        **selectors
    ):
        subject_root = self.root / subject
        variant_dir = self.root / subject / variant
        sid = self._parse_subject_id(subject)

        def image_name(st: str) -> str:
            return f'case{sid}_{st}'

        if asset_type != 'disp':
            assert state in self.VALID_STATES

        if asset_type == 'image':
            if variant == 'Images': # Analyze 7.5 format
                return variant_dir / f'{image_name(state)}.img'
            return variant_dir / f'{image_name(state)}.nii.gz'

        elif asset_type == 'mask':
            mask_name = selectors['mask_name']
            return variant_dir / 'TotalSegment' / image_name(state) / f'{mask_name}.nii.gz'

        elif asset_type == 'mesh':
            mask_name = selectors['mask_name']
            mesh_tag = selectors['mesh_tag']
            return variant_dir / 'pygalmesh' / f'{image_name(state)}_{mask_name}_{mesh_tag}.xdmf'

        elif asset_type == 'disp':
            fix = selectors['fixed_state']
            mov = selectors['moving_state']
            assert fix in self.VALID_STATES
            assert mov in self.VALID_STATES
            return variant_dir / 'CorrField' / f'case{sid}_{moving_state}_{fixed_state}.nii.gz'

        elif asset_type == 'kpts':
            return subject_root / 'Sampled4D' / f'case{sid}_4D-75_{state}.txt'

        raise RuntimeError(f'unrecognized asset type: {asset_type}')

    def _parse_subject_id(subject: str) -> int:
        # expected format: Case<sid>_(Pack|Deploy)
        parts = subject.split('_')
        if len(parts) == 2 and parts[0].startswith('Case'):
            return int(parts[0][4:])

        raise RuntimeError(f'failed to parse subject id: {subject}')

    def examples(
        self,
        subjects: Optional[List[str]] = None,
        variant: str = 'NIFTI',
        state_pairs: Optional[List[Tuple[str, str]]] = None,
        mask_name: str = None,
        mesh_tag: str = None,
        source_variant: str = 'Images',
        ref_state: str = 'T00'
    ):
        subjects = subjects or self.subjects()
        for subj in subjects:
            pairs = state_pairs or self.state_pairs(subj, None)
            for fixed, moving in pairs:
                paths = {}
                paths['ref_image'] = self.get_path(subj, source_variant, None, ref_state, 'image')
                paths['fixed_source'] = self.get_path(subj, source_variant, None, fixed, 'image')
                paths['moving_source'] = self.get_path(subj, source_variant, None, moving, 'image')
                paths['fixed_image'] = self.get_path(subj, variant, None, fixed, 'image')
                paths['moving_image'] = self.get_path(subj, variant, None, moving, 'image')
                paths['fixed_mask'] = self.get_path(subj, variant, None, fixed, 'mask', mask_name=mask_name)
                paths['fixed_mesh'] = self.get_path(subj, variant, None, fixed, 'mesh', mask_name=mask_name, mesh_tag=mesh_tag)
                paths['disp_field'] = self.get_path(subj, variant, None, None, 'disp', fixed_state=fixed, moving_state=moving)
                yield base.Example(
                    dataset='Emory4DCT',
                    subject=subj,
                    variant=variant,
                    visit=None,
                    fixed_state=fixed,
                    moving_state=moving,
                    paths=paths,
                    metadata={}
                )


class Emory4DCT: # DEPRECATED

    def __init__(self, data_root, case_names, phases):
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
        import pandas as pd
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

    def get_examples(
        self, mask_roi='lung_regions', mesh_version=10
    ):
        examples = []
        for case in self.cases:
            for fixed_phase in self.phases:
                moving_phase = (fixed_phase + 10) % 100
                examples.append({
                    'name': case.nifti_file(fixed_phase).stem,
                    'anat_file': case.nifti_file(fixed_phase),
                    'disp_file': case.disp_file(moving_phase, fixed_phase),
                    'mask_file': case.totalseg_mask_file(fixed_phase, mask_roi),
                    'mask_file1': case.medpseg_mask_file(fixed_phase, 'findings'),
                    'mask_file2': case.medpseg_mask_file(fixed_phase, 'consolidation'),
                    'mask_file3': case.medpseg_mask_file(fixed_phase, 'ggo'),
                    'mesh_file': case.mesh_file(fixed_phase, mask_roi, mesh_version),
                    'has_labels': (mesh_version >= 20)
                })
        return examples


class Emory4DCTCase: # DEPRECATED
    
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
    def totalseg_mask_dir(self):
        return self.case_dir / 'TotalSegment'

    @property
    def medpseg_mask_dir(self):
        return self.case_dir / 'medpseg'

    @property
    def corrfield_dir(self):
        return self.case_dir / 'CorrField'

    @property
    def mesh_dir(self):
        return self.case_dir / 'pygalmesh'

    def image_file(self, phase):
        img_glob = self.image_dir.glob(f'case{self.case_id}_T{phase:02d}*.img')
        return next(img_glob) # assume exactly one match

    def nifti_file(self, phase):
        return self.nifti_dir / f'case{self.case_id}_T{phase:02d}.nii.gz'

    def totalseg_mask_file(self, phase, roi):
        return self.totalseg_mask_dir / f'case{self.case_id}_T{phase:02d}/{roi}.nii.gz'

    def medpseg_mask_file(self, phase, roi):
        return self.medpseg_mask_dir / f'case{self.case_id}_T{phase:02d}_{roi}.nii.gz'

    def disp_file(self, moving_phase, fixed_phase):
        return self.corrfield_dir / \
            f'case{self.case_id}_T{moving_phase:02d}_T{fixed_phase:02d}.nii.gz' 

    def mesh_file(self, phase, mask_roi, mesh_version):
        return self.mesh_dir / \
            f'case{self.case_id}_T{phase:02d}_{mask_roi}_{mesh_version}.xdmf'
        
    def load_images(self, shape, resolution, shift=-1000, flip_z=True):

        images = []
        for phase in self.phases:
            img_file = self.image_file(phase)
            image = load_img_file(img_file, shape)
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

    def load_totalseg_masks(self, roi='lung_regions'):
        all_data = []
        for phase in self.phases:
            phase_data = []
            for r in as_iterable(roi):
                mask_file = self.totalseg_mask_file(phase, roi=r)
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

        self.totalseg_mask = xr.DataArray(
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

    def load_medpseg_masks(self, roi):
        all_data = []
        for phase in self.phases:
            phase_data = []
            for r in as_iterable(roi):
                mask_file = self.medpseg_mask_file(phase, roi=r)
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

        self.medpseg_mask = xr.DataArray(
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

    def load_meshes(self, mask_roi, mesh_version):
        self.meshes = []
        for phase in self.phases:
            mesh_file = self.mesh_file(phase, mask_roi, mesh_version)
            mesh = meshing.load_mesh_fenics(mesh_file)
            self.meshes.append(mesh)

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

    def save_totalseg_masks(self, roi):
        for phase in self.phases:
            data = self.totalseg_mask.sel(phase=phase).data
            affine = np.diag(list(self.resolution) + [1])
            nifti = nib.nifti1.Nifti1Image(data, affine)
            mask_file = self.totalseg_mask_file(phase, roi)
            self.totalseg_mask_dir.mkdir(exist_ok=True)
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



