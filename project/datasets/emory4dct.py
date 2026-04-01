from typing import Optional, Any, Dict, List, Tuple, Iterable
from pathlib import Path
import numpy as np

from . import base


def _parse_case_number(s: str) -> int:
    import re
    match = re.match(r'Case(\d+)(Pack|Deploy)', s)
    if match:
        return int(match.group(1))
    raise RuntimeError(f'Failed to parse case number: {s!r}')


class Emory4DCTDataset(base.Dataset):
    '''
    <data_root>/
        Case<case_num>(Pack|Deploy)/
            Images/
                case<case_num>_T<state><img_suffix>.img
            Sampled4D/
                case<case_num>_4D-75_T<state>.txt
            (ExtremePhases|extremePhases)/
                Case<case_num>_300_T<state>_xyz.txt OR
                case<case_num>_dirLab300_T<state>_xyz.txt
        <variant>/
            <subject>/
                images/<image_name>.nii.gz
                fields/<field_name>.nii.gz
                masks/<mask_name>.nii.gz
                meshes/<mesh_name>.xdmf

    <subject> = case<case_num>

    PREVIOUS FORMAT:
        <subject>/
            TotalSeg/
                <image_name>/
                    <mask_name>.nii.gz
            pygalmesh/
                <image_name>_<mask_name>_<mesh_tag>.xdmf
            Corrfield/
                case<sid>_<fixed_state>_<moving_state>.nii.gz
    '''
    ID_COLUMN = 'case_id'

    def __init__(self, data_root: str|Path):
        self.root = Path(data_root)
        if not self.root.is_dir():
            raise RuntimeError(f'Invalid directory: {data_root}')
        self._metadata_loaded = False

    def ensure_metadata(self):
        if not self._metadata_loaded:
            self.load_metadata()

    def load_metadata(self):
        import pandas as pd
        self.phases = pd.read_csv(self.root / 'phases.csv')
        self.metadata = pd.read_csv(self.root / 'metadata.csv')
        self.metadata['img_suffix'] = self.metadata['img_suffix'].fillna('')
        self.metadata.set_index(self.ID_COLUMN, inplace=True)
        self._metadata_loaded = True

    def subject_metadata(self, subject: str):
        self.ensure_metadata()
        return self.metadata.loc[subject]

    def subjects(self) -> Iterable[str]:
        self.ensure_metadata()
        return self.metadata.index.to_list()

    def image_phases(self) -> Iterable[str]:
        return self.phases[self.phases.image].phase.to_list()

    def sampled_phases(self) -> Iterable[str]:
        return self.phases[self.phases.sampled_feats].phase.tolist()

    def extreme_phases(self) -> Iterable[str]:
        return self.phases[self.phases.extreme_feats].phase.to_list()

    def phase_pairs(self) -> Iterable[Tuple[str, str]]:
        for f in self.image_phases():
            for m in self.image_phases():
                if f != m:
                    yield (f, m)

    def path(self, subject: str, variant: str, asset_type: str, phase: str) -> Path:
        meta = self.subject_metadata(subject)

        if not variant: # source path
            base_dir = self.root / meta.case_dir

            if asset_type == 'image':
                return base_dir / 'Images' / f'case{meta.case_num}_T{phase:02d}{meta.img_suffix}.img'
            elif asset_type == 'sampled_feats':
                return base_dir / 'Sampled4D' / f'case{meta.case_num}_4D-75_T{phase:02d}.txt'
            elif asset_type == 'extreme_feats':
                if meta.case_num in {1, 2, 3, 4, 5}:
                    return base_dir / 'ExtremePhases' / f'Case{meta.case_num}_300_T{phase:02d}_xyz.txt'
                else:
                    return base_dir / 'extremePhases' / f'case{meta.case_num}_dirLab300_T{phase:02d}_xyz.txt'
            else:
                raise RuntimeError(f'Unrecognized source asset type: {asset_type!r}')

        else: # derived path
            base_dir = self.root / variant / subject

            if asset_type == 'image':
                return base_dir / 'images' / f'{name}.nii.gz'
            elif asset_type == 'mask':
                return base_dir / 'masks' / f'{name}.nii.gz'
            elif asset_type == 'field':
                return base_dir / 'fields' / f'{name}.nii.gz'
            elif asset_type == 'mesh':
                return base_dir / 'meshes' / f'{name}.xdmf'
            else:
                raise RuntimeError(f'Unrecognized derived asset type: {asset_type!r}')

    def examples(
        self,
        subjects: Optional[List[str]] = None,
        variant:  Optional[str] = None,
        phase_pairs: Optional[List[Tuple[str, str]]] = None
    ):
        subject_list = list(subjects or self.subjects())
        phase_pairs = list(phase_pairs or self.phase_pairs())

        if variant is not None:
            variant = str(variant)

        for sid in subject_list:
            for f_phase, m_phase in phase_pairs:
                paths = {}
                paths['fixed_source'] = self.path(sid, None, asset_type='image', phase=f_phase)
                paths['moving_source'] = self.path(sid, None, asset_type='image', phase=m_phase)

                m = self.subject_metadata(sid)
                metadata = {'raw': dict(m)}
                metadata['fixed_phase'] = f_phase
                metadata['moving_phase'] = m_phase
                metadata['source_shape'] = (m.shape_x, m.shape_y, m.shape_z)
                metadata['source_spacing'] = (m.spacing_x, m.spacing_y, m.spacing_z)
                metadata['unit'] = 1e-3 # meters per world unit (mm)

                if variant: # assets generated by preprocessing
                    paths['fixed_nifti'] = self.path(sid, variant, asset_type='image', phase=f_phase)
                    paths['moving_nifti'] = self.path(sid, variant, asset_type='image', phase=m_phase)

                    paths['input_image'] = self.path(sid, variant, asset_type='image')
                    paths['region_mask'] = self.path(sid, variant, asset_type='mask')
                    paths['interp_mesh'] = self.path(sid, variant, asset_type='mesh')
                    paths['disp_field'] = self.path(sid, variant, asset_type='field')

                yield base.Example(
                    dataset='Emory4DCT',
                    variant=variant,
                    subject=sid,
                    paths=paths,
                    metadata=metadata
                )

