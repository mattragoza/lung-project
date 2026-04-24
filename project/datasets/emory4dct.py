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
        downloads/
            Case<case_num>(Pack|Deploy).zip
        extracted/
            Case<case_num>(Pack|Deploy)/
                Images/
                    case<case_num>_T<state><img_suffix>.img
                Sampled4D/
                    case<case_num>_4D-75_T<state>.txt
                (E|e)xtremePhases/
                    Case<case_num>_300_T<state>_xyz.txt OR
                    case<case_num>_dirLab300_T<state>_xyz.txt
        processed/
            <variant>/
                <subject>/
                    images/<image_name>.nii.gz
                    fields/<field_name>.nii.gz
                    masks/<mask_name>.nii.gz
                    meshes/<mesh_name>.xdmf
        metadata/
            cases.csv
            phases.csv

    <subject> = case<case_num>

    LEGACY LAYOUT:
        <subject>/
            TotalSeg/
                <image_name>/
                    <mask_name>.nii.gz
            pygalmesh/
                <image_name>_<mask_name>_<mesh_tag>.xdmf
            Corrfield/
                case<sid>_<fixed_state>_<moving_state>.nii.gz
    '''
    SUBJ_ID_COLUMN = 'case_id'
    STATE_ID_COLUMN = 'phase_name'
    EI_STATE = 'T00'
    EE_STATE = 'T50'

    def load_metadata(self):
        import pandas as pd
        self._states = pd.read_csv(self.root / 'metadata' / 'phases.csv')
        self._states.set_index(self.STATE_ID_COLUMN, inplace=True)
        self._metadata = pd.read_csv(self.root / 'metadata' / 'cases.csv')
        self._metadata['img_suffix'] = self._metadata['img_suffix'].fillna('')
        self._metadata.set_index(self.SUBJ_ID_COLUMN, inplace=True)
        self._metadata_loaded = True

    def subject_metadata(self, subject: str):
        self.require_metadata()
        return self._metadata.loc[subject]

    def subjects(self) -> List[str]:
        self.require_metadata()
        return self._metadata.index.to_list()

    def state_metadata(self, state: int):
        self.require_metadata()
        return self._states.loc[state]

    def states(self) -> List[str]:
        self.require_metadata()
        return self._states.index.to_list()

    def image_states(self) -> List[str]:
        self.require_metadata()
        return self._states[self._states.has_image].index.to_list()

    def sampled_states(self) -> List[str]:
        self.require_metadata()
        return self._states[self._states.has_sampled].index.tolist()

    def extreme_states(self) -> List[str]:
        self.require_metadata()
        return self._states[self._states.has_extreme].index.to_list()

    def state_pairs(self) -> Iterable[Tuple[str, str]]:
        for a in self.image_states():
            for b in self.image_states():
                if a != b:
                    yield (a, b)

    def source_path(self, subject: str, state: str, asset_type: str):
        meta = self.subject_metadata(subject)
        base_dir = self.root / 'extracted' / meta.case_dir
        case_num = meta.case_num

        if asset_type == 'image':
            return base_dir / 'Images' / f'case{case_num}_{state}{meta.img_suffix}.img'
        elif asset_type == 'sampled_feats':
            return base_dir / 'Sampled4D' / f'case{case_num}_4D-75_{state}.txt'
        elif asset_type == 'extreme_feats':
            if case_num in {1, 2, 3, 4, 5}:
                return base_dir / 'ExtremePhases' / f'Case{case_num}_300_{state}_xyz.txt'
            else:
                return base_dir / 'extremePhases' / f'case{case_num}_dirLab300_{state}_xyz.txt'

        raise RuntimeError(f'Invalid source asset type: {asset_type!r}')

    def derived_path(
        self,
        subject: str,
        variant: str,
        asset_type: str,
        asset_name: str
    ):
        base_dir = self.root / 'processed' / variant / subject

        if asset_type == 'image':
            return base_dir / 'images' / f'{asset_name}.nii.gz'
        elif asset_type == 'mask':
            return base_dir / 'masks' / f'{asset_name}.nii.gz'
        elif asset_type == 'mask_dir':
            return base_dir / 'masks' / f'{asset_name}'
        elif asset_type == 'field':
            return base_dir / 'fields' / f'{asset_name}.nii.gz'
        elif asset_type == 'mesh':
            return base_dir / 'meshes' / f'{asset_name}.xdmf'

        raise RuntimeError(f'Invalid derived asset type: {asset_type!r}')

    def examples(
        self,
        subjects: Optional[List[str]] = None,
        variant: Optional[str] = None,
        state_pairs: Optional[List[Tuple[str, str]]] = None,
        selectors: Dict[str, str] = None
    ):
        subject_list = list(subjects or self.subjects())
        state_pairs = list(state_pairs or self.state_pairs())

        if variant is not None:
            variant = str(variant)

        selectors = selectors or {}
        img_tag = selectors.get('image_resampling', 'std')
        seg_tag = selectors.get('image_segmentation', 'tseg')
        reg_tag = selectors.get('image_registration', 'corr')
        map_tag = selectors.get('region_labeling', 'regions')
        gen_tag = selectors.get('mesh_generation', 'pyg')
        int_tag = selectors.get('mesh_interpolation', 'int')

        for sid in subject_list:
            m = self.subject_metadata(sid)

            for init_state, curr_state in state_pairs:
                ref_state = self.EI_STATE

                meta = {'raw': dict(m)}
                meta['init_state'] = init_state
                meta['curr_state'] = curr_state
                meta['image_params'] = {
                    'shape': (m.shape_x, m.shape_y, m.shape_z),
                    'dtype': 'h',
                    'system': 'LPI',
                    'spacing': (m.spacing_x, m.spacing_y, m.spacing_z),
                    'slope': 1.0,
                    'intercept': -1024.
                }
                paths = {}
                paths['ref_source'] = self.source_path(sid, ref_state, asset_type='image')
                paths['init_source'] = self.source_path(sid, init_state, asset_type='image')
                paths['curr_source'] = self.source_path(sid, curr_state, asset_type='image')

                if variant:
                    paths['ref_nifti'] = self.derived_path(sid, variant, 'image', f'{sid}_{ref_state}')
                    paths['init_nifti'] = self.derived_path(sid, variant, 'image', f'{sid}_{init_state}')
                    paths['curr_nifti'] = self.derived_path(sid, variant, 'image', f'{sid}_{curr_state}')

                    paths['init_resample'] = self.derived_path(sid, variant, 'image', f'{sid}_{init_state}_{img_tag}')
                    paths['curr_resample'] = self.derived_path(sid, variant, 'image', f'{sid}_{curr_state}_{img_tag}')

                    paths['segment_dir']  = self.derived_path(sid, variant, 'mask_dir', f'{sid}_{init_state}_{img_tag}_{seg_tag}')
                    paths['segment_mask'] = self.derived_path(sid, variant, 'mask', f'{sid}_{init_state}_{img_tag}_{seg_tag}_combined')
                    paths['region_map']   = self.derived_path(sid, variant, 'mask',  f'{sid}_{init_state}_{img_tag}_{seg_tag}_{map_tag}')

                    #paths['input_image']  = self.derived_path(sid, variant, 'image', f'{sid}_{init_state}_{image_tag}')
                    #paths['disp_field']  = self.derived_path(sid, variant, 'field', f'{sid}_{init_state}_{curr_state}_{image_tag}_{mask_tag}')
                    #paths['interp_mesh'] = self.derived_path(sid, variant, 'mesh',  f'{sid}_{init_state}_{curr_state}_{image_tag}_{mask_tag}_{mesh_tag}')

                yield base.Example(
                    dataset='Emory-4DCT',
                    variant=variant,
                    subject=sid,
                    paths=paths,
                    metadata=meta
                )

