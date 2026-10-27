# datasets/bmc4dct.py

from typing import List, Dict, Tuple, Optional, Any

from pathlib import Path

from .base import Dataset, Example


def _parse_series_desc(s: str) -> Tuple[str, bool]:
    import re
    state, extended = None, False
    for part in s.split(','):
        part = part.strip()
        if part == 'Extended FOV':
            extended = True
        match = re.search(r'(\d+%)')
        if match:
            state = match.group(2)
    return state, extended


class BMC4DCTDataset(Dataset):
    '''
    <data_root>/
        downloads/
            2026-03-25_batch/
                4DCT.zip
                Prior.zip
                Prior2.zip
                Prior3.zip
                Next1.zip
                Next2.zip
        extracted/
            2026-03-25_batch/
                4DCT/
                    4D00<subject_id>_<study_id>/
                        image<slice_index>.dcm
                Prior/
                    Prior2_<subject_id>_<study_id>/
                        image<slice_index>.dcm
                Prior2/
                    Prior2_<subject_id>_<study_id>/
                        image<slice_index>.dcm
                Prior3/
                    Prior2_<subject_id>_<study_id>/
                        image<slice_index>.dcm
                Next1/
                    Next_001_01/DICOM/...
                    Next_<subject_id>_<study_id>/
                        image<slice_index>.dcm
                Next2/
                    Next_<subject_id>_<study_id>/
                        image<slice_index>.dcm
        converted/
            2026-03-25_batch/
                4DCT/
                    4D001_1/
                        4D001_1_%p_%e_%s.nii.gz
                        4D001_1_%p_%e_%s.json
        processed/
            <variant>/
                <subject>/
        metadata/
            batches.csv
    '''
    INDEX_COLUMNS = ('case_id', 'resp_phase')
    EI_RESP_STATE = 0
    EE_RESP_STATE = 50
    DEFAULT_STUDY_NUM = 1

    def load_metadata(self):
        import pandas as pd
        self._metadata = pd.read_csv(self.root / 'metadata' / 'phase_manifest.csv')
        self._metadata.set_index(list(self.INDEX_COLUMNS), inplace=True)
        self._metadata_loaded = True

    def subjects(self):
        self.require_metadata()
        return self._metadata.index.get_level_values('case_id').unique().tolist()

    def source_path(self, subject, state, asset_type='image') -> Path:
        self.require_metadata()

        meta = self._metadata.loc[(subject, state)]
        if len(meta) > 1:
            raise ValueError(f'Multiple rows for subject/state:\n{meta})')

        row = meta.iloc[0]

        if asset_type == 'image':
            return self.root / row.image_path
        elif asset_type == 'json':
            return self.root / row.json_path

        raise RuntimeError(f'Invalid source asset type: {asset_type!r}')

    def derived_path(
        self,
        subject: int,
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
        elif asset_type == 'mesh':
            return base_dir / 'meshes' /  f'{asset_name}.xdmf'
        elif asset_type == 'field':
            return base_dir / 'fields' / f'{asset_name}.nii.gz'

        raise RuntimeError(f'Invalid derived asset type: {asset_type!r}')

    def examples(
        self,
        subjects: Optional[str|Path|List[str]] = None,
        variant:  Optional[str] = None,
        studies:  Optional[str] = None,
        state_pairs: Optional[List[Tuple[str, str]]] = None,
        pipeline_tags: Dict[str, str] = None
    ):
        subject_list = list(subjects or self.subjects())

        if variant is not None:
            variant = str(variant)

        state_pairs = list(state_pairs or self.state_pairs())

        pipeline_tags = pipeline_tags or {}

        valid_tags = {
            'image_resampling',
            'image_segmentation',
            'image_registration',
            'anatomical_regions',
            'mesh_generation',
            'field_interpolation',
        }
        invalid_tags = set(pipeline_tags.keys()) - valid_tags
        if invalid_tags:
            raise ValueError(f'Invalid pipeline tags: {invalid_tags} vs. {valid_tags}')

        img_tag = pipeline_tags.get('image_resampling',    'std')
        seg_tag = pipeline_tags.get('image_segmentation',  'seg')
        reg_tag = pipeline_tags.get('image_registration',  'reg')
        map_tag = pipeline_tags.get('anatomical_regions', 'anat')
        gen_tag = pipeline_tags.get('mesh_generation',     'gen')
        int_tag = pipeline_tags.get('field_interpolation', 'int')

        for sid in subject_list:

            for init_state, curr_state in state_pairs:
                ref_state = self.EI_RESP_STATE

                meta = {}
                meta['ref_state'] = ref_state
                meta['init_state'] = init_state
                meta['curr_state'] = curr_state

                paths = {'ref_state': {}, 'init_state': {}, 'curr_state': {}}
                paths['ref_state']['source_image'] = self.source_path(sid, ref_state, asset_type='image')
                paths['init_state']['source_image'] = self.source_path(sid, init_state, asset_type='image')
                paths['curr_state']['source_image'] = self.source_path(sid, curr_state, asset_type='image')

                if variant:
                    paths['init_state']['resampled_image'] = self.derived_path(sid, variant, 'image', f'{sid}_{init_state}_{img_tag}')
                    paths['curr_state']['resampled_image'] = self.derived_path(sid, variant, 'image', f'{sid}_{curr_state}_{img_tag}')

                    paths['init_state']['segment_dir'] = self.derived_path(sid, variant, 'mask_dir', f'{sid}_{init_state}_{img_tag}_{seg_tag}')
                    paths['curr_state']['segment_dir'] = self.derived_path(sid, variant, 'mask_dir', f'{sid}_{curr_state}_{img_tag}_{seg_tag}')

                    paths['init_state']['combined_mask'] = self.derived_path(sid, variant, 'mask', f'{sid}_{init_state}_{img_tag}_{seg_tag}_domain')
                    paths['curr_state']['combined_mask'] = self.derived_path(sid, variant, 'mask', f'{sid}_{curr_state}_{img_tag}_{seg_tag}_domain')

                    paths['anatomical_map'] = self.derived_path(sid, variant, 'mask',  f'{sid}_{init_state}_{img_tag}_{seg_tag}_{map_tag}')
                    paths['anatomical_mesh'] = self.derived_path(sid, variant, 'mesh', f'{sid}_{init_state}_{img_tag}_{seg_tag}_{map_tag}_{gen_tag}')

                    paths['disp_field'] = self.derived_path(sid, variant, 'field', f'{sid}_{init_state}_{img_tag}_{seg_tag}_{reg_tag}_{curr_state}')
                    paths['interp_mesh'] = self.derived_path(sid, variant, 'mesh', f'{sid}_{init_state}_{img_tag}_{seg_tag}_{reg_tag}_{curr_state}_{map_tag}_{gen_tag}_{int_tag}')

                    # required training paths
                    paths['input_image'] = paths['init_state']['resampled_image']
                    paths['domain_mask'] = paths['init_state']['combined_mask']
                    paths['target_mesh'] = paths['interp_mesh']

                yield Example(
                    dataset='BMC-4DCT',
                    variant=variant,
                    subject=sid,
                    paths=paths,
                    metadata=meta
                )

