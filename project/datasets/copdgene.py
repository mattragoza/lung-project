from typing import Optional, Any, Dict, List, Tuple, Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from . import base


class COPDGeneDataset(base.BaseDataset):
    '''
    <data_root>/Images/<subject>/<visit>/
        <variant>/
            <image_name>.nii.gz
            TotalSegmentator/
                <image_name>/
                    <mask_name>.nii.gz
            pygalmesh/
                <image_name>/
                    <mask_name>_<mesh_tag>.xdmf
            CorrField/
                <fixed_name>__<moving_name>.nii.gz

    <image_name> = <subject>_<state>_<recon>_<site>_COPD
    '''
    VALID_STATES = ['EXP', 'INSP']
    DEFAULT_VISIT = 'Phase-1'
    DEFAULT_RECON = 'STD'

    def __init__(self, data_root: str | Path):
        self.root = Path(data_root)
        self._site_cache = {}

    def subjects(self) -> List[str]:
        image_root = self.root / 'Images'
        if image_root.is_dir():
            return sorted([p.name for p in image_root.iterdir()])
        return []
    
    def visits(self, subject: str) -> List[str]:
        subject_dir = self.root / 'Images' / subject
        if subject_dir.is_dir():
            return sorted([p.name for p in subject_dir.iterdir()])
        return []

    def states(self, subject: str, visit: str) -> List[str]:
        return list(self.VALID_STATES)

    def variants(self, subject: str, visit: str) -> List[str]:
        visit_dir = self.root / 'Images' / subject / visit
        if visit_dir.is_dir():
            return sorted([p.name for p in visit_dir.iterdir() if p.is_dir()])
        return []

    def get_path(
        self,
        subject: str,
        visit: str,
        variant: str,
        state: Optional[str] = None,
        recon: Optional[str] = None,
        asset_type: str = 'image',
        **selectors
    ):
        base_dir = self.root / 'Images' / subject / visit / variant

        if asset_type != 'disp':
            assert state in self.VALID_STATES, str(state)

        recon = recon or self.DEFAULT_RECON
        site = self._infer_site_code(subject, visit)

        def image_name(st: str) -> str:
            return f'{subject}_{st}_{recon}_{site}_COPD'

        if asset_type == 'image':
            return base_dir / f'{image_name(state)}.nii.gz'

        elif asset_type == 'mask':
            mask_name = selectors['mask_name']
            return base_dir / 'TotalSegmentator' / image_name(state) / f'{mask_name}.nii.gz'

        elif asset_type == 'mesh':
            mask_name = selectors['mask_name']
            mesh_tag = selectors['mesh_tag']
            return base_dir / 'pygalmesh' / image_name(state) / f'{mask_name}_{mesh_tag}.xdmf'

        elif asset_type == 'disp':
            fix = selectors['fixed_state']
            mov = selectors['moving_state']
            assert fix in self.VALID_STATES, fix
            assert mov in self.VALID_STATES, mov
            return base_dir / 'CorrField' / f'{image_name(fix)}__{image_name(mov)}.nii.gz'

        raise RuntimeError(f'unrecognized asset type: {asset_type}')

    def _infer_site_code(self, subject, visit):
        key = (subject, visit)
        if key in self._site_cache:
            return self._site_cache[key]

        visit_dir = self.root / 'Images' / subject / visit
        for variant in self.variants(subject, visit):
            variant_dir = visit_dir / variant
            for nii_file in sorted(variant_dir.glob('*.nii.gz')):
                name = nii_file.name[:-7]
                parts = name.split('_')
                # expected format: <subject>_<state>_<recon>_<site>_COPD
                if len(parts) == 5 and parts[0] == subject and parts[1] in self.VALID_STATES:
                    self._site_cache[key] = parts[3]
                    return self._site_cache[key]

        raise RuntimeError(f'failed to infer site code: {subject} {visit}')

    def examples(
        self,
        subjects: Optional[List[str]] = None,
        visit: str = None,
        variant: str = 'ISO',
        state_pairs: Optional[List[Tuple[str, str]]] = None,
        recon: str = None,
        mask_name: str = 'lung_regions',
        mesh_tag: str = 'volume',
        source_variant: str = 'RAW',
        ref_state: str = 'EXP'
    ):
        subjects = subjects or self.subjects()
        visit = visit or self.DEFAULT_VISIT
        recon = recon or self.DEFAULT_RECON
        for subj in subjects:
            pairs = state_pairs or self.state_pairs(subj, visit)
            for fixed, moving in pairs:
                paths = {}
                paths['ref_image'] = self.get_path(subj, visit, source_variant, ref_state, recon, 'image')
                paths['fixed_source'] = self.get_path(subj, visit, source_variant, fixed, recon, 'image')
                paths['moving_source'] = self.get_path(subj, visit, source_variant, moving, recon, 'image')
                paths['fixed_image'] = self.get_path(subj, visit, variant, fixed, recon, 'image')
                paths['moving_image'] = self.get_path(subj, visit, variant, moving, recon, 'image')
                paths['fixed_mask'] = self.get_path(subj, visit, variant, fixed, recon, 'mask', mask_name=mask_name)
                paths['fixed_mesh'] = self.get_path(subj, visit, variant, fixed, recon, 'mesh', mask_name=mask_name, mesh_tag=mesh_tag)
                paths['disp_field'] = self.get_path(subj, visit, variant, None, recon, 'disp', fixed_state=fixed, moving_state=moving)
                yield base.Example(
                    dataset='COPDGene',
                    subject=subj,
                    visit=visit,
                    variant=variant,
                    fixed_state=fixed,
                    moving_state=moving,
                    paths=paths,
                    metadata={'recon': recon}
                )


class COPDGeneVisit: # DEPRECATED

    def __init__(self, data_root, subject_id, visit_name, site_code=None):
        self.data_root  = pathlib.Path(data_root)
        self.subject_id = subject_id
        self.visit_name = visit_name
        self.site_code  = site_code

    # Path helpers

    @property
    def visit_dir(self):
        return self.data_root / 'Images' / self.subject_id / self.visit_name

    def get_image_dir(self, variant):
        return self.visit_dir / variant

    def get_image_path(self, variant, image_name, ext='.nii.gz'):
        return self.get_image_dir(variant) / (image_name + ext)

    def get_mask_root(self, variant):
        return self.get_image_dir(variant) / 'TotalSegmentator'

    def get_mask_dir(self, variant, image_name):
        return self.get_mask_root(variant) / image_name

    def get_mask_path(self, variant, image_name, mask_name, ext='.nii.gz'):
        return self.get_mask_dir(variant, image_name) / (mask_name + ext)

    def get_mesh_root(self, variant):
        return self.get_image_dir(variant) / 'pygalmesh'

    def get_mesh_dir(self, variant, image_name):
        return self.get_mesh_root(variant) / image_name

    def get_mesh_path(self, variant, image_name, mask_name, mesh_tag, ext='.xdmf'):
        return self.get_mesh_dir(variant, image_name) / f'{mask_name}_{mesh_tag}{ext}'

    def get_disp_root(self, variant):
        return self.get_image_dir(variant) / 'CorrField'

    def get_disp_path(self, variant, target_name, source_name):
        return self.get_disp_root(variant) / (target_name + '__' + source_name + '.nii.gz')

    # Directory exploration

    def list_variants(self):
        return sorted([x.name for x in self.visit_dir.iterdir() if x.is_dir()])

    def list_images(self, variant, ext='.nii.gz'):
        image_dir = self.get_image_dir(variant)
        assert image_dir.is_dir(), f'Directory does not exist: {image_dir}'
        return sorted([x.name[:-len(ext)] for x in image_dir.glob('*' + ext)])

    # Image name parsing and validation

    def parse_image_name(self, image_name, validate=True):
        parts = image_name.split('_')
        parsed = {
            'subject_id': parts[0],
            'state': parts[1],
            'recon': parts[2],
            'site_code': parts[3],
            'condition': parts[4],
        }
        if validate:
            assert parsed['subject_id'] == self.subject_id, (parsed['subject_id'], self.subject_id)
            assert parsed['state'] in {'EXP', 'INSP'}, parsed['state']
            assert parsed['recon'] in {'STD', 'SHARP', 'B35f', 'LUNG'}, parsed['recon']
            assert parsed['site_code'] == self.site_code, (parsed['site_code'], self.site_code)
            assert parsed['condition'] in {'COPD'}, parsed['condition']
        return parsed

    def get_image_name(self, state, recon):
        return f'{self.subject_id}_{state}_{recon}_{self.site_code}_COPD'

    def has_valid_image_pair(self, variant, recon, ext='.nii.gz'):
        for state in ['INSP', 'EXP']:
            image_name = self.get_image_name(state, recon)
            image_path = self.get_image_path(variant, image_name, ext)
            if not image_path.is_file():
                print(f'File does not exist: {image_path}')
                return False
            try: # check if loadable
                nib.load(image_path)
            except Exception:
                print(f'Failed to load file: {image_path}')
                return False
        return True

    # Metadata loaders

    def load_metadata_from_filenames(self, variant, filters=None):
        rows = []
        for image_name in self.list_images(variant):
            parsed = self.parse_image_name(image_name)
            if filters and any(parsed.get(k) != filters[k] for k in filters):
                continue
            parsed.update({
                'subject_id': self.subject_id,
                'visit_name': self.visit_name,
                'site_code': self.site_code,
                'variant': variant,
                'image_name': image_name
            })
            rows.append(parsed)
        return pd.DataFrame(rows)

    def load_metadata_from_headers(self, variant, filters=None):
        rows = []
        for t in self.load_metadata_from_filenames(variant, filters).itertuples():
            image_path = self.get_image_path(variant, t.image_name)
            image = nib.load(image_path)
            row = t._asdict()
            row.update({
                'shape': image.header.get_data_shape(),
                'resolution': image.header.get_zooms(),
            })
            rows.append(row)
        return pd.DataFrame(rows)

    # Image loaders

    def load_image(self, variant, state, recon='STD'):
        image_name = self.get_image_name(state, recon)
        image_path = self.get_image_path(variant, image_name)
        return nib.load(image_path)

    def load_images(self, variant, states=['EXP', 'INSP'], recon='STD'):
        images = {}
        for state in states:
            images[state] = self.load_image(variant, state, recon)
        return images

    def load_mask(self, variant, state, mask_name, recon='STD'):
        image_name = self.get_image_name(state, recon)
        mask_path = self.get_mask_path(variant, image_name, mask_name)
        return nib.load(mask_path)

    def load_masks(self, variant, state, mask_list, recon='STD'):
        image_name = self.get_image_name(state, recon)
        image_path = self.get_image_path(variant, image_name)

        image = nib.load(image_path)
        resolution = image.header.get_zooms()

        masks = {}
        for mask_name in mask_list:
            mask_path = self.get_mask_path(variant, image_name, mask_name)
            mask = nib.load(mask_path)
            assert mask.shape == image.shape
            masks[mask_name] = mask

        return masks

    def load_mesh(self, variant, state, mask_name, mesh_tag, recon='STD'):
        image_name = self.get_image_name(state, recon)
        mesh_path = self.get_mesh_path(variant, image_name, mask_name, mesh_tag)
        return meshio.read(mesh_path)

    def load_mesh_with_fenics(self, variant, state, mask_name, mesh_tag, recon='STD'):
        image_name = self.get_image_name(state, recon)
        mesh_path = self.get_mesh_path(variant, image_name, mask_name, mesh_tag)
        return meshing.load_mesh_fenics(mesh_path)

    def load_displacement_field(self, variant, source_state, target_state, recon='STD'):
        source_name = self.get_image_name(source_state, recon)
        target_name = self.get_image_name(target_state, recon)
        disp_path = self.get_disp_path(variant, target_name, source_name)
        return nib.load(disp_path)

    # Preprocessing operations

    def resample_images(
        self,
        input_variant='RAW',
        output_variant='ISO',
        states=['EXP', 'INSP'],
        ref_state='EXP',
        recon='STD',
        **kwargs
    ):
        from ..preprocess.api import resample_image_using_reference
        ref_name = self.get_image_name(ref_state, recon)
        ref_path = self.get_image_path(input_variant, ref_name)
        for i, state in enumerate(states):
            image_name = self.get_image_name(state, recon)
            input_path = self.get_image_path(input_variant, image_name)
            output_path = self.get_image_path(output_variant, image_name)
            resample_image_using_reference(
                input_path, output_path, ref_path, **kwargs
            )

    def create_segmentation_masks(
        self,
        variant='ISO',
        state='EXP',
        recon='STD',
        combined_mask_name='lung_combined_mask'
    ):
        from ..preprocess.api import create_segmentation_masks
        image_name = self.get_image_name(state, recon)
        image_path = self.get_image_path(variant, image_name)
        mask_dir = self.get_mask_dir(variant, image_name)
        combined_path = self.get_mask_path(variant, image_name, mask_name=combined_mask_name)
        create_segmentation_masks(image_path, mask_dir, combined_path)

    def create_multi_region_mask(
        self,
        variant='ISO',
        state='EXP',
        recon='STD',
        mask_name='lung_regions',
        **kwargs
    ):
        from ..preprocess.api import create_multi_region_mask
        image_name = self.get_image_name(state, recon)
        input_paths = {
            roi: self.get_mask_path(variant, image_name, roi)
                for roi in segmentation.ALL_TASK_ROIS
        }
        output_path = self.get_mask_path(variant, image_name, mask_name)
        create_multi_region_mask(input_paths, output_path, **kwargs)

    def create_anatomical_meshes(
        self,
        variant='ISO',
        state='EXP',
        recon='STD',
        mask_name='lung_regions',
        volume_tag='volume',
        surface_tag='surface',
        **kwargs
    ):
        from ..preprocess.api import create_anatomical_meshes
        image_name = self.get_image_name(state, recon)
        mask_path = self.get_mask_path(variant, image_name, mask_name)
        mesh_dir = self.get_mesh_dir(variant, image_name)
        mesh_dir.mkdir(parents=True, exist_ok=True)
        volume_path = self.get_mesh_path(variant, image_name, mask_name, mesh_tag=volume_tag)
        surface_path = self.get_mesh_path(variant, image_name, mask_name, mesh_tag=surface_tag)
        create_anatomical_meshes(mask_path, volume_path, surface_path, **kwargs)

    def create_displacement_field(
        self,
        variant='ISO',
        source_state='INSP',
        target_state='EXP',
        recon='STD',
        mask_name='lung_combined_mask'
    ):
        from ..preprocess.api import create_corrfield_displacement
        source_name = self.get_image_name(source_state, recon)
        target_name = self.get_image_name(target_state, recon)
        source_path = self.get_image_path(variant, source_name)
        target_path = self.get_image_path(variant, target_name)
        mask_path = self.get_mask_path(variant, target_name, mask_name)
        disp_path = self.get_disp_path(variant, target_name, source_name)
        disp_path.parent.mkdir(parents=True, exist_ok=True)
        create_corrfield_displacement(
            source_path, target_path, mask_path, disp_path
        )

