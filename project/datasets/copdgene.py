from typing import Optional, Any, Dict, List, Tuple, Iterable
from pathlib import Path
from functools import lru_cache

from . import base


def _parse_image_name(name: str):
    # expected format: <subject>_<state>_<recon>_<site>_COPD
    parts = name.split('_')
    if len(parts) == 5 and parts[4] == 'COPD':
        return {
            'subject': parts[0],
            'state': parts[1], 
            'recon': parts[2],
            'site': parts[3],
        }
    raise RuntimeError(f'failed to parse image name: {name}')


class COPDGeneDataset(base.Dataset):
    '''
    <data_root>/
        Images/<subject>/<visit>/
            RAW/
                <image_name>.nii.gz
        <variant>/<subject>/<visit>/
            images/
                <image_name>.nii.gz
            masks/
                <image_name>/
                    <mask_tag>.nii.gz
            meshes/
                <image_name>/
                    <mesh_tag>.xdmf
            disps/
                <image_name>/
                    <fix_state>_to_<mov_state>.nii.gz

    <image_name> = <subject>_<state>_<recon>_<site>_COPD
    '''
    DEFAULT_VISIT = 'Phase-1'
    SOURCE_VARIANT = 'Images'
    DEFAULT_RECON = 'STD'
    VALID_RECONS = ['STD', 'SHARP', 'B35f']
    DEFAULT_STATE = 'EXP'
    VALID_STATES = ['EXP', 'INSP']

    def __init__(self, data_root: str|Path):
        self.root = Path(data_root)

    def variants(self) -> List[str]:
        variants_dir = self.root
        if variants_dir.is_dir():
            return sorted([p.name for p in variants_dir.iterdir() if p.is_dir()])
        return []

    def subjects(self) -> List[str]:
        subjects_dir = self.root / 'Images'
        if subjects_dir.is_dir():
            return sorted([p.name for p in subjects_dir.iterdir() if p.is_dir()])
        return []
    
    def visits(self, subject: str) -> List[str]:
        visits_dir = self.root / 'Images' / subject
        if visits_dir.is_dir():
            return sorted([p.name for p in visits_dir.iterdir() if p.is_dir()])
        return []

    @lru_cache
    def site_code(self, subject: str, visit: str):
        image_dir = self.root / 'Images' / subject / visit / 'RAW'
        for nii_file in sorted(image_dir.glob('*.nii.gz')):
            try:
                return _parse_image_name(nii_file.name[:-7])['site']
            except RuntimeError:
                continue
        raise RuntimeError(f'failed to infer site code: {subject} / {visit}')

    def recons(self) -> List[str]:
        return list(self.VALID_RECONS)

    def states(self) -> List[str]:
        return list(self.VALID_STATES)

    def state_pairs(self) -> Iterable[Tuple[str, str]]:
        from itertools import permutations
        return list(permutations(self.states(), 2))

    def path(
        self,
        subject: str,
        variant: str,
        visit: str,
        state: str,
        recon: str,
        asset_type: str,
        **selectors
    ):
        visit = visit or self.DEFAULT_VISIT
        state = state or self.DEFAULT_STATE
        recon = recon or self.DEFAULT_RECON

        site = self.site_code(subject, visit)

        def image_name(stat: str) -> str:
            return f'{subject}_{stat}_{recon}_{site}_COPD'

        if variant == self.SOURCE_VARIANT:
            base_dir = self.root / 'Images' / subject / visit

            if asset_type == 'image':
                return base_dir / 'RAW' / f'{image_name(state)}.nii.gz'
        else:
            base_dir = self.root / variant / subject / visit

            if asset_type == 'image':
                return base_dir / 'images' / f'{image_name(state)}.nii.gz'

            elif asset_type == 'mask':
                mask_tag = selectors['mask_tag']
                return base_dir / 'masks' / image_name(state) / f'{mask_tag}.nii.gz'

            elif asset_type == 'mesh':
                mesh_tag = selectors['mesh_tag']
                return base_dir / 'meshes' / image_name(state) / f'{mesh_tag}.xdmf'

            elif asset_type == 'disp':
                fix_state = selectors['fixed_state']
                mov_state = selectors['moving_state']
                assert fix_state in self.VALID_STATES, fix_state
                assert mov_state in self.VALID_STATES, mov_state
                return base_dir / 'disps' / image_name(state) / f'{fix_state}_to_{mov_state}.nii.gz'

            elif asset_type == 'field':
                field_tag = selectors['field_tag']
                return base_dir / 'fields' / image_name(state) / f'{field_tag}.nii.gz'

        raise RuntimeError(f'unrecognized asset type: {asset_type}')

    def examples(
        self,
        subjects: Optional[List[str]]=None,
        variant: Optional[str]=None,
        visit: Optional[str]=None,
        state_pairs: Optional[List[Tuple[str, str]]]=None,
        recon: Optional[str]=None,
    ):
        visit = visit or self.DEFAULT_VISIT
        recon = recon or self.DEFAULT_RECON

        for subj in subjects or self.subjects():
            pairs = state_pairs or self.state_pairs()
            for fixed, moving in pairs:
                meta = {}
                meta['visit'] = visit
                meta['recon'] = recon
                meta['states'] = {'fixed': fixed, 'moving': moving}

                paths = {}
                paths['source_ref'] = self.path(subj, self.SOURCE_VARIANT, visit, self.DEFAULT_STATE, recon, asset_type='image')
                paths['source_fixed'] = self.path(subj, self.SOURCE_VARIANT, visit, fixed, recon, asset_type='image')
                paths['source_moving'] = self.path(subj, self.SOURCE_VARIANT, visit, moving, recon, asset_type='image')

                if variant: # generated by preprocessing pipeline
                    fixed_args = (subj, variant, visit, fixed, recon)
                    moving_args = (subj, variant, visit, moving, recon)

                    paths['fixed_image']  = self.path(*fixed_args, asset_type='image')
                    paths['moving_image'] = self.path(*moving_args, asset_type='image')

                    paths['binary_mask'] = self.path(*fixed_args, asset_type='mask', mask_tag='lung_combined')
                    paths['region_mask'] = self.path(*fixed_args, asset_type='mask', mask_tag='lung_regions')

                    paths['surface_mesh'] = self.path(*fixed_args, asset_type='mesh', mesh_tag='lung_surface')
                    paths['volume_mesh']  = self.path(*fixed_args, asset_type='mesh', mesh_tag='lung_volume')

                    paths['disp_field']  = self.path(*fixed_args, asset_type='disp', fixed_state=fixed, moving_state=moving)
                    paths['input_image'] = paths['fixed_image']

                    paths['node_values'] = self.path(*fixed_args, asset_type='mesh', mesh_tag='node_values')
                    paths['node_values_opt'] = self.path(*fixed_args, asset_type='mesh', mesh_tag='node_values_opt')
                    paths['elastic_field_opt'] = self.path(*fixed_args, asset_type='field', field_tag='elasticity_opt')

                yield base.Example(
                    dataset='COPDGene',
                    subject=subj,
                    variant=variant,
                    paths=paths,
                    metadata=meta
                )


# --- DEPRECATED ---


class COPDGeneVisit:

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
        import pandas as pd
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
        import pandas as pd
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

