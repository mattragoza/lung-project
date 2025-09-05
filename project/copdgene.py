import sys, os, pathlib
import numpy as np
import pandas as pd
import nibabel as nib
import xarray as xr

from . import registration
from . import segmentation
from . import deformation
from . import meshing
from . import utils


class COPDGeneVisit:
    '''
    <data_root>/Images/<subject_id>/<visit_name>/
        <variant>/
            <image_name>.nii.gz
            TotalSegmentator/
                <image_name>/
                    <mask_name>.nii.gz
            pygalmesh/
                <image_name>/
                    <mask_name>_<mesh_tag>.xdmf
            CorrField/
                <target_name>__<source_name>.nii.gz
    '''
    def __init__(self, data_root, subject_id, visit_name, site_code=None):
        self.data_root = pathlib.Path(data_root)
        self.subject_id = subject_id
        self.visit_name = visit_name
        self.site_code = site_code

    def __repr__(self):
        return (
            f'{type(self).__name__}('
            f'data_root={self.data_root}, '
            f'subject_id={self.subject_id}, '
            f'visit_name={self.visit_name}, '
            f'site_code={self.site_code})'
        )

    # Path helpers

    @property
    def visit_dir(self):
        return self.data_root / 'Images' / self.subject_id / self.visit_name

    def image_dir(self, variant):
        return self.visit_dir / variant

    def image_file(self, variant, image_name, ext='.nii.gz'):
        return self.image_dir(variant) / (image_name + ext)

    def mask_root(self, variant):
        return self.image_dir(variant) / 'TotalSegmentator'

    def mask_dir(self, variant, image_name):
        return self.mask_root(variant) / image_name

    def mask_file(self, variant, image_name, mask_name, ext='.nii.gz'):
        return self.mask_dir(variant, image_name) / (mask_name + ext)

    def mesh_root(self, variant):
        return self.image_dir(variant) / 'pygalmesh'

    def mesh_dir(self, variant, image_name):
        return self.mesh_root(variant) / image_name

    def mesh_file(self, variant, image_name, mask_name, mesh_tag, ext='.xdmf'):
        return self.mesh_dir(variant, image_name) / f'{mask_name}_{mesh_tag}{ext}'

    def disp_root(self, variant):
        return self.image_dir(variant) / 'CorrField'

    def disp_file(self, variant, target_name, source_name):
        return self.disp_root(variant) / (target_name + '__' + source_name + '.nii.gz')

    # Directory exploration

    def list_variants(self):
        return sorted([x.name for x in self.visit_dir.iterdir() if x.is_dir()])

    def list_images(self, variant, ext='.nii.gz'):
        image_dir = self.image_dir(variant)
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

    def build_image_name(self, state, recon):
        return f'{self.subject_id}_{state}_{recon}_{self.site_code}_COPD'

    def has_valid_image_pair(self, variant, recon, ext='.nii.gz'):
        for state in ['INSP', 'EXP']:
            image_name = self.build_image_name(state, recon)
            image_path = self.image_file(variant, image_name, ext)
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
        for tup in self.load_metadata_from_filenames(variant, filters).itertuples():
            image = nib.load(self.image_file(variant, tup.image_name))
            dct = tup._asdict()
            dct.update({
                'shape': image.header.get_data_shape(),
                'resolution': image.header.get_zooms(),
            })
            rows.append(dct)
        return pd.DataFrame(rows)

    # Image loaders

    def load_image(self, variant, image_name):
        image_path = self.image_file(variant, image_name)
        image = nib.load(image_path)
        return utils.as_xarray(
            image.get_fdata(),
            dims=['x', 'y', 'z'],
            resolution=image.header.get_zooms(),
            name=image_name
        )

    def load_images(self, variant, recon, site_code):
        images = {}
        for state in ['INSP', 'EXP']:
            image_name = self.build_image_name(state, recon, site_code)
            images[state] = self.load_image(variant, image_name)
        return images

    def load_masks(self, variant, image_name, roi_list):
        masks = []

        image_path = self.image_file(variant, image_name)
        image = nib.load(image_path)
        resolution = image.header.get_zooms()

        for roi in roi_list:
            mask_path = self.mask_file(variant, image_name, roi)
            mask = nib.load(mask_path)
            assert mask.shape == image.shape
            masks.append(mask.get_fdata())

        return utils.as_xarray(
            np.stack(masks, axis=0),
            dims=['roi', 'x', 'y', 'z'],
            coords={'roi': roi_list},
            resolution=resolution
        )

    # Preprocessing operations

    def resample_images(
        self,
        input_variant='RAW',
        output_variant='ISO',
        states=['EXP', 'INSP'],
        recon='STD',
        resolution=[1.0, 1.0, 1.0],
        interp='bspline',
        default=-1024
    ):
        import SimpleITK as sitk

        ref_image_name = self.build_image_name(states[0], recon)
        ref_image_path = self.image_file(input_variant, ref_image_name)

        print(f'[{ref_image_name}] Creating reference grid')
        ref_image = sitk.ReadImage(ref_image_path)
        ref_grid = registration.create_reference_grid(ref_image, new_spacing=resolution)

        for i, state in enumerate(states):
            image_name = self.build_image_name(state, recon)
            input_image_path = self.image_file(input_variant, image_name)
            output_image_path = self.image_file(output_variant, image_name)

            print(f'[{image_name}] Resampling image on reference grid')
            input_image = sitk.ReadImage(input_image_path) if i > 0 else ref_image
            output_image = registration.resample_image_on_grid(input_image, ref_grid, interp, default)
            sitk.WriteImage(output_image, output_image_path)

    def create_segmentation_masks(
        self,
        variant='ISO',
        state='EXP',
        recon='STD',
        combined_mask_name='lung_combined_mask'
    ):
        import totalsegmentator as ts
        import totalsegmentator.python_api
        import totalsegmentator.libs

        image_name = self.build_image_name(state, recon)
        image_path = self.image_file(variant, image_name)
        mask_path = self.mask_file(variant, image_name, mask_name=combined_mask_name)
        mask_dir = self.mask_dir(variant, image_name)
        mask_dir.mkdir(parents=True, exist_ok=True)

        print(f'[{image_name}] Running segmentation task: total')
        ts.python_api.totalsegmentator(image_path, mask_dir, task='total', roi_subset=segmentation.TOTAL_TASK_ROIS)

        print(f'[{image_name}] Running segmentation task: lung_vessels')
        ts.python_api.totalsegmentator(image_path, mask_dir, task='lung_vessels')

        print(f'[{image_name}] Combining segmentation masks: lung')
        combined_nifti = ts.libs.combine_masks(mask_dir=mask_dir, class_type='lung')
        nib.save(combined_nifti, mask_path)

    def create_lung_regions_mask(
        self,
        variant='ISO',
        state='EXP',
        recon='STD',
        lungs_mask_name='lung_combined_mask',
        airways_mask_name='lung_trachea_bronchia',
        vessels_mask_name='lung_vessels',
        regions_mask_name='lung_regions',
        **kwargs
    ):
        image_name = self.build_image_name(state, recon)

        lungs_path = self.mask_file(variant, image_name, mask_name=lungs_mask_name)
        airways_path = self.mask_file(variant, image_name, mask_name=airways_mask_name)
        vessels_path = self.mask_file(variant, image_name, mask_name=vessels_mask_name)
        regions_path = self.mask_file(variant, image_name, mask_name=regions_mask_name)

        lungs_nifti = nib.load(lungs_path)
        lungs_mask = lungs_nifti.get_fdata()
        airways_mask = nib.load(airways_path).get_fdata()
        vessels_mask = nib.load(vessels_path).get_fdata()

        print(f'[{image_name}] Creating lung regions mask')
        regions_mask = segmentation.create_lung_regions_mask(lungs_mask, airways_mask, vessels_mask, **kwargs)

        regions_nifti = nib.nifti1.Nifti1Image(regions_mask, lungs_nifti.affine)
        nib.save(regions_nifti, regions_path)

    def create_anatomical_mesh(
        self,
        variant='ISO',
        state='EXP',
        recon='STD',
        mask_name='lung_regions',
        volume_mesh_tag='volume',
        surface_mesh_tag='surface',
        **kwargs
    ):
        import meshio

        image_name = self.build_image_name(state, recon)
        mask_path = self.mask_file(variant, image_name, mask_name)

        mesh_dir = self.mesh_dir(variant, image_name)
        mesh_dir.mkdir(parents=True, exist_ok=True)

        volume_path = self.mesh_file(variant, image_name, mask_name, mesh_tag=volume_mesh_tag)
        surface_path = self.mesh_file(variant, image_name, mask_name, mesh_tag=surface_mesh_tag)

        mask_nifti = nib.load(mask_path)
        mask_array = mask_nifti.get_fdata()
        resolution = mask_nifti.header.get_zooms()[:3]
        A = mask_nifti.affine

        print(f'[{image_name}] Creating anatomical mesh')
        mesh_dict = meshing.generate_mesh_from_mask(mask_array, resolution, **kwargs)

        print(f'[{image_name}] Applying affine to mesh')
        volume_mesh = meshing.apply_affine_to_mesh(mesh_dict['tetra'], resolution, A)
        surface_mesh = meshing.apply_affine_to_mesh(mesh_dict['triangle'], resolution, A)

        meshio.xdmf.write(volume_path, volume_mesh)
        meshio.xdmf.write(surface_path, surface_mesh)

        return meshing.load_mesh_with_fenics(volume_path)

    def create_displacement_field(
        self,
        variant='ISO',
        source_state='INSP',
        target_state='EXP',
        recon='STD',
        mask_name='lung_combined_mask'
    ):
        import torch
        import corrfield

        source_name = self.build_image_name(source_state, recon)
        target_name = self.build_image_name(target_state, recon)

        source_path = self.image_file(variant, source_name)
        target_path = self.image_file(variant, target_name)
        mask_path = self.mask_file(variant, target_name, mask_name)

        disp_path = self.disp_file(variant, target_name, source_name)
        disp_path.parent.mkdir(parents=True, exist_ok=True)

        source_nifti = nib.load(source_path)
        target_nifti = nib.load(target_path)
        mask_nifti = nib.load(mask_path)

        source_array = source_nifti.get_fdata()
        target_array = target_nifti.get_fdata()
        mask_array = mask_nifti.get_fdata()

        print('Creating displacement field')

        disp_array = corrfield.corrfield.corrfield(
            img_mov=torch.as_tensor(source_array, dtype=torch.float)[None,None,...].cuda(),
            img_fix=torch.as_tensor(target_array, dtype=torch.float)[None,None,...].cuda(),
            mask_fix=torch.as_tensor(mask_array, dtype=torch.float)[None,None,...].cuda()
        )[0][0].detach().cpu().numpy()

        disp_nifti = nib.nifti1.Nifti1Image(disp_array, target_nifti.affine)
        nib.save(disp_nifti, disp_path)


class COPDGeneDataset:

    @classmethod
    def from_csv(cls, data_file, *args, **kwargs):
        df = pd.read_csv(data_file, sep='\t', low_memory=False)
        return cls(df, *args, **kwargs)

    def __init__(self, df, data_root, visit_name='Phase-1'):
        self.df = df
        self.data_root = pathlib.Path(data_root)
        self.visit_name = visit_name

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        visit = COPDGeneVisit(self.data_root, row.sid, self.visit_name, row.ccenter)
        return row, visit


class COPDGene:

    def __init__(self, data_root, subject_ids=None, visit_names=None):
        self.data_root = pathlib.Path(data_root)
        self.subject_ids = subject_ids or self.list_subjects()
        self.subjects = []
        for subject_id in self.subject_ids:
            subject = COPDGeneSubject(self.data_root, subject_id, visit_names)
            self.subjects.append(subject)

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        return self.subjects[idx]

    def __repr__(self):
        return f'{type(self).__name__}(data_root={self.data_root}, #subjects={len(self)})'

    @property
    def image_root(self):
        return self.data_root / 'Images'

    def list_subjects(self):
        return sorted([x.name for x in self.image_root.iterdir()])

    def load_metadata(self, subdir):
        metadata = []
        for subject in self.subjects:
            m = subject.load_metadata(subdir)
            metadata.append(m)
        return pd.concat(metadata)

    def get_examples(
        self,
        image_dir='Resized',
        mask_dir='TotalSegment',
        mask_roi='lung_regions',
        mesh_dir='pygalmesh',
        mesh_version=10,
        disp_dir='CorrField'
    ):
        examples = []
        for subject in self.subjects:
            for visit in subject.visits:
                for fix_name in visit.list_images(image_dir):
                    if '_SHARP_' in fix_name:
                        continue
                    if '_INSP_' in fix_name:
                        mov_name = fix_name.replace('_INSP_', '_EXP_')
                    elif '_EXP_' in fix_name:
                        mov_name = fix_name.replace('_EXP_', '_INSP_')
                    examples.append({
                        'name': visit.image_file(image_dir, fix_name).name[:-7],
                        'anat_file': visit.image_file(image_dir, fix_name),
                        'disp_file': visit.disp_file(disp_dir, fix_name, mov_name),
                        'mask_file': visit.mask_file(mask_dir, fix_name, mask_roi),
                        'mesh_file': visit.mesh_file(mesh_dir, fix_name, mask_roi, mesh_version),
                        'has_labels': True
                    })
        return examples


def is_nifti_file(path):
    try:
        nib.load(path)
        return True
    except Exception:
        return False
