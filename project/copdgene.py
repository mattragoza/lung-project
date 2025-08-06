import sys, os, pathlib
import numpy as np
import pandas as pd
import nibabel as nib
import xarray as xr

from . import meshing
from . import utils


class COPDGeneVisit:
    '''
    <data_root>/Images/<subject_id>/<visit_name>/
        <variant>/
            <image_name>.nii.gz
            TotalSegmentator/
                <image_name>/
                    <roi_name>.nii.gz
            pygalmesh/
            CorrField/
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

    # Directory exploration

    def list_variants(self):
        return sorted([x.name for x in self.visit_dir.iterdir() if x.is_dir()])

    def has_variant(self, variant):
        return self.image_dir(variant).is_dir()

    def list_images(self, variant, ext='.nii.gz'):
        image_dir = self.image_dir(variant)
        assert image_dir.is_dir(), f'Directory does not exist: {image_dir}'
        return sorted([x.name[:-len(ext)] for x in image_dir.glob('*' + ext)])

    def has_image(self, variant, image_name):
        return self.image_file(variant, image_name).is_file()

    def has_masks(self, variant, image_name):
        return self.mask_dir(variant, image_name).is_dir()

    def list_mask_rois(self, variant, image_name, ext='.nii.gz'):
        mask_dir = self.mask_dir(variant, image_name)
        assert mask_dir.is_dir(), f'Directory does not exist: {mask_dir}'
        return sorted([x.name[:-len(ext)] for x in mask_dir.glob('*' + ext)])

    def has_mask_roi(self, variant, image_name, roi):
        return self.mask_file(variant, image_name, roi).is_file()

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

    # Displacement fields

    def disp_dir(self, subdir, fix_image_name):
        return self.visit_dir / subdir / fix_image_name

    def disp_file(self, subdir, fix_image_name, mov_image_name):
        return self.disp_dir(subdir, fix_image_name) / (mov_image_name + '.nii.gz')

    def disp_files(self, subdir, fix_image_name, pattern='*'):
        return self.disp_dir(subdir, fix_image_name).glob(pattern + '.nii.gz')

    def list_displacements(self, subdir, fix_image_name, pattern='*'):
        return sorted([
            x.name[:-7] for x in self.disp_files(subdir, fix_image_name, pattern)
        ])


    def save_images(self, subdir, images):
        self.image_dir(subdir).mkdir(parents=True, exist_ok=True)
        for image in images:
            image_file = self.image_file(subdir, image.name)
            print(f'Saving {image_file}')
            x_res = (image.x[1] - image.x[0]).item() 
            y_res = (image.y[1] - image.y[0]).item()
            z_res = (image.z[1] - image.z[0]).item()
            affine = np.diag([x_res, y_res, z_res, 1])
            nifti = nib.nifti1.Nifti1Image(image, affine)
            nib.save(nifti, image_file)


    def load_meshes(self, image_dir, mesh_dir, mask_roi, mesh_version):
        meshes = []
        labels = []
        for image_name in self.list_images(image_dir):
            mesh_file = self.mesh_file(mesh_dir, image_name, mask_roi, mesh_version)
            has_labels = True
            print(f'Loading {mesh_file}')
            mesh, cell_labels = meshing.load_mesh_fenics(mesh_file, has_labels)
            meshes.append(mesh)
            labels.append(cell_labels)
        return meshes, labels

    def load_displacements(self, image_dir, disp_dir):
        disps = []
        for fix_image_name in self.list_images(image_dir):
            fix_image_disps = []
            mov_image_names = []
            for mov_image_name in self.list_displacements(disp_dir, fix_image_name):
                disp_file = self.disp_file(disp_dir, fix_image_name, mov_image_name)
                print(disp_file, disp_file.exists())
                disp = nib.load(disp_file)
                shape = disp.header.get_data_shape()
                resolution = disp.header.get_zooms()
                fix_image_disps.append(disp.get_fdata())
                mov_image_names.append(mov_image_name)

            disps.append(xr.DataArray(
                data=np.stack(fix_image_disps),
                dims=['mov_image_name', 'x', 'y', 'z', 'component'],
                coords={
                    'mov_image_name': mov_image_names,
                    'x': np.arange(shape[0]) * resolution[0],
                    'y': np.arange(shape[1]) * resolution[1],
                    'z': np.arange(shape[2]) * resolution[2],
                    'component': ['x', 'y', 'z']
                },
                name=fix_image_name
            ))
        return disps


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
