import sys, os, pathlib
import numpy as np
import xarray as xr
import pandas as pd
import nibabel as nib

from . import meshing


class COPDGene(object):

    def __init__(self, data_root, subject_ids=None, visit_names=None):
        self.data_root = pathlib.Path(data_root)
        self.subject_ids = subject_ids or self.list_subjects()
        self.subjects = []
        for subject_id in self.subject_ids:
            subject = COPDGeneSubject(data_root, subject_id, visit_names)
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


class COPDGeneSubject(object):

    def __init__(self, data_root, subject_id, visit_names=None):
        self.data_root = pathlib.Path(data_root)
        self.subject_id = subject_id
        self.visit_names = visit_names or self.list_visits()
        self.visits = []
        for visit_name in self.visit_names:
            visit = COPDGeneVisit(data_root, subject_id, visit_name)
            self.visits.append(visit)

    def __len__(self):
        return len(self.visits)

    def __getitem__(self, idx):
        return self.visits[idx]

    def __repr__(self):
        return f'{type(self).__name__}(data_root={self.data_root}, subject_id={self.subject_id}, #visits={len(self)})'

    @property
    def subject_dir(self):
        return self.data_root / 'Images' / self.subject_id

    def list_visits(self):
        return sorted([x.name for x in self.subject_dir.iterdir()])

    def load_images(self, subdir):
        images = []
        for visit in self.visits:
            images.append(visit.load_images(subdir))
        return images

    def load_metadata(self, subdir):
        metadata = []
        for visit in self.visits:
            m = visit.load_metadata(subdir)
            metadata.append(m)
        return pd.concat(metadata)


class COPDGeneVisit(object):

    def __init__(self, data_root, subject_id, visit_name):
        self.data_root = pathlib.Path(data_root)
        self.subject_id = subject_id
        self.visit_name = visit_name

    def __repr__(self):
        return f'{type(self).__name__}(data_root={self.data_root}, subject_id={self.subject_id}, visit_name={self.visit_name})'

    @property
    def visit_dir(self):
        return self.data_root / 'Images' / self.subject_id / self.visit_name

    def list_subdirs(self):
        return sorted([x.name for x in self.visit_dir.iterdir() if x.is_dir()])

    def image_dir(self, subdir):
        return self.visit_dir / subdir

    def image_file(self, subdir, image_name):
        return self.image_dir(subdir) / (image_name + '.nii.gz')

    def image_files(self, subdir, pattern='*'):
        return self.image_dir(subdir).glob(pattern + '.nii.gz')

    def list_images(self, subdir, pattern='*'):
        return sorted([x.name[:-7] for x in self.image_files(subdir, pattern)])

    def mask_dir(self, subdir, image_name):
        return self.visit_dir / subdir / image_name

    def mask_file(self, subdir, image_name, roi):
        return self.mask_dir(subdir, image_name) / (roi + '.nii.gz')

    def mask_files(self, subdir, image_name, pattern='*'):
        return self.mask_dir(subdir, image_name).glob(pattern + '.nii.gz')

    def list_masks(self, subdir, image_name, pattern='*'):
        return sorted([
            x.name[:-7] for x in self.mask_files(subdir, image_name, pattern)
        ])

    def mesh_dir(self, subdir, image_name):
        return self.visit_dir / subdir / image_name

    def mesh_file(self, subdir, image_name, mask_roi, mesh_version):
        return self.mesh_dir(subdir, image_name) / f'{mask_roi}_{mesh_version}.xdmf'

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

    def load_metadata(self, subdir, pattern='*'):
        metadata = []
        for image_name in self.list_images(subdir, pattern):
            image_file = self.image_file(subdir, image_name)
            image = nib.load(image_file)
            shape = image.header.get_data_shape()
            resolution = image.header.get_zooms()
            metadata.append({
                'subject_id': self.subject_id,
                'visit_name': self.visit_name,
                'image_name': image_name,
                'shape': shape,
                'resolution': resolution
            })
        return pd.DataFrame(metadata)

    def load_images(self, subdir, pattern='*'):
        images = []
        for image_name in self.list_images(subdir, pattern):
            image_file = self.image_file(subdir, image_name)
            print(f'Loading {image_file}')
            image = nib.load(image_file)
            shape = image.header.get_data_shape()
            resolution = image.header.get_zooms()
            print(image_file, shape)
            images.append(xr.DataArray(
                data=image.get_fdata(),
                dims=['x', 'y', 'z'],
                coords={
                    'x': np.arange(shape[0]) * resolution[0],
                    'y': np.arange(shape[1]) * resolution[1],
                    'z': np.arange(shape[2]) * resolution[2],
                },
                name=image_name
            ))
        return images

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

    def load_masks(self, image_dir, mask_dir):
        masks = []
        for image_name in self.list_images(image_dir):
            image_masks, mask_rois = [], []
            for mask_roi in self.list_masks(mask_dir, image_name):
                mask_file = self.mask_file(mask_dir, image_name, mask_roi)
                print(f'Loading {mask_file}')
                mask = nib.load(mask_file)
                shape = mask.header.get_data_shape()
                resolution = mask.header.get_zooms()
                print(mask_file, shape)
                image_masks.append(mask.get_fdata())
                mask_rois.append(mask_roi)

            masks.append(xr.DataArray(
                data=np.stack(image_masks),
                dims=['roi', 'x', 'y', 'z'],
                coords={
                    'roi': mask_rois,
                    'x': np.arange(shape[0]) * resolution[0],
                    'y': np.arange(shape[1]) * resolution[1],
                    'z': np.arange(shape[2]) * resolution[2],
                },
                name=image_name
            ))
        return masks

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

