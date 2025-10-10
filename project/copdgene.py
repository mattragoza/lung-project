import sys, os, pathlib
import numpy as np
import pandas as pd
import nibabel as nib
import xarray as xr

from . import registration
from . import segmentation
from . import meshing


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
            image_path = selfe.get_image_path(variant, t.image_name)
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
        image_name = self.get_image_name(variant, state, recon)
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
        import meshio
        image_name = self.get_image_name(state, recon)
        mesh_path = self.get_mesh_path(variant, image_name, mask_name, mesh_tag)
        return meshio.read(mesh_path)

    def load_mesh_with_fenics(self, variant, state, mask_name, mesh_tag, recon='STD'):
        import meshio
        image_name = self.get_image_name(state, recon)
        mesh_path = self.get_mesh_path(variant, image_name, mask_name, mesh_tag)
        return meshing.load_mesh_with_fenics(mesh_path)

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
        recon='STD',
        resolution=[1.0, 1.0, 1.0],
        interp='bspline',
        default=-1024
    ):
        import SimpleITK as sitk

        ref_image_name = self.get_image_name(states[0], recon)
        ref_image_path = self.get_image_path(input_variant, ref_image_name)

        print(f'[{ref_image_name}] Creating reference grid')
        ref_image = sitk.ReadImage(ref_image_path)
        ref_grid = registration.create_reference_grid(ref_image, new_spacing=resolution)

        for i, state in enumerate(states):
            image_name = self.get_image_name(state, recon)
            input_path = self.get_image_path(input_variant, image_name)
            output_path = self.get_image_path(output_variant, image_name)

            print(f'[{image_name}] Resampling image on reference grid')
            input_image = sitk.ReadImage(input_path) if i > 0 else ref_image
            output_image = registration.resample_image_on_grid(input_image, ref_grid, interp, default)
            sitk.WriteImage(output_image, output_path)

    def create_segmentation_masks(
        self,
        variant='ISO',
        state='EXP',
        recon='STD',
        combined_mask_name='lung_combined_mask'
    ):
        import totalsegmentator as totalseg
        import totalsegmentator.python_api
        import totalsegmentator.libs

        image_name = self.get_image_name(state, recon)
        image_path = self.get_image_path(variant, image_name)

        mask_path = self.get_mask_path(variant, image_name, mask_name=combined_mask_name)
        mask_dir = self.get_mask_dir(variant, image_name)
        mask_dir.mkdir(parents=True, exist_ok=True)

        print(f'[{image_name}] Running segmentation task: total')
        totalseg.python_api.totalsegmentator(
            image_path, mask_dir, task='total', roi_subset=segmentation.TOTAL_TASK_ROIS
        )

        print(f'[{image_name}] Running segmentation task: lung_vessels')
        totalseg.python_api.totalsegmentator(image_path, mask_dir, task='lung_vessels')

        print(f'[{image_name}] Combining segmentation masks: lung')
        combined_nifti = totalseg.libs.combine_masks(mask_dir=mask_dir, class_type='lung')
        nib.save(combined_nifti, mask_path)

    def create_multi_region_mask(
        self,
        variant='ISO',
        state='EXP',
        recon='STD',
        mask_name='lung_regions',
        connectivity=1
    ):
        image_name = self.get_image_name(state, recon)

        region_arrays = []
        for i, roi in enumerate(segmentation.ALL_TASK_ROIS):
            print(f'[{image_name}] Processing roi mask: {roi}')
            mask_path = self.get_mask_path(variant, image_name, mask_name=roi)
            mask_nifti = nib.load(mask_path)
            mask_array = mask_nifti.get_fdata().astype(int)
            filtered_array = segmentation.filter_connected_components(
                mask_array,
                connectivity=connectivity,
                max_components=(1 if roi in segmentation.TOTAL_TASK_ROIS else None)
            )
            region_arrays.append(filtered_array * (i + 1))

        regions_array = np.max(region_arrays, axis=0).astype(np.float64)
        regions_nifti = nib.nifti1.Nifti1Image(regions_array, mask_nifti.affine)
        regions_path = self.get_mask_path(variant, image_name, mask_name=mask_name)
        nib.save(regions_nifti, regions_path)

    def create_anatomical_mesh(
        self,
        variant='ISO',
        state='EXP',
        recon='STD',
        mask_name='lung_regions',
        volume_tag='volume',
        surface_tag='surface',
        **kwargs
    ):
        import meshio

        image_name = self.get_image_name(state, recon)
        mask_path = self.get_mask_path(variant, image_name, mask_name)

        mesh_dir = self.get_mesh_dir(variant, image_name)
        mesh_dir.mkdir(parents=True, exist_ok=True)

        volume_path = self.get_mesh_path(variant, image_name, mask_name, mesh_tag=volume_tag)
        surface_path = self.get_mesh_path(variant, image_name, mask_name, mesh_tag=surface_tag)

        mask_nifti = nib.load(mask_path)
        mask_array = mask_nifti.get_fdata().astype(int)
        resolution = mask_nifti.header.get_zooms()[:3]
        affine = mask_nifti.affine

        print(f'[{image_name}] Creating anatomical mesh')
        mesh_dict = meshing.generate_mesh_from_mask(mask_array, resolution, **kwargs)

        print(f'[{image_name}] Applying affine to mesh')
        volume_mesh = meshing.apply_affine_to_mesh(mesh_dict['tetra'], resolution, affine)
        surface_mesh = meshing.apply_affine_to_mesh(mesh_dict['triangle'], resolution, affine)

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
        import torch.nn.functional as F
        import corrfield

        source_name = self.get_image_name(source_state, recon)
        target_name = self.get_image_name(target_state, recon)

        source_path = self.get_image_path(variant, source_name)
        target_path = self.get_image_path(variant, target_name)
        mask_path = self.get_mask_path(variant, target_name, mask_name)

        disp_path = self.get_disp_path(variant, target_name, source_name)
        disp_path.parent.mkdir(parents=True, exist_ok=True)

        source_nifti = nib.load(source_path)
        target_nifti = nib.load(target_path)
        mask_nifti = nib.load(mask_path)

        source_array = source_nifti.get_fdata()
        target_array = target_nifti.get_fdata()
        mask_array = mask_nifti.get_fdata()

        source_tensor = torch.as_tensor(source_array, dtype=torch.float, device='cuda')
        target_tensor = torch.as_tensor(target_array, dtype=torch.float, device='cuda')
        mask_tensor = torch.as_tensor(mask_array, dtype=torch.int, device='cuda')

        print(f'[{source_name} -> {target_name}] Creating displacement field')

        disp_tensor_ijk = corrfield.corrfield.corrfield(
            img_mov=source_tensor[None,None,...],
            img_fix=target_tensor[None,None,...],
            mask_fix=mask_tensor[None,None,...]
        )[0][0]

        # check registration error
        target_shape = target_tensor.shape
        disp_normalized = corrfield.utils.flow_pt(
            disp_tensor_ijk[None,...],
            shape=target_shape,
            align_corners=True
        )
        base = F.affine_grid(
            torch.eye(3, 4, dtype=torch.float, device='cuda')[None,...],
            size=(1,1,target_shape[0],target_shape[1],target_shape[2]),
            align_corners=True
        )
        warped_tensor = F.grid_sample(
            input=source_tensor[None,None,...],
            grid=base + disp_normalized,
            align_corners=True
        )

        def compute_error(t):
            mask = (mask_tensor[None,...] > 0)
            return torch.norm((t - target_tensor) * mask) / torch.norm(target_tensor * mask)

        error_before = compute_error(source_tensor)
        error_after  = compute_error(warped_tensor)

        print(error_before)
        print(error_after)

        assert error_after < error_before, 'registration error did not decrease'

        # convert from voxel to world coordinates (mm)
        R = torch.as_tensor(target_nifti.affine[:3,:3], dtype=torch.float, device='cuda')
        disp_tensor_world = torch.einsum('ij,dhwj->dhwi', R, disp_tensor_ijk)

        disp_array = disp_tensor_world.detach().cpu().numpy()
        disp_nifti = nib.nifti1.Nifti1Image(disp_array, target_nifti.affine)
        disp_nifti.header.set_intent('vector', (), name='displacement_mm')
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

