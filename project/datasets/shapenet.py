from typing import Optional, Any, List, Dict, Tuple, Iterable
from pathlib import Path
import numpy as np

from . import base


def _parse_sid_code(subj: str):
    parts = subj.split('.')
    if len(parts) == 2 and parts[0] == 'wss':
        return parts[1]
    raise RuntimeError(f'failed to parse subject ID code: {subj}')


def _parse_category(s: str) -> List[str]:
    return [c.strip() for c in str(s).split(',')]


def _parse_aligned_dims(s: str) -> np.ndarray:
    parts = s.replace('\\,', ',').split(',')
    return np.array([float(p) for p in parts])


class ShapeNetDataset:
    '''
    <data_root>/
        metadata.csv
        materials.csv
        densities.csv
        taxonomy.txt
        categories.synset.txt
        models-OBJ/models/
            <sid_code>.obj
            <sid_code>.mtl
        models-COLLADA/COLLADA/
            models/
        models-textures/
            <tid_code>.jpg
        models-binvox/
        models-binvox-solid/
            <sid_code>.binvox
    '''
    ID_COLUMN = 'fullId'
    SOURCE_VARIANT = 'models'

    def __init__(self, data_root: str | Path):
        self.root = Path(data_root)
        self.load_metadata()

    def load_metadata(self):
        import pandas as pd
        self.metadata   = pd.read_csv(self.root / 'metadata.csv')
        self.categories = pd.read_csv(self.root / 'categories.synset.csv')
        self.materials  = pd.read_csv(self.root / 'materials.csv')
        self.densities  = pd.read_csv(self.root / 'densities.csv')
        self.taxonomy = load_taxonomy(self.root / 'taxonomy.txt')

    def subjects(self) -> List[str]:
        return self.metadata[self.ID_COLUMN].to_list()

    def path(
        self,
        subject: str,
        variant: str,
        asset_type: str,
        **selectors
    ):
        sid_code = _parse_sid_code(subject)

        if variant == self.SOURCE_VARIANT:
            if asset_type == 'mesh':
                return self.root / 'models-OBJ' / 'models' / f'{sid_code}.obj'

            elif asset_type == 'mask':
                return self.root / 'models-binvox-solid' / f'{sid_code}.binvox'
        else:
            base_dir = self.root / variant / sid_code

            if asset_type == 'mesh':
                mesh_tag = selectors['mesh_tag']
                return base_dir / 'meshes' / f'{mesh_tag}.xdmf'

            elif asset_type == 'mask':
                mask_tag = selectors['mask_tag']
                return base_dir / 'masks' / f'{mask_tag}.nii.gz'

            elif asset_type == 'field':
                field_tag = selectors['field_tag']
                return base_dir / 'fields' / f'{field_tag}.nii.gz'

            elif asset_type == 'image':
                image_tag = selectors['image_tag']
                return base_dir / 'images' / f'{image_tag}.nii.gz'

            raise RuntimeError(f'unrecognized asset type: {asset_type}')

    def examples(self, subjects: List[str], variant: str):
        meta = self.metadata.set_index(self.ID_COLUMN)

        for subj in subjects or self.subjects():
            paths = {}
            paths['source_mesh'] = self.path(subj, self.SOURCE_VARIANT, asset_type='mesh')
            paths['source_mask'] = self.path(subj, self.SOURCE_VARIANT, asset_type='mask')

            paths['surface_mesh'] = self.path(subj, variant, asset_type='mesh', mesh_tag='surface')
            paths['binary_mask']  = self.path(subj, variant, asset_type='mask', mask_tag='binary')

            paths['region_mask'] = self.path(subj, variant, asset_type='mask', mask_tag='regions')
            paths['volume_mesh'] = self.path(subj, variant, asset_type='mesh', mesh_tag='volume')

            paths['material_mask'] = self.path(subj, variant, asset_type='mask', mask_tag='material')
            paths['density_field'] = self.path(subj, variant, asset_type='field', field_tag='density')
            paths['elastic_field'] = self.path(subj, variant, asset_type='field', field_tag='elasticity')

            paths['node_values'] = self.path(subj, variant, asset_type='mesh',  mesh_tag='node_values')
            paths['disp_field']  = self.path(subj, variant, asset_type='field', field_tag='displacement')
            paths['input_image'] = self.path(subj, variant, asset_type='image', image_tag='generated')

            yield base.Example(
                dataset='ShapeNet',
                subject=subj,
                variant=variant,
                paths=paths,
                metadata=dict(meta.loc[subj])
            )


def load_taxonomy(path):
    parent_of = {}
    children_of = {}
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        tokens = line.split()
        parent = tokens[0]
        if len(tokens) > 1:
            children = [c.rstrip(',') for c in tokens[1:]]
        else:
            children = []
        for c in children:
            parent_of[c] = parent
        if parent in children_of:
            children_of[parent].update(children)
        else:
            children_of[parent] = set(children)

    return {'parents': parent_of, 'children': children_of}


def get_resolver(data_root):
    import os, trimesh

    class ShapeNetResolver(trimesh.resolvers.Resolver):
        '''
        Trimesh path resolver for ShapeNet dataset.
        '''
        def __init__(self, data_root):
            self.root = Path(data_root)
            self.obj_root = self.root / 'models-OBJ' / 'models'
            self.tex_root = self.root / 'models-textures' / 'textures'

        def get_path(self, name):
            name = str(name).strip().replace('\\', '/').lstrip('/')
            ext = os.path.splitext(name)[1].lower()
            if ext in {'.obj', '.mtl'}:
                return self.obj_root / name
            elif ext in {'.jpg', '.jpeg', '.png'}:
                return self.tex_root / name
            else:
                return name

        def get(self, name):
            print('get', name)
            path = self.get_path(name)
            data = path.read_bytes()
            return data

        def write(self, name, data):
            print('write', name, data)
            raise NotImplementedError

        def namespaced(self, namespace):
            print('namespaced', namespace)
            raise NotImplementedError

        def keys(self):
            print('keys')
            raise NotImplementedError

    return ShapeNetResolver(data_root)

