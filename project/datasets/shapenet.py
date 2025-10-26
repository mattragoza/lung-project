from typing import Optional, Any, List, Dict, Tuple, Iterable
from pathlib import Path
import numpy as np

import trimesh # for resolver

from . import base


def _parse_id_code(s: str):
    parts = s.split('.')
    if len(parts) == 2:
        return parts[1]
    raise RuntimeError(f'failed to parse ID code: {full_id}')


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
            <id_code>.obj
            <id_code>.mtl
        models-COLLADA/COLLADA/
            models/
        models-textures/
            <texture>.tex
        models-binvox/
        models-binvox-solid/
    '''
    ID_COLUMN = 'fullId'

    def __init__(self, data_root: str | Path):
        import pandas as pd
        self.root = Path(data_root)
        self._load_metadata()

    def _load_metadata(self):
        import pandas as pd
        self.metadata   = pd.read_csv(self.root / 'metadata.csv')
        self.categories = pd.read_csv(self.root / 'categories.synset.csv')
        self.materials  = pd.read_csv(self.root / 'materials.csv')
        self.densities  = pd.read_csv(self.root / 'densities.csv')
        self.taxonomy = load_taxonomy(self.root / 'taxonomy.txt')

    def subjects(self) -> List[str]:
        return self.metadata[self.ID_COLUMN].to_list()

    def get_path(
        self,
        subject: str,
        variant: str,
        asset_type: str,
        **selectors
    ):
        variant_dir = self.root / variant
        id_code = _parse_id_code(subject)

        if asset_type == 'mesh':
            if variant == 'RAW':
                mesh_root = self.root / 'models-OBJ' / 'models'
                return mesh_root / f'{id_code}.obj'

            mesh_tag = selectors['mesh_tag']
            return variant_dir / id_code / 'meshes' / f'{mesh_tag}.xdmf'

        elif asset_type == 'mask':
            if variant == 'RAW':
                mask_root = self.root / 'models-binvox-solid'
                return mask_root / f'{id_code}.binvox'

            mask_tag = selectors['mask_tag']
            return variant_dir / id_code / 'masks' / f'{mask_tag}.nii.gz'

        elif asset_type == 'field':
            assert variant != 'RAW'
            field_tag = selectors['field_tag']
            return variant_dir / id_code / 'fields' / f'{field_tag}.nii.gz'

        elif asset_type == 'image':
            assert variant != 'RAW'
            image_tag = selectors['image_tag']
            return variant_dir / id_code / 'images' / f'{image_tag}.nii.gz'

        raise RuntimeError(f'unrecognized asset type: {asset_type}')

    def examples(
        self,
        subjects: Optional[List[str]]=None,
        variant: Optional[str]=None
    ):
        subjects = subjects or self.subjects()
        metadata = self.metadata.set_index(self.ID_COLUMN)

        for subj in subjects:
            paths = {}
            paths['source_mesh'] = self.get_path(subj, 'RAW', 'mesh')
            paths['source_mask'] = self.get_path(subj, 'RAW', 'mask')

            paths['surface_mesh'] = self.get_path(subj, variant, 'mesh', mesh_tag='surface')
            paths['binary_mask']  = self.get_path(subj, variant, 'mask', mask_tag='binary')

            paths['region_mask'] = self.get_path(subj, variant, 'mask', mask_tag='regions')
            paths['volume_mesh'] = self.get_path(subj, variant, 'mesh', mesh_tag='volume')

            paths['material_mask'] = self.get_path(subj, variant, 'mask', mask_tag='material')
            paths['density_field'] = self.get_path(subj, variant, 'field', field_tag='density')
            paths['elastic_field'] = self.get_path(subj, variant, 'field', field_tag='elasticity')

            paths['node_values'] = self.get_path(subj, variant, 'mesh',  mesh_tag='node_values')
            paths['disp_field']  = self.get_path(subj, variant, 'field', field_tag='displacement')
            paths['input_image'] = self.get_path(subj, variant, 'image', image_tag='generated')

            yield base.Example(
                dataset='ShapeNet',
                subject=subj,
                variant=variant,
                paths=paths,
                metadata=dict(metadata.loc[subj])
            )


class ShapeNetResolver(trimesh.resolvers.Resolver):
    '''
    Trimesh path resolver for ShapeNet dataset.
    '''
    def __init__(self, data_root):
        self.root = pathlib.Path(data_root)
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


