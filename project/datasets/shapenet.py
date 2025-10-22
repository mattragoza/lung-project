from typing import Optional, Any, List, Dict, Tuple, Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from . import base


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
        self.root = Path(data_root)

    def load_metadata(self):
        self.metadata = pd.read_csv(self.root / 'metadata.csv')

    def load_categories(self):
        self.categories = pd.read_csv(self.root / 'categories.synset.csv')

    def load_materials(self):
        self.materials = pd.read_csv(self.root / 'materials.csv')

    def load_properties(self):
        self.properties = pd.read_csv(self.root / 'densities.csv')

    def load_taxonomy(self):
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
            return variant_dir / 'meshes' / f'{id_code}_{mesh_tag}.xdmf'

        elif asset_type == 'mask':
            if variant == 'RAW':
                mask_root = self.root / 'models-binvox-solid'
                return mask_root / f'{id_code}.binvox'

            mask_tag = selectors['mask_tag']
            return variant_dir / 'masks' / f'{id_code}_{mask_tag}.nii.gz'

        elif asset_type == 'image':
            if variant == 'RAW':
                raise ValueError('no RAW image data')
            image_tag = selectors['image_tag']
            return variant_dir / 'images' / f'{id_code}_{image_tag}.nii.gz'

        return None

    def examples(
        self,
        subjects: Optional[List[str]] = None,
        variant:  Optional[str] = None,
        source_variant: str = 'RAW'
    ):
        subjects = subjects or self.subjects()
        metadata = self.metadata.set_index(self.ID_COLUMN)
        for subj in subjects:

            paths = {}
            paths['source_mesh'] = self.get_path(subj, source_variant, 'mesh')
            paths['source_mask'] = self.get_path(subj, source_variant, 'mask')
            paths['object_mesh'] = self.get_path(subj, variant, 'mesh', mesh_tag='object')
            paths['binary_mask'] = self.get_path(subj, variant, 'mask', mask_tag='binary')
            paths['region_mask'] = self.get_path(subj, variant, 'mask', mask_tag='regions')
            paths['volume_mesh'] = self.get_path(subj, variant, 'mesh', mesh_tag='volume')

            #paths['input_image'] = self.get_path(subj, variant, 'image')
            #paths['elast_field'] = self.get_path(subj, variant, 'elast')
            #paths['disp_field']  = self.get_path(subj, variant, 'disp')

            yield base.Example(
                dataset='ShapeNet',
                subject=subj,
                variant=variant,
                paths=paths,
                metadata=dict(metadata.loc[subj])
            )



def _parse_id_code(full_id):
    parts = full_id.split('.')
    if len(parts) == 2:
        return parts[1]
    raise RuntimeError(f'failed to parse subject code: {full_id}')

