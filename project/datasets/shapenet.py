from typing import Optional, Any, List, Dict, Tuple, Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from . import base


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

    def load_materials(self):
        self.materials = pd.read_csv(self.root / 'materials.csv')

    def load_densities(self):
        self.densities = pd.read_csv(self.root / 'densities.csv')

    def subjects(self) -> List[str]:
        return self.metadata[self.ID_COLUMN].to_list()

    def get_path(
        self,
        subject: str,
        variant: str,
        asset_type: str,
        **selectors
    ):
        id_code = _parse_id_code(subject)

        if asset_type == 'mesh':
            mesh_root = self.root / 'models-OBJ' / 'models'
            return mesh_root / f'{id_code}.obj'

        elif asset_type == 'material':
            mesh_root = self.root / 'models-OBJ' / 'models'
            return mesh_root / f'{id_code}.mtl'

        elif asset_type == 'mask':
            mask_root = self.root / 'models-binvox-solid'
            return mask_root / f'{id_code}.binvox'

        return None

    def examples(
        self,
        subjects: Optional[List[str]] = None,
        variant:  Optional[str] = None,
    ):
        subjects = subjects or self.subjects()
        metadata = self.metadata.set_index(self.ID_COLUMN)
        for subj in subjects:

            paths = {}
            paths['raw_mask'] = self.get_path(subj, variant, 'mask')
            paths['raw_mesh'] = self.get_path(subj, variant, 'mesh')
            paths['material'] = self.get_path(subj, variant, 'material')

            #paths['fixed_image'] = self.get_path(subj, variant, 'image')
            #paths['fixed_mesh']  = self.get_path(subj, variant, 'mesh')
            #paths['disp_field']  = self.get_path(subj, variant, 'disp')
            #paths['elast_field'] = self.get_path(subj, variant, 'elast')

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

