from typing import Optional, Any, List, Dict, Tuple, Iterable
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from . import base
from ..core import utils


def _parse_sid_code(subj: str) -> str:
    parts = subj.split('.')
    if len(parts) == 2 and parts[0] == 'wss':
        return parts[1]
    raise RuntimeError(f'failed to parse subject ID code: {subj:r}')


def _parse_category(s: str) -> List[str]:
    return [c.strip() for c in str(s).split(',')]


def _parse_aligned_dims(s: str) -> np.ndarray:
    parts = s.replace('\\,', ',').split(',')
    return np.array([float(p) for p in parts])


def _parse_unit(s: str) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return np.nan


def _resolve_unit(s: str, default: float, policy: str) -> float:
    assert policy in {'require_metadata', 'prefer_metadata', 'force_default'}
    if policy == 'force_default':
        return float(default)
    unit = _parse_unit(s)
    if np.isfinite(unit) and unit > 0:
        return float(unit)
    if policy == 'require_metadata':
        raise ValueError('missing or invalid unit encountered in metadata')
    return float(default)


DEFAULT_TAGS = {
    'binary_mask':   'mask',
    'surface_mesh':  'trimesh',
    'region_mask':   'regions',
    'volume_mesh':   'tetmesh',
    'material_mask': 'material',
    'density_field': 'density',
    'elastic_field': 'elastic',
    'input_image':   'image',
    'material_mesh': 'mat',
    'simulate_mesh': 'sim',
    'interp_mesh':   'int',
}

DEPENDENCY_GRAPH = {
    'binary_mask':   [],
    'surface_mesh':  [],
    'region_mask':   ['binary_mask'],
    'volume_mesh':   ['region_mask'],
    'material_mask': ['region_mask'],
    'density_field': ['material_mask'],
    'elastic_field': ['material_mask'],
    'input_image':   ['material_mask'],
    'material_mesh': ['material_mask', 'volume_mesh'],
    'interp_mesh':   ['input_image', 'material_mesh'],
    'simulate_mesh': ['material_mesh'],
}

def _resolve_dependencies(node, graph):
    deps, visited = [], set()
    def _visit(n):
        for d in graph.get(n, []):
            if d not in visited:
                _visit(d)
                visited.add(d)
                deps.append(d)
    _visit(node)
    return deps


def _resolve_names_from_tags(tags, sep='_'):
    tags = utils.update_defaults(tags, **DEFAULT_TAGS)
    names = {}
    for role, tag in tags.items():
        deps = _resolve_dependencies(role, DEPENDENCY_GRAPH)
        names[role] = sep.join([tags[d] for d in deps] + [tag])
    return names


class ShapeNetDataset(base.Dataset):
    '''
    <data_root>/
        models-COLLADA/
        models-OBJ/models/
            <sid_code>.obj
            <sid_code>.mtl
        models-binvox-solid/
            <sid_code>.binvox
        models-textures/
            <tid_code>.jpg
        <variant>/
            <sid_code>/
                masks/<mask_tag>.nii.gz
                meshes/<mesh_tag>.xdmf
                images/<image_tag>.nii.gz
                fields/<field_tag>.nii.gz
        metadata.csv
        materials.csv
        densities.csv
        taxonomy.txt
        categories.synset.csv
    '''
    ID_COLUMN = 'fullId'

    def __init__(self, root: str|Path):
        self.root = Path(root)
        if not self.root.is_dir():
            raise RuntimeError(f'Invalid directory: {root}')
        self._metadata_loaded = False

    def ensure_metadata(self):
        if not self._metadata_loaded:
            self.load_metadata()

    def load_metadata(self):
        import pandas as pd
        self.metadata   = pd.read_csv(self.root / 'metadata.csv')
        self.categories = pd.read_csv(self.root / 'categories.synset.csv')
        self.materials  = pd.read_csv(self.root / 'materials.csv')
        self.densities  = pd.read_csv(self.root / 'densities.csv')
        self.taxonomy = load_taxonomy(self.root / 'taxonomy.txt')
        self._metadata_loaded = True

    def subjects(self) -> Iterable[str]:
        self.ensure_metadata()
        return self.metadata[self.ID_COLUMN].to_list()

    def path(self, subject: str, variant: str, asset_type: str, name: Optional[str]=None) -> Path:
        sid_code = _parse_sid_code(subject)

        if not variant: # source paths ignore name arg
            if asset_type == 'mesh':
                return self.root / 'models-OBJ' / 'models' / f'{sid_code}.obj'
            elif asset_type == 'mask':
                return self.root / 'models-binvox-solid'/ f'{sid_code}.binvox'
            else:
                raise RuntimeError(f'Invalid source asset type: {asset_type:r}')

        elif name:
            base_dir = self.root / variant / sid_code
            if asset_type == 'mesh':
                return base_dir / 'meshes' / f'{name}.xdmf'
            elif asset_type == 'mask':
                return base_dir / 'masks' / f'{name}.nii.gz'
            elif asset_type == 'field':
                return base_dir / 'fields' / f'{name}.nii.gz'
            elif asset_type == 'image':
                return base_dir / 'images' / f'{name}.nii.gz'
            else:
                raise RuntimeError(f'Invalid variant asset type: {asset_type:r}')
        else:
            raise ValueError(f'Variant paths require a name')

    def examples(
        self,
        subjects: Optional[str|Path|List[str]]=None,
        variant: Optional[str]=None,
        parse_metadata: bool=None,
        unit_policy: str='prefer_metadata',
        default_unit: float=1e-2,
        selectors: Dict[str, str]=None
    ):
        from .base import _resolve_subject_list
        subject_iter = subjects or self.subjects()
        subject_list = _resolve_subject_list(subject_iter)

        if variant is not None:
            variant = str(variant)

        self.ensure_metadata()
        md = self.metadata.set_index(self.ID_COLUMN)
        if parse_metadata is None:
            parse_metadata = (variant is not None)

        names = _resolve_names_from_tags(selectors)

        for s in subject_list:
            paths = {}
            paths['source_mesh'] = self.path(s, variant=None, asset_type='mesh')
            paths['source_mask'] = self.path(s, variant=None, asset_type='mask')

            metadata = {'raw': dict(md.loc[s])}
            if parse_metadata:
                metadata['category'] = _parse_category(md.loc[s, 'category'])
                metadata['dims'] = _parse_aligned_dims(md.loc[s, 'aligned.dims'])
                metadata['unit'] = _resolve_unit(md.loc[s, 'unit'], default_unit, unit_policy)

            if variant: # assets generated by preprocessing
                v = variant
                paths['binary_mask']   = self.path(s, v, 'mask', name=names['binary_mask'])
                paths['region_mask']   = self.path(s, v, 'mask', name=names['region_mask'])
                paths['material_mask'] = self.path(s, v, 'mask', name=names['material_mask'])

                paths['surface_mesh']  = self.path(s, v, 'mesh', name=names['surface_mesh'])
                paths['volume_mesh']   = self.path(s, v, 'mesh', name=names['volume_mesh'])
                paths['material_mesh'] = self.path(s, v, 'mesh', name=names['material_mesh'])
                paths['simulate_mesh'] = self.path(s, v, 'mesh', name=names['simulate_mesh'])
                paths['interp_mesh']   = self.path(s, v, 'mesh', name=names['interp_mesh'])

                paths['input_image']   = self.path(s, v, 'image', name=names['input_image'])
                paths['density_field'] = self.path(s, v, 'field', name=names['density_field'])
                paths['elastic_field'] = self.path(s, v, 'field', name=names['elastic_field'])

            yield base.Example(
                dataset='ShapeNet',
                variant=variant,
                subject=s,
                paths=paths,
                metadata=metadata
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

