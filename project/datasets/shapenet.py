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
    raise RuntimeError(f'Failed to parse subject ID code: {subj:r}')


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
        raise ValueError('Missing or invalid unit encountered in metadata')
    return float(default)


DEFAULT_TAGS = {
    'binary_mask':   'mask',
    'surface_mesh':  'trimesh',
    'region_mask':   'regions',
    'volume_mesh':   'tetmesh',
    'material_mask': 'material',
    'density_field': 'density',
    'elastic_field': 'elastic',
    'poisson_field': 'poisson',
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
    'poisson_field': ['material_mask'],
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
        downloads/
            ShapeNetSem.zip
        extracted/
            models-COLLADA/
            models-OBJ/models/
                <sid_code>.obj
                <sid_code>.mtl
            models-binvox-solid/
                <sid_code>.binvox
            models-textures/
                <tid_code>.jpg
            metadata.csv
            materials.csv
            densities.csv
            taxonomy.txt
            categories.synset.csv
        processed/
            <variant>/
                <sid_code>/
                    masks/<mask_name>.nii.gz
                    meshes/<mesh_name>.xdmf
                    images/<image_name>.nii.gz
                    fields/<field_name>.nii.gz

    <subject> = wss.<sid_code>
    '''
    ID_COLUMN = 'fullId'

    def load_metadata(self):
        import pandas as pd
        base_dir = self.root / 'extracted'
        self._categories = pd.read_csv(base_dir / 'categories.synset.csv')
        self._materials = pd.read_csv(base_dir / 'materials.csv')
        self._densities = pd.read_csv(base_dir / 'densities.csv')
        self._taxonomy = load_taxonomy(base_dir / 'taxonomy.txt')
        self._metadata = pd.read_csv(base_dir / 'metadata.csv')
        self._metadata = self._metadata.set_index(self.ID_COLUMN, inplace=True)
        self._metadata_loaded = True

    def subject_metadata(self, subject: str):
        self.require_metadata()
        return self._metadata.loc[subject]

    def subjects(self) -> List[str]:
        self.require_metadata()
        return sorted(self._metadata.index.to_list())

    def source_path(self, subject: str, asset_type: str):
        sid_code = _parse_sid_code(subject)
        base_dir = self.root / 'extracted'

        if asset_type == 'mesh':
            return base_dir / 'models-OBJ/models' / f'{sid_code}.obj'
        elif asset_type == 'mask':
            return base_dir / 'models-binvox-solid'/ f'{sid_code}.binvox'

        raise RuntimeError(f'Invalid source asset type: {asset_type!r}')

    def derived_path(
        self,
        subject: str,
        variant: str,
        asset_type: str,
        asset_name: str
    ):
        sid_code = _parse_sid_code(subject)
        base_dir = self.root / 'processed' / variant / sid_code

        if asset_type == 'mesh':
            return base_dir / 'meshes' / f'{asset_name}.xdmf'
        elif asset_type == 'mask':
            return base_dir / 'masks' / f'{asset_name}.nii.gz'
        elif asset_type == 'field':
            return base_dir / 'fields' / f'{asset_name}.nii.gz'
        elif asset_type == 'image':
            return base_dir / 'images' / f'{asset_name}.nii.gz'

        raise RuntimeError(f'Invalid derived asset type: {asset_type!r}')

    def examples(
        self,
        subjects: Optional[str|Path|List[str]] = None,
        variant:  Optional[str] = None,
        selectors: Dict[str, str] = None,
        parse_metadata: bool = True,
        unit_policy: str = 'prefer_metadata',
        default_unit: float = 1e-2
    ):
        from .base import _resolve_subject_list
        subject_iter = subjects or self.subjects()
        subject_list = _resolve_subject_list(subject_iter)

        if variant is not None:
            variant = str(variant)

        self.require_metadata()
        if parse_metadata is None:
            parse_metadata = (variant is not None)

        names = _resolve_names_from_tags(selectors)

        for sid in subject_list:
            m = self.subject_metadata(sid)

            paths = {}
            paths['source_mesh'] = self.source_path(sid, asset_type='mesh')
            paths['source_mask'] = self.source_path(sid, asset_type='mask')

            meta = {'raw': dict(md.loc[s])}
            if parse_metadata:
                meta['category'] = _parse_category(md.loc[s, 'category'])
                meta['dims'] = _parse_aligned_dims(md.loc[s, 'aligned.dims'])
                meta['unit'] = _resolve_unit(md.loc[s, 'unit'], default_unit, unit_policy)

            if variant: # assets generated by preprocessing
                s, v = sid, variant
                paths['binary_mask']   = self.derived_path(s, v, 'mask', names['binary_mask'])
                paths['region_mask']   = self.derived_path(s, v, 'mask', names['region_mask'])
                paths['material_mask'] = self.derived_path(s, v, 'mask', names['material_mask'])

                paths['surface_mesh']  = self.derived_path(s, v, 'mesh', names['surface_mesh'])
                paths['volume_mesh']   = self.derived_path(s, v, 'mesh', names['volume_mesh'])
                paths['material_mesh'] = self.derived_path(s, v, 'mesh', names['material_mesh'])
                paths['simulate_mesh'] = self.derived_path(s, v, 'mesh', names['simulate_mesh'])
                paths['interp_mesh']   = self.derived_path(s, v, 'mesh', names['interp_mesh'])

                paths['input_image']   = self.derived_path(s, v, 'image', names['input_image'])
                paths['density_field'] = self.derived_path(s, v, 'field', names['density_field'])
                paths['elastic_field'] = self.derived_path(s, v, 'field', names['elastic_field'])

            yield base.Example(
                dataset='ShapeNet',
                variant=variant,
                subject=sid,
                paths=paths,
                metadata=meta
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

