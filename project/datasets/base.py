from __future__ import annotations
from typing import Optional, Any, Dict, List, Tuple, Iterable
from dataclasses import dataclass
from pathlib import Path


def _resolve_dataset_name(name: str):
    n = name.lower()
    if n in {'shapenet', 'shapenetsem'}:
        from . import shapenet
        return shapenet.ShapeNetDataset
    elif n in {'copdgene'}:
        from . import copdgene
        return copdgene.COPDGeneDataset
    elif n in {'emory4dct', 'emory-4dct', 'dirlab'}:
        from . import emory4dct
        return emory4dct.Emory4DCTDataset
    elif n in {'bmc4dct', 'bmc-4dct', '4d_lungs'}:
        from . import bmc4dct
        return bmc4dct.BMC4DCTDataset
    raise ValueError(f'Invalid dataset name: {name!r}')


def _resolve_subject_list(val: str|Path|List[str]) -> List[str]:
    from ..core import fileio
    if isinstance(val, Path):
        return fileio.load_subject_list(val)
    elif isinstance(val, str):
        if val.endswith('.csv'):
            return fileio.load_subject_list(val, key='subject')
        elif val.endswith('.txt'):
            return fileio.load_subject_list(val, key=0, header=None)
        return val.split(',')
    elif hasattr(val, '__iter__'):
        return [str(v) for v in val]
    return [str(val)]


@dataclass
class Example:
    dataset:  str
    subject:  str
    variant:  str
    paths:    Dict[str, Path]
    metadata: Dict[str, Any]


class Dataset:

    @classmethod
    def get_subclass(cls, name: str):
        return _resolve_dataset_name(name)

    def __init__(self, root: str|Path):
        self.root = Path(root)
        if not self.root.is_dir():
            raise RuntimeError(f'Invalid directory: {root}')
        self._metadata_loaded = False

    def require_metadata(self):
        if not self._metadata_loaded:
            self.load_metadata()

    def load_metadata(self):
        raise NotImplementedError

    def subjects(self, *args, **kwargs) -> Iterable[str]:
        raise NotImplementedError

    def variants(self, *args, **kwargs) -> Iterable[str]:
        raise NotImplementedError

    def source_path(self, subject: str, *, asset_type: str) -> Path:
        raise NotImplementedError

    def derived_path(
        self,
        subject: str,
        variant: str,
        asset_type: str,
        asset_name: str
    ) -> Path:
        raise NotImplementedError

    def examples(
        self,
        subjects: Optional[List[str]] = None,
        variant:  Optional[str] = None,
        *, # source selectors (e.g. visit, state, etc.)
        selectors: Dict[str, str] = None, # pipeline tags
        **kwargs
    ) -> Iterable[Example]:
        raise NotImplementedError

    def list_examples(self, *args, **kwargs) -> List[Example]:
        return list(self.examples(*args, **kwargs))

