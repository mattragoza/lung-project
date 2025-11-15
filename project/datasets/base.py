from __future__ import annotations
from typing import Optional, Any, Dict, List, Tuple, Iterable
from dataclasses import dataclass
from pathlib import Path


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


@dataclass
class DataSubset:
    dataset:  Dataset
    subjects: List[str]
    variant:  Optional[str]

    def examples(self, **kwargs) -> Iterable[Example]:
        return self.dataset.examples(self.subjects, self.variant, **kwargs)

    def list_examples(self, **kwargs) -> List[Example]:
        return list(self.examples(**kwargs))


class Dataset:

    @classmethod
    def get_subclass(cls, name: str):
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
        raise ValueError('Invalid dataset name: {name:r}')

    def get_subset(self, subjects: str|Path|List[str], variant: str) -> DataSubset:
        return DataSubset(self, subjects=subjects, variant=variant)

    def __init__(self, root: Path):
        raise NotImplementedError

    def subjects(self, *args, **kwargs) -> Iterable[str]:
        raise NotImplementedError

    def variants(self, *args, **kwargs) -> Iterable[str]:
        raise NotImplementedError

    def path(
        self,
        subject: str,
        variant: str,
        *, # all args after this are keyword-only
        asset_type: str,
        **selectors
    ) -> Path:
        raise NotImplementedError

    def examples(
        self,
        subjects: Optional[List[str]]=None,
        variant:  Optional[str]=None,
        **kwargs
    ) -> Iterable[Example]:
        raise NotImplementedError

    def list_examples(self, *args, **kwargs) -> List[Example]:
        return list(self.examples(*args, **kwargs))

