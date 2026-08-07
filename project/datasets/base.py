from typing import Optional, List, Dict, Iterable, Any
from dataclasses import dataclass
from pathlib import Path


def _resolve_subject_list(subjects: str|Path|Iterable[str], col='subject', sep='\t') -> List[str]:
    from ..core import fileio
    if isinstance(subjects, (str, Path)):
        subjects = str(subjects)
        if subjects.endswith('.csv'):
            return fileio.load_subject_list(subjects, key=col, sep=sep)
        elif subjects.endswith('.txt'):
            return fileio.load_subject_list(subjects, key=0, header=None)
    elif hasattr(subjects, '__iter__'):
        return [str(v) for v in subjects]
    raise TypeError(f'Invalid subject list: {subjects!r}')


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


@dataclass
class Example:
    dataset:  str
    subject:  str
    variant:  str
    paths:    Dict[str, Path]
    metadata: Dict[str, Any]


class Dataset:
    SUBJ_COLUMN = None
    META_SEP = None

    @classmethod
    def get_subclass(cls, name: str):
        return _resolve_dataset_name(name)

    def __init__(self, root: str|Path):
        self.root = Path(root)
        if not self.root.is_dir():
            raise RuntimeError(f'Invalid directory: {root}')
        self._metadata_loaded = False

    def __repr__(self):
        cls = self.__class__
        return f'{cls.__module__}.{cls.__name__}({str(self.root)!r})'

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
        pipeline_tags: Dict[str, str] = None,
        **kwargs
    ) -> Iterable[Example]:
        raise NotImplementedError

    # ----- convenience API wrapper -----

    def list_examples(
        self,
        subjects: str|Path|Iterable[str],
        **kwargs
    ) -> List[Example]:

        subject_list = _resolve_subject_list(
            subjects,
            col=self.SUBJ_COLUMN,
            sep=self.META_SEP
        )
        return list(self.examples(subject_list, **kwargs))

