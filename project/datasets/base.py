from typing import Optional, Any, Dict, List, Tuple, Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Example:
    dataset:  str
    subject:  str
    variant:  str
    paths:    Dict[str, Path]
    metadata: Dict[str, Any]


class BaseDataset:

    def subjects(self, *args, **kwargs) -> List[str]:
        raise NotImplementedError

    def variants(self, *args, **kwargs) -> List[str]:
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
        variant: Optional[str]=None,
        **kwargs
    ) -> Iterable[Example]:
        raise NotImplementedError

