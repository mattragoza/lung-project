from typing import Optional, Any, Dict, List, Tuple, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class Example:

    dataset:  str
    subject:  str
    variant:  Optional[str] = None
    visit:    Optional[str] = None
    fixed_state:  Optional[str] = None
    moving_state: Optional[str] = None
    paths:    Dict[str, Path] = None
    metadata: Dict[str, Any]  = None


class BaseDataset:

    def subjects(self) -> List[str]:
        raise NotImplementedError

    def visits(self, subject: str) -> List[str]:
        raise NotImplementedError

    def variants(self, subject: str, visit: str) -> List[str]:
        raise NotImplementedError

    def states(self, subject: str, visit: str) -> List[str]:
        raise NotImplementedError

    def state_pairs(self, subject: str, visit: str) -> List[Tuple[str, str]]:
        from itertools import permutations
        states = self.states(subject, visit)
        return list(permutations(states, 2))

    def get_path(
        self,
        subject: str,
        variant: str,
        visit: str,
        state: str,
        asset_type: str,
        **selectors
    ) -> Path:
        raise NotImplementedError

    def examples(self, *args, **kwargs) -> Iterable[Example]:
        raise NotImplementedError

