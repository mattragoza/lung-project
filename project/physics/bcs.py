from typing import Any
from dataclasses import dataclass


@dataclass(frozen=True)
class BoundaryConditionSpec:
    type: str
    value: Any

