from dataclasses import dataclass


@dataclass(frozen=True)
class BoundaryConditionSpec:
    name: str

