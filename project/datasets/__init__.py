# datasets/__init__.py

from . import api, base, torch

from .api import (
    get_subclass,
    get_dataset,
    get_examples,
    load_example
)

from .base import Example, Dataset
from .torch import TorchDataset, collate_fn

