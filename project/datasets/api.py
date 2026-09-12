# datasets/api.py

from ..common import utils


def get_subclass(name: str):
    from .base import Dataset
    return Dataset.get_subclass(name)


def get_dataset(name: str, root: str):
    dataset_cls = get_subclass(name)
    return dataset_cls(root)


def get_examples(name: str, root: str, **kwargs):
    dataset = get_dataset(name, root)
    return dataset.list_examples(**kwargs)


def load_example(ex, **kwargs):
    from .torch import TorchDataset
    return TorchDataset([ex], **kwargs)[0]

