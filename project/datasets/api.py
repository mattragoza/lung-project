from typing import List, Iterable


def get_subclass(config):
    from .base import Dataset
    return Dataset.get_subclass(config['name'])


def get_dataset(config):
    dataset_cls = get_subclass(config)
    return dataset_cls(config['root'])


def load_example(ex, **kwargs):
    from .torch import TorchDataset
    return TorchDataset([ex], **kwargs)[0]

