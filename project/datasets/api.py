from . import base

from ..core import utils


def get_subjects(config):
    raise NotImplementedError


def get_subclass(config):
    return base.Dataset.get_subclass(config['name'])


def get_dataset(config):
    dataset_cls = get_subclass(config)
    return dataset_cls(config['root'])


def load_example(ex, **kwargs):
    from . import torch
    return torch.TorchDataset([ex], **kwargs)[0]

