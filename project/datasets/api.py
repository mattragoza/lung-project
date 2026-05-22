from . import base

from ..core import utils


def get_dataset(config):
    dataset_cls = base.Dataset.get_subclass(config['name'])
    return dataset_cls(config['root'])


def get_subjects(config):
    raise NotImplementedError

