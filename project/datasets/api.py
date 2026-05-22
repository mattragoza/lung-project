from . import base

from ..core import utils


def get_dataset(config):
    dataset_cls = base.Dataset.get_subclass(config['name'])
    return dataset_cls(config['root'])


def get_subjects(config):
    raise NotImplementedError


def get_examples(config):
    utils.check_keys(
        config,
        valid={'name', 'root', 'examples', 'selectors'},
        where='dataset'
    )
    from . import datasets

    utils.log('Gathering examples')
    dataset = get_dataset(config)

    example_kws = config.get('examples', {})
    return dataset.list_examples(**example_kws)

