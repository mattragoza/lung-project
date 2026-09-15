# api.py

from .common import utils


def get_config(argv: list) -> dict:
    from . import cli
    config = cli.resolve_config(argv)
    utils.pprint(config, max_depth=4)
    return config


def get_examples(config: dict) -> list:
    from . import datasets
    utils.log('Gathering examples:', end=' ')
    examples = datasets.get_examples(**config)
    utils.log(f'{len(examples)} total')
    return examples


def run_preprocess(examples: list, config: dict):
    from . import preprocessing

    N = len(examples)
    for idx, ex in enumerate(examples):
        utils.log(f'[Example {idx}/{N}] Preprocessing subject: {ex.subject}')
        preprocessing.preprocess_example(ex, config)

    utils.log(f'[Example {N}/{N}] Done')


def run_optimize(examples: list, config: dict):
    from . import optimization

    N = len(examples)
    for idx, ex in enumerate(examples):
        utils.log(f'[Example {idx}/{N}] Optimizing example: {ex.subject}')
        optimization.optimize_example(ex, config)

    utils.log(f'[Example {N}/{N}] Done')


def run_training(examples: list, config: dict):
    from . import training
    training.run_training(examples, config)
    utils.log('Done')

