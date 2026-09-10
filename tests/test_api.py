import sys, os, pytest

sys.path.insert(0, os.environ['LP'])
from project import api, core


@pytest.fixture
def config():
    return core.fileio.load_config('config.yaml')


def test_preprocess(config):
    examples = api.get_examples(config['dataset'])[:1]
    api.run_preprocess(examples, config['preprocessing'])


def test_optimize(config):
    examples = api.get_examples(config['dataset'])[:1]
    api.run_optimize(examples, config['optimization'])


def test_training(config):
    examples = api.get_examples(config['dataset'])[:1]
    api.run_training(examples, config['training'])

