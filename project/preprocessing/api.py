# preprocessing/api.py

from .pipelines import shapenet, copdgene, emory4dct, bmc4dct, phantom


PIPELINE_REGISTRY = {
    'shapenet': shapenet,
    'copdgene': copdgene,
    'emory4dct': emory4dct,
    'emory-4dct': emory4dct,
    'bmc4dct': bmc4dct,
    'bmc-4dct': bmc4dct,
    'phantom': phantom
}


def preprocess_example(ex, config):
    key = ex.dataset.lower()

    try:
        pipeline = PIPELINE_REGISTRY[key]
    except KeyError:
        raise ValueError(f'Invalid dataset: {ex.dataset!r}')

    return pipeline.preprocess(ex, config)

