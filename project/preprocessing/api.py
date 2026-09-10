# preprocessing/api.py


def get_pipeline(name: str):
    from . import pipelines

    key = name.lower()

    if key in {'shapenet'}:
        return pipelines.shapenet

    elif key in {'copdgene'}:
        return pipelines.copdgene

    elif key in {'emory4dct', 'emory-4dct'}:
        return piplines.emory4dct

    elif key in {'bmc4dct', 'bmc-4dct'}:
        return pipelines.bmc4dct

    elif key in {'phantom'}:
        return pipelines.phantom

    raise ValueError(f'Invalid pipeline: {name!r}')


def preprocess_example(ex, config):
    pipeline = get_pipeline(ex.dataset)
    return pipeline.preprocess(ex, config)

