from . import pipelines


def preprocess_example(ex, config):
    dataset = ex.dataset.lower()

    if dataset == 'shapenet':
        return pipelines.preprocess_shapenet(ex, config)

    elif dataset == 'copdgene':
        return pipelines.preprocess_copdgene(ex, config)

    elif dataset in {'emory4dct', 'emory-4dct'}:
        return pipelines.preprocess_emory4dct(ex, config)

    elif dataset in {'bmc4dct', 'bmc-4dct'}:
        return pipelines.preprocess_bmc4dct(ex, config)

    raise ValueError(f'Invalid dataset: {ex.dataset!r}')


