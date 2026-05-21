from .core import paths, utils


def get_config(argv):
    from .core import cli, fileio

    args = cli.parse_args(argv)
    config = fileio.load_config(args.config)
    config = cli.apply_overrides(config, args.set)

    utils.pprint(config, 4, 20)
    return config


def get_examples(config):
    utils.check_keys(
        config,
        valid={'name', 'root', 'examples', 'selectors'},
        where='dataset'
    )
    from . import datasets

    dataset_name = config['name'] # required
    dataset_root = config['root'] # required
    examples_kws = config.get('examples', {})
    selector_kws = config.get('selectors', {})

    utils.log('Gathering examples')
    dataset_cls = datasets.base.Dataset.get_subclass(dataset_name)
    dataset = dataset_cls(dataset_root)
    return dataset.list_examples(selectors=selector_kws, **examples_kws)


def run_validate(examples, config):
    from . import validation

    config = config.copy()
    outputs = paths.RunOutputs(stage='validate', **config.pop('outputs', {}))

    rows = []
    for ex in examples:
        utils.log(f'Validating subject: {ex.subject}')
        try:
            result = validation.validate_example(ex)
            if result is not None:
                rows.append(result)

        except Exception as e:
            utils.log(f'ERROR: {e}; Skipping subject {ex.subject}')
            continue

    if rows:
        import pandas as pd
        csv_path = outputs.csv_path(name='metrics')
        paths.ensure_parent_directory(csv_path)
        df = pd.DataFrame(rows).to_csv(csv_path, index=False)

    utils.log('Done')


def run_preprocess(examples, config):
    from . import preprocessing

    config = config.copy()
    outputs = paths.RunOutputs(stage='preprocess', **config.pop('outputs', {}))

    rows = []
    for ex in examples:
        utils.log(f'Preprocessing subject: {ex.subject}')
        try:
            result = preprocessing.api.preprocess_example(ex, config)
            if result is not None:
                rows.append(result)

        except Exception as e:
            utils.log(f'ERROR: {e}; Skipping subject {ex.subject}')
            raise e

    if rows:
        import pandas as pd
        csv_path = outputs.csv_path(name='metrics')
        paths.ensure_parent_directory(csv_path)
        df = pd.DataFrame(rows).to_csv(csv_path, index=False)

    utils.log('Done')


def run_optimize(examples, config):
    from . import optimization

    config = config.copy()
    outputs = paths.RunOutputs(stage='optimize', **config.pop('outputs', {}))

    all_metrics = []
    failed_sids = []

    for ex in examples:
        utils.log(f'Optimizing subject: {ex.subject}')
        try:
            metrics = optimization.optimize_example(ex, config, outputs)
            if metrics is not None:
                all_metrics.append(metrics)

        except Exception as e:
            utils.log(f'ERROR: {e}; Skipping subject {ex.subject}')
            failed_sids.append(ex.subject)
            if len(examples) == 1:
                raise

    if all_metrics:
        import pandas as pd
        csv_path = outputs.csv_path(name='metrics')
        paths.ensure_parent_directory(csv_path)
        df = pd.concat(all_metrics).to_csv(csv_path, index=False)

    utils.log('Done')
    utils.log(f'Failed subjects: {failed_sids}')


def run_training(examples, config):
    from . import training

    config = config.copy()
    outputs = paths.RunOutputs(stage='training', **config.pop('outputs', {}))

    training.api.run_training(examples, config, outputs)

    utils.log('Done')

