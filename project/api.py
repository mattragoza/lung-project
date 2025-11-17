from .core import utils, fileio


def _check_keys(config, valid, where=None):
    invalid = set(config.keys()) - set(valid)
    if invalid:
        _loc = f' for {where}' if where else ''
        raise KeyError(f'Unexpected keys{_loc}: {invalid} vs. {valid}')


def get_examples(config):
    _check_keys(
        config,
        valid={'name', 'root', 'examples', 'metadata', 'selectors'},
        where='dataset'
    )
    from . import datasets

    config = config.copy()
    dataset_name = config.pop('name')
    dataset_root = config.pop('root')
    examples_kws = config.pop('examples')
    metadata_kws = config.pop('metadata')
    selector_kws = config.pop('selectors')

    utils.log('Gathering examples')
    dataset_cls = datasets.base.Dataset.get_subclass(dataset_name)
    dataset = dataset_cls(dataset_root).get_subset(**examples_kws)
    return dataset.list_examples(selectors=selector_kws, **metadata_kws)


def run_validate(examples, output):
    from . import validation
    rows = []
    for ex in examples:
        utils.log('Validating subject {ex.subject}')
        try:
            reuslt = validation.validate_example(ex)
            rows.append(result)
        except Exception as e:
            utils.log(f'ERROR: {e}; Skipping subject {ex.subject}')
            continue
    df = pd.DataFrame(rows)
    if output:
        df.to_csv(output, index=False)


def run_preprocess(examples, config):
    # config keys are validated in preprocessing.api pipelines
    from . import preprocessing
    for ex in examples:
        utils.log(f'Preprocessing subject {ex.subject}')
        try:
            preprocessing.api.preprocess_example(ex, config)
        except Exception as e:
            utils.log(f'ERROR: {e}; Skipping subject {ex.subject}')
            raise


def run_optimize(examples, config, output):
    from . import optimization
    import pandas as pd
    rows = []
    for ex in examples:
        utils.log(f'Optimizing subject {ex.subject}')
        try:
            result = optimization.optimize_example(ex, config)
            rows.append({
                'dataset': ex.dataset,
                'subject': ex.subject,
                'variant': ex.variant,
                'method':  'optimize',
                **result
            })
        except Exception as e:
            utils.log(f'ERROR: {e}; Skipping subject {ex.subject}')
            raise
    df = pd.DataFrame(rows)
    if output:
        df.to_csv(output, index=False)


def run_training(examples, config):
    _check_keys(
        config,
        {'split', 'loader', 'model', 'optimizer', 'evaluator'} |
        {'physics_adapter', 'pde_solver', 'trainer'},
        where='training'
    )
    from . import datasets, models, training, evaluation, physics
    import torch
    utils.log('Running training')

    split_kws = config.get('split', {})
    train_ex, test_ex, val_ex = training.split_on_metadata(examples, **split_kws)

    train_set  = datasets.torch.TorchDataset(train_ex)
    test_set   = datasets.torch.TorchDataset(test_ex)
    val_set    = datasets.torch.TorchDataset(val_ex)
    collate_fn = datasets.torch.collate_fn

    loader_kws = config.get('loader', {})
    train_loader = torch.utils.data.DataLoader(train_set, collate_fn=collate_fn, **loader_kws)
    test_loader  = torch.utils.data.DataLoader(test_set, collate_fn=collate_fn, **loader_kws)
    val_loader   = torch.utils.data.DataLoader(val_set, collate_fn=collate_fn, **loader_kws)

    model_kws = config.get('model', {}).copy()
    model_cls = getattr(models, model_kws.pop('_class'))
    model = model_cls(in_channels=1, out_channels=1, **model_kws)

    #n_params = models.count_params(model)
    #n_activs = models.count_activations(model, train_set[0]['image'].unsqueeze(0))
    #n_bytes  = (n_params + n_activs) * 2 * 4

    optimizer_kws = config.get('optimizer', {}).copy()
    optimizer_cls = getattr(torch.optim, optimizer_kws.pop('_class'))
    optimizer = optimizer_cls(model.parameters(), **optimizer_kws)

    evaluator_kws = config.get('evaluator', {})
    evaluator = evaluation.Evaluator(**evaluator_kws)

    physics_adapter_kws = config.get('physics_adapter', {})
    pde_solver_kws = config.get('pde_solver', {}).copy()
    pde_solver_cls = pde_solver_kws.pop('_class')

    physics_adapter = physics.PhysicsAdapter(
        pde_solver_cls=pde_solver_cls,
        pde_solver_kws=pde_solver_kws,
        **physics_adapter_kws
    )

    trainer_kws = config.get('trainer', {}).copy()
    supervised = trainer_kws.pop('supervised', False)
    trainer = training.Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        test_loader=test_loader,
        val_loader=val_loader,
        evaluator=evaluator,
        physics_adapter=physics_adapter,
        supervised=supervised
    )
    trainer.train(**trainer_kws)

