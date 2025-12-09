from .core import utils


class RunOutputs:

    def __init__(self, stage: str, root: str='outputs'):
        from pathlib import Path
        self.root = Path(root)
        self.stage = str(stage)

    @property
    def base_dir(self):
        return self.root / self.stage

    def csv_path(self, name):
        return self.base_dir / (name + '.csv')

    def mesh_path(self, ex, name):
        return self.base_dir / ex.subject / 'meshes' / (name + '.xdmf')


def get_examples(config):
    utils.check_keys(
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


def run_validate(examples, config):
    from . import validation

    config = config.copy()
    outputs = RunOutputs(stage='validate', **config.pop('outputs', {}))

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
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows).to_csv(csv_path, index=False)


def run_preprocess(examples, config):
    from . import preprocessing

    config = config.copy()
    outputs = RunOutputs(stage='preprocess', **config.pop('outputs', {}))

    rows = []
    for ex in examples:
        utils.log(f'Preprocessing subject: {ex.subject}')
        try:
            result = preprocessing.api.preprocess_example(ex, config)
            if result is not None:
                rows.append(result)
        except Exception as e:
            utils.log(f'ERROR: {e}; Skipping subject {ex.subject}')
            continue

    if rows:
        import pandas as pd
        csv_path = outputs.csv_path(name='metrics')
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows).to_csv(csv_path, index=False)


def run_optimize(examples, config):
    from . import optimization

    config = config.copy()
    outputs = RunOutputs(stage='optimize', **config.pop('outputs', {}))

    all_metrics = []
    for ex in examples:
        utils.log(f'Optimizing subject: {ex.subject}')
        try:
            output_path = outputs.mesh_path(ex, name='optimized')
            metrics = optimization.optimize_example(ex, config, output_path)
            if metrics is not None:
                all_metrics.append(metrics)
        except Exception as e:
            utils.log(f'ERROR: {e}; Skipping subject {ex.subject}')
            raise

    if all_metrics:
        import pandas as pd
        csv_path = outputs.csv_path(name='metrics')
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.concat(all_metrics).to_csv(csv_path, index=False)


def run_training(examples, config):
    utils.check_keys(
        config,
        {'split', 'loader', 'model', 'optimizer', 'evaluator'} |
        {'physics_adapter', 'pde_solver', 'trainer'},
        where='training'
    )
    from . import datasets, models, training, evaluation, physics
    import torch

    config = config.copy()
    outputs = RunOutputs(stage='training', **config.pop('outputs', {}))

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
    logger = evaluation.Logger()
    plotter = evaluation.Plotter()
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
        callbacks=[logger, plotter, evaluator],
        physics_adapter=physics_adapter,
        supervised=supervised
    )
    trainer.train(**trainer_kws)

