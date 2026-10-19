# training/api.py

from ..common import utils, outputs

from . import splits, tasks, trainer


def run_training(examples, config, out_name='training'):
    trainer = get_trainer(examples, config, out_name)
    return trainer.train(**config.get('trainer', {}))


def get_trainer(examples, config, out_name='training'):
    utils.check_keys(
        config,
        {'seed', 'split', 'transform', 'loaders'} |
        {'task', 'model', 'optimizer', 'trainer'} |
        {'pde_solver', 'physics_adapter', 'boundary_condition'},
        where='training'
    )
    import torch

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    seed = config.get('seed', 0)
    torch.manual_seed(seed)

    from .. import datasets, models, optimization
    from .. import physics, evaluation, callbacks

    split_kws = config.get('split', {})
    train_ex, test_ex, val_ex = splits.split_data(examples, seed=seed, **split_kws)

    transform_kws = config.get('transform', {}).copy()
    train_set = datasets.TorchDataset(train_ex, **transform_kws)

    transform_kws['do_augment'] = False
    test_set  = datasets.TorchDataset(test_ex, **transform_kws)
    val_set   = datasets.TorchDataset(val_ex, **transform_kws)

    def _get_data_loader(dataset, **kwargs):
        if len(dataset) > 0:
            return torch.utils.data.DataLoader(dataset, **kwargs)

    loader_kws = config.get('loaders', {}).copy()
    loader_kws['collate_fn'] = datasets.collate_fn

    train_loader = _get_data_loader(train_set, **loader_kws)
    test_loader  = _get_data_loader(test_set, **loader_kws)
    val_loader   = _get_data_loader(val_set, **loader_kws)

    task_kws = config.get('task', {})
    task = tasks.TaskSpec(**task_kws)

    model_cfg = config.get('model', {})
    model = models.build_model(task, config=model_cfg)
    utils.log(model)
    utils.log(f'# params: {models.count_params(model)}')

    optimizer_kws = config.get('optimizer', {})
    optimizer = optimization.get_optimizer(model, **optimizer_kws)

    solver  = physics.get_solver(**config.get('pde_solver', {}))
    adapter = physics.get_adapter(solver, **config.get('physics_adapter', {}))
    bc_spec = physics.get_bc_spec(**config.get('boundary_condition', {}))

    out = outputs.Outputs(stage='training')

    trainer_obj = trainer.Trainer(
        task, model, optimizer, adapter,
        train_loader=train_loader,
        test_loader=test_loader,
        val_loader=val_loader,
        evaluator=evaluation.Evaluator(),
        callbacks=[callbacks.LoggingCallback()],
        csv_path=out.csv_path(name='metrics'),
        bc_spec=bc_spec
    )

    try:
        trainer_obj.load_checkpoint(by_mtime=True)
        trainer_obj.load_metrics()

    except FileNotFoundError as e:
        utils.warn(f'WARNING: {e}')

    return trainer_obj

