# training/api.py

from pathlib import Path

from ..common import utils, outputs

from . import splits, tasks, trainer


def run_inference(examples, config, output_dir='outputs'):
    import copy
    config = copy.deepcopy(config)
    config['split']['method'] = None
    config['loaders']['batch_size'] = 1
    config['loaders']['shuffle'] = False
    trainer = get_trainer(examples, config, output_dir, use_callbacks=False)
    return trainer.run_phase(
        trainer.train_loader, phase='infer', train_mode=False
    )


def run_training(examples, config, outputs_dir='outputs', run_name='default'):
    trainer = get_trainer(
        examples, config,
        output_dir=output_dir,
        run_name=run_name,
        use_callbacks=True
    )
    return trainer.train(**config.get('trainer', {}))


def get_trainer(
    examples,
    config,
    output_dir='outputs',
    run_name='default',
    use_callbacks=False
):
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

    seed = config.get('seed')
    if seed is not None:
        torch.manual_seed(seed)

    from .. import datasets, models, optimization
    from .. import physics, evaluation, callbacks

    split_kws = config.get('split', {}).copy()
    split_kws.setdefault('seed', seed)

    train_ex, test_ex, val_ex = splits.split_data(examples, **split_kws)

    transform_kws = config.get('transform', {}).copy()
    transform_kws.setdefault('seed', seed)
    augment = transform_kws.pop('do_augment')

    train_set = datasets.TorchDataset(train_ex, do_augment=augment, **transform_kws)
    test_set  = datasets.TorchDataset(test_ex, do_augment=False, **transform_kws)
    val_set   = datasets.TorchDataset(val_ex, do_augment=False, **transform_kws)

    def _get_data_loader(dataset, shuffle, seed=None, **kwargs):
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
        if len(dataset) > 0:
            return torch.utils.data.DataLoader(
                dataset, collate_fn=datasets.collate_fn,
                shuffle=shuffle,
                generator=rng,
                **kwargs
            )

    loader_kws = config.get('loaders', {}).copy()
    loader_kws.setdefault('seed', seed)
    shuffle = loader_kws.pop('shuffle', False)

    train_loader = _get_data_loader(train_set, shuffle, **loader_kws)
    test_loader  = _get_data_loader(test_set, False, **loader_kws)
    val_loader   = _get_data_loader(val_set, False, **loader_kws)

    task_kws = config.get('task', {})
    task = tasks.TaskSpec(**task_kws)

    model_config = config.get('model', {})
    model = models.build_model(task, config=model_config)
    utils.log(f'total params: {models.count_params(model)}')

    optimizer_kws = config.get('optimizer', {})
    optimizer = optimization.get_optimizer(model, **optimizer_kws)

    solver  = physics.get_solver(**config.get('pde_solver', {}))
    adapter = physics.get_adapter(solver, **config.get('physics_adapter', {}))
    bc_spec = physics.get_bc_spec(**config.get('boundary_condition', {}))

    output_dir = Path(output_dir) / 'training' / run_name

    if use_callbacks:
        callback_list = [
            callbacks.LoggingCallback(),
            callbacks.PlotterCallback(
                output_dir, metrics=['loss', 'E_cell.nrmse', 'E_cell.pcorr']
            ),
            callbacks.ViewerCallback(
                output_dir, keys=['image', 'E_pred', 'E_true']
            )
        ]
    else:
        callback_list = []

    trainer_obj = trainer.Trainer(
        task, model, optimizer, adapter,
        train_loader=train_loader,
        test_loader=test_loader,
        val_loader=val_loader,
        evaluator=evaluation.Evaluator(),
        callbacks=callback_list,
        output_dir=output_dir,
        bc_spec=bc_spec
    )

    import pandas as pd
    try:
        trainer_obj.load_checkpoint(by_mtime=True)
        trainer_obj.load_metrics()

    except (FileNotFoundError, pd.errors.EmptyDataError) as e:
        utils.warn(f'WARNING: {e}')

    return trainer_obj

