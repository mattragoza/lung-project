# training/api.py

from ..core import utils

from . import splits, tasks, trainer


def run_training(examples, config, outputs):
    utils.check_keys(
        config,
        {'seed', 'split', 'transform', 'loader'} |
        {'task', 'model', 'optimizer', 'trainer'} |
        {'pde_solver', 'physics_adapter', 'boundary_condition'},
        where='training'
    )
    import torch

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(config.get('seed', 1234))

    from .. import datasets, models, physics, evaluation, callbacks

    split_kws = config.get('split', {})
    train_ex, test_ex, val_ex = splits.split_on_metadata(examples, **split_kws)

    transform_kws = config.get('transform', {})
    train_set = datasets.TorchDataset(train_ex, **transform_kws)
    test_set  = datasets.TorchDataset(test_ex, **transform_kws)
    val_set   = datasets.TorchDataset(val_ex, **transform_kws)

    def _create_data_loader(dataset, **kwargs):
        if len(dataset) > 0:
            return torch.utils.data.DataLoader(dataset, **kwargs)

    loader_kws = config.get('loader', {}).copy()
    loader_kws['collate_fn'] = datasets.collate_fn

    train_loader = _create_data_loader(train_set, **loader_kws)
    test_loader  = _create_data_loader(test_set, **loader_kws)
    val_loader   = _create_data_loader(val_set, **loader_kws)

    task_kws = config.get('task', {})
    task = tasks.TaskSpec(**task_kws)

    model_kws = config.get('model', {})
    model = models.build_model(task, model_kws)
    utils.log(models.count_params(model))

    optimizer_kws = config.get('optimizer', {}).copy()
    optimizer_cls = getattr(torch.optim, optimizer_kws.pop('_class'))
    optimizer = optimizer_cls(model.parameters(), **optimizer_kws)

    solver  = physics.get_solver(config.get('pde_solver', {}))
    adapter = physics.get_adapter(solver, config.get('physics_adapter', {}))
    bc_spec = physics.get_bc_spec(config.get('boundary_condition', {}))

    evaluator_obj = evaluation.Evaluator()
    callback_list = [
        callbacks.LoggerCallback(),
        callbacks.EvaluatorCallback(evaluator_obj),
    ]

    trainer_kws = config.get('trainer', {})
    trainer_obj = trainer.Trainer(
        task, model, optimizer, adapter,
        train_loader=train_loader,
        test_loader=test_loader,
        val_loader=val_loader,
        callbacks=callback_list,
        bc_spec=bc_spec,
        output_dir=outputs.base_dir / 'checkpoints'
    )

    try:
        trainer_obj.load_checkpoint(by_mtime=True)
    except FileNotFoundError as e:
        utils.warn(f'WARNING: {e}')

    trainer_obj.train(**trainer_kws)

