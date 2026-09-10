# training/api.py

from ..core import utils

from . import splits, tasks, trainer


def run_training(examples, config, outputs):
    utils.check_keys(
        config,
        {'split', 'transform', 'loader', 'model', 'optimizer', 'evaluator'} |
        {'physics_adapter', 'pde_solver', 'trainer', 'task', 'random_seed'}
        where='training'
    )
    import torch

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(config.get('random_seed', 1234))

    from .. import datasets, models, physics, evaluation, callbacks

    split_kws = config.get('split', {})
    train_ex, test_ex, val_ex = splits.split_on_metadata(examples, **split_kws)

    transform_kws = config.get('transform', {})
    train_set = datasets.torch.TorchDataset(train_ex, **transform_kws)
    test_set  = datasets.torch.TorchDataset(test_ex, **transform_kws)
    val_set   = datasets.torch.TorchDataset(val_ex, **transform_kws)

    def _create_data_loader(dataset, **kwargs):
        if len(dataset) > 0:
            return torch.utils.data.DataLoader(dataset, **kwargs)

    loader_kws = config.get('loader', {}).copy()
    loader_kws['collate_fn'] = datasets.torch.collate_fn

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

    adapter = physics.api.get_adapter(config)
    bc_spec = physics.api.get_bc_spec(config)

    evaluator_kws = config.get('evaluator', {})
    evaluator = evaluation.Evaluator(**evaluator_kws)

    callback_list = [
        callbacks.LoggerCallback(),
        callbacks.EvaluatorCallback(evaluator),
    ]

    trainer_kws = config.get('trainer', {}).copy()
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


def attach_pseudo_labels(examples, path_list):
    import dataclasses
    import pandas as pd

    def read_paths(path_list):
        from pathlib import Path
        paths, subjs = [], []
        with open(path_list) as f:
            for line in f:
                p = line.strip()
                paths.append(Path(p))
                subjs.append(p.split('/')[-3])
        return dict(zip(subjs, paths))

    paths_by_subject = read_paths(path_list)

    out = []
    for ex in examples:
        ex = dataclasses.replace(ex)
        ex.paths = dict(ex.paths)
        ex.paths['elastic_pseudo'] = paths_by_subject[ex.subject]
        if ex.paths['elastic_pseudo'].is_file():
            out.append(ex)
        else:
            utils.warn(f'WARNING: No pseudo-label for example {ex.subject}')
    return out

