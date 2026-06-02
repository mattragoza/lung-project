from ..core import fileio, utils

from . import splits, tasks
from . import trainer as trainer_module


def run_training(examples, config, outputs):
    utils.check_keys(
        config,
        {'split', 'transform', 'loader', 'model', 'optimizer', 'evaluator'} |
        {'physics_adapter', 'pde_solver', 'trainer', 'task', 'random_seed'} |
        {'use_pseudo', 'pseudo_path'},
        where='training'
    )
    import torch
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(config.get('random_seed'))

    from .. import datasets, models, evaluation, physics
    from .. import callbacks as callbacks_module

    utils.log('Start training')

    config = config.copy()

    use_pseudo = config.get('use_pseudo', False)
    pseudo_path = config.get('pseudo_path', '.')
    if use_pseudo:
        examples = attach_pseudo_labels(examples, pseudo_path)

    split_kws = config.get('split', {})
    train_ex, test_ex, val_ex = splits.split_on_metadata(examples, **split_kws)

    # TODO is there a better way to determine this?
    img = fileio.load_nibabel(examples[0].paths['input_image']).get_fdata()
    rgb = (img.ndim == 4 and img.shape[-1] == 3)

    transform_kws = config.get('transform', {})
    train_set  = datasets.torch.TorchDataset(train_ex, use_pseudo=use_pseudo, rgb=rgb, **transform_kws)
    test_set   = datasets.torch.TorchDataset(test_ex, use_pseudo=False, rgb=rgb, **transform_kws)
    val_set    = datasets.torch.TorchDataset(val_ex, use_pseudo=False, rgb=rgb, **transform_kws)
    collate_fn = datasets.torch.collate_fn

    loader_kws = config.get('loader', {})
    train_loader = torch.utils.data.DataLoader(train_set, collate_fn=collate_fn, **loader_kws)

    if len(test_set) > 0:
        test_loader = torch.utils.data.DataLoader(test_set, collate_fn=collate_fn, **loader_kws)
    else:
        test_loader = None

    if len(val_set) > 0:
        val_loader = torch.utils.data.DataLoader(val_set, collate_fn=collate_fn, **loader_kws)
    else:
        val_loader = None

    task_kws = config.pop('task', {})
    task = tasks.TaskSpec(rgb=rgb, **task_kws)

    # instantiate model architecture
    model_kws = config.get('model', {})
    model = models.build_model(task, model_kws)
    n_params = models.count_params(model)
    utils.log(n_params)

    optimizer_kws = config.get('optimizer', {}).copy()
    optimizer_cls = getattr(torch.optim, optimizer_kws.pop('_class'))
    optimizer = optimizer_cls(model.parameters(), **optimizer_kws)

    phys_adapter = physics.api.get_adapter(config)

    evaluator_kws = config.get('evaluator', {})
    callbacks = [
        callbacks_module.LoggerCallback(keys=task.metric_keys),
        evaluation.PlotterCallback(
            keys=task.metric_keys,
            output_dir=outputs.base_dir / 'plotter'
        ),
        evaluation.ViewerCallback(
            keys=task.viewer_keys,
            output_dir=outputs.base_dir / 'viewer'
        ),
        evaluation.EvaluatorCallback(
            output_dir=outputs.base_dir / 'metrics',
            **evaluator_kws
        ),
        callbacks_module.TimerCallback()
    ]

    trainer_kws = config.get('trainer', {}).copy()
    trainer = trainer_module.Trainer(
        task=task,
        model=model,
        optimizer=optimizer,
        phys_adapter=phys_adapter,
        train_loader=train_loader,
        test_loader=test_loader,
        val_loader=val_loader,
        callbacks=callbacks,
        output_dir=outputs.base_dir / 'checkpoints'
    )
    try:
        trainer.load_state(by_mtime=True)
    except FileNotFoundError:
        utils.warn('Checkpoint not found.')

    trainer.train(**trainer_kws)


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

