import sys, os
import project


def parse_args(argv):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--subject', default=None, help='Subject IDs (comma-separated)')
    p.add_argument('--data_root', default=None, help='Dataset root directory')
    p.add_argument('--variant', default='TEST', help='Preprocessed data variant')
    p.add_argument('--config', default=None, help='Path to JSON configuration file')
    p.add_argument('--output', default=None, help='Output csv path')
    return p.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    print(vars(args))

    dataset_cls = CONFIG[args.dataset]['dataset_cls']
    data_root = args.data_root or CONFIG[args.dataset]['default_root']

    if not os.path.isdir(data_root):
        raise RuntimeError(f'{data_root} is not a valid directory')

    ds = dataset_cls(data_root)

    config = project.core.fileio.load_json(args.config) if args.config else {}

    examples_cfg = config.get('examples')
    examples = list(ds.examples(subjects, args.variant, **examples_cfg))

    split_kws = training_cfg.get('cross_val')
    train_ex, test_ex, val_ex = split_by_category(examples, **split_kws)

    loader_kws = dict(batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    loader_kws['collate_fn'] = project.datasets.torch.collate_fn

    train_set = project.datasets.torch.TorchDataset(train_ex)
    train_loader = torch.utils.data.DataLoader(train_set, **loader_kws)

    if test_ex:
        test_set = project.datasets.torch.TorchDataset(test_ex)
        test_loader = torch.utils.data.DataLoader(test_set, **loader_kws)
    else:
        test_set = test_loader = None

    if val_ex:
        val_set = project.datasets.torch.TorchDataset(val_ex)
        val_loader = torch.utils.data.DataLoader(val_set, **loader_kws)
    else:
        val_set = val_loader = None

    model_kws = training_cfg.get('model_arch')
    model = project.model.UNet3D(in_channels=1, out_channels=1, **model_kws)
    print(project.model.count_params(model))

    test_input = torch.zeros((1, 1, 256, 256, 256), dtype=torch.float32, device='cuda')
    print(project.model.count_activs(model, test_input))

    optimizer_kws = training_cfg.get('optimizer')
    optimizer = torch.optim.AdamW(model.parameters(), **optim_kws)

    solver_kws = training_cfg.get('solver')
    evaluator = project.evaluation.Evaluator()
    trainer = project.training.Trainer(
        model, optimizer, train_loader, test_loader, val_loader, evaluator, solver_kws
    )
    trainer.train(**train_kws)


if __name__ == '__main__':
    main(sys.argv[1:])


## DEPRECATED

#@project.utils.main
def train(
    random_seed=None,

    data_name='emory',
    data_root='data/Emory-4DCT',
    mask_roi='lung_regions2',
    mesh_version=11,
    test_case=None,
    test_phase=-1,
    test_pid=None,

    input_anat=True,
    input_coords=False,

    model_arch='unet3d',
    num_levels=3,
    num_conv_layers=2,
    conv_channels=32,
    conv_kernel_size=3,
    output_func='relu',

    trainer_task='train',
    rho_value='anat',
    interp_size=5,
    interp_type='tent',
    batch_size=1,
    learning_rate=1e-5,
    num_epochs=200,
    test_every=10,
    save_every=10,
    save_prefix='ASDF',
    load_epoch=0
):
    print(data_name, input_anat, input_coords, model_arch, trainer_task)

    if isinstance(random_seed, str):
        random_seed = int(random_seed)

    project.utils.set_random_seed(random_seed)

    assert data_name in {'emory', 'phantom'}

    if data_name == 'emory':

        if not test_case or test_case == 'None':
            train_cases = test_cases = project.imaging.ALL_CASES
        else:
            assert test_case in project.imaging.ALL_CASES
            train_cases = [c for c in project.imaging.ALL_CASES if c != test_case]
            test_cases = [test_case]

        if test_phase == -1:
            train_phases = test_phases = project.imaging.ALL_PHASES
        else:
            assert test_phase in project.imaging.ALL_PHASES
            train_phases = [p for p in project.imaging.ALL_PHASES if p != test_phase]
            test_phases = [test_phase]

        print(test_cases)
        print(test_phases)

        train_images = project.imaging.Emory4DCT(data_root, train_cases, train_phases)
        test_images = project.imaging.Emory4DCT(data_root, test_cases, test_phases)

        train_examples = train_images.get_examples(mask_roi, mesh_version)
        test_examples = test_images.get_examples(mask_roi, mesh_version)

    elif data_name == 'phantom':
        phantom_set = project.phantom.PhantomSet(data_root, phantom_ids=range(100))
        all_examples = phantom_set.get_examples(mesh_version)

        if test_pid in {None, 'None', -1}:
            train_examples = all_examples[10:]
            test_examples = all_examples[:10]
        else:
            train_examples = all_examples
            test_examples = [train_examples.pop(int(test_pid))]

    train_data = project.data.Dataset(train_examples)
    test_data = project.data.Dataset(test_examples)

    assert model_arch in {'unet3d', 'param_map'}

    if model_arch == 'unet3d':
        model = project.model.UNet3D(
            in_channels=1*input_anat + 3*input_coords,
            out_channels=1,
            num_levels=num_levels,
            num_conv_layers=num_conv_layers,
            conv_channels=conv_channels,
            conv_kernel_size=conv_kernel_size,
            output_func=output_func
        ).cuda()

    elif model_arch == 'param_map':
        shape = test_data[0][1].shape
        model = project.model.ParameterMap(
            shape=(1, shape[1]//2, shape[2]//2, shape[3]//2),
            upsample_mode='nearest',
            conv_kernel_size=conv_kernel_size,
            output_func=output_func
        ).cuda()

    assert trainer_task in {'train', 'fit'}

    trainer = project.training.Trainer(
        model=model,
        train_data=train_data,
        test_data=test_data,
        batch_size=batch_size,
        learning_rate=learning_rate,
        rho_value=rho_value,
        interp_size=interp_size,
        interp_type=interp_type,
        test_every=test_every,
        save_every=save_every,
        save_prefix=save_prefix,
        input_anat=input_anat,
        input_coords=input_coords
    )

    if load_epoch > 0:
        trainer.load_epoch(epoch=load_epoch)

    # test forward pass and initialize viewers
    trainer.timer.start()
    trainer.run_next_batch(phase='test', epoch=trainer.epoch)

    trainer.save_metrics()
    trainer.save_viewers()
    trainer.save_state()

    trainer.timer.start()
    try:
        if trainer_task == 'train':
            trainer.train(num_epochs)

        elif trainer_task == 'fit':
            assert len(test_data) == 1
            trainer.fit(num_epochs)
    except:
        trainer.save_metrics()
        trainer.save_viewers()
        trainer.save_state()
        raise
