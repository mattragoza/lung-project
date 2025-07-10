import project

@project.utils.main
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
