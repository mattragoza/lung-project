import project

@project.utils.main
def train(
    data_name='emory',
    data_root='data/Emory-4DCT',
    mask_roi='lung_regions',
    mesh_version=10,
    test_case=None,
    test_phase=-1,
    num_levels=3,
    num_conv_layers=2,
    conv_channels=8,
    conv_kernel_size=3,
    output_func='relu',
    rho_value='anat',
    interp_size=7,
    interp_type='tent',
    batch_size=1,
    learning_rate=1e-5,
    num_epochs=200,
    save_every=10,
    save_prefix='ASDF',
    load_epoch=0,
    random_seed=None,
):
    if isinstance(random_seed, str):
        random_seed = int(random_seed)
    project.utils.set_random_seed(random_seed)

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

        train_examples = all_examples[10:]
        test_examples = all_examples[:10]

    train_data = project.data.Dataset(train_examples)
    test_data = project.data.Dataset(test_examples)

    model = project.model.UNet3D(
        in_channels=1,
        out_channels=1,
        num_levels=num_levels,
        num_conv_layers=num_conv_layers,
        conv_channels=conv_channels,
        conv_kernel_size=conv_kernel_size,
        output_func=output_func,
    ).cuda()

    trainer = project.training.Trainer(
        model=model,
        train_data=train_data,
        test_data=test_data,
        batch_size=batch_size,
        learning_rate=learning_rate,
        rho_value=rho_value,
        interp_size=interp_size,
        interp_type=interp_type,
        save_every=save_every,
        save_prefix=save_prefix
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
        trainer.train(num_epochs)
    except:
        trainer.save_metrics()
        trainer.save_viewers()
        trainer.save_state()
        raise
