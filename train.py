import project

@project.utils.main
def train(
    data_name='emory',
    data_root='data/Emory-4DCT',
    mask_roi='lung_regions',
    mesh_version=10,
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
    load_epoch=0
):
    if data_name == 'emory':
        train_image_set = project.imaging.Emory4DCT(data_root, phases=range(10,100,10))
        test_image_set = project.imaging.Emory4DCT(data_root, phases=[0])

        train_examples = train_image_set.get_examples(mask_roi, mesh_version)
        test_examples = test_image_set.get_examples(mask_roi, mesh_version)

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
