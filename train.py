import project

@project.utils.main
def train(
    data_root='data/Emory-4DCT',
    mask_roi='lung_combined_mask',
    mesh_radius=20,
    interp_radius=20,
    interp_sigma=10,
    num_levels=3,
    num_conv_layers=2,
    conv_channels=4,
    conv_kernel_size=3,
    output_func='exp',
    batch_size=4,
    learning_rate=1e-5,
    num_epochs=100,
    save_every=10,
    save_prefix='ASDF'
):
    train_images = project.imaging.Emory4DCT(data_root, phases=range(10,100,10))
    test_images = project.imaging.Emory4DCT(data_root, phases=[0])

    train_examples = train_images.get_examples(mask_roi, mesh_radius)
    test_examples = test_images.get_examples(mask_roi, mesh_radius)

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
        interp_radius=interp_radius,
        interp_sigma=interp_sigma,
        save_every=save_every,
        save_prefix=save_prefix,
    )
    trainer.train(num_epochs)
