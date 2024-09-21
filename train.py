import project

@project.utils.main
def train(
    data_root='data/Emory-4DCT',
    test_index=0,
    mask_roi='lung_combined_mask',
    mesh_radius=20,
    batch_size=4,
    learning_rate=1e-5,
    num_epochs=1
):
    emory4dct = project.imaging.Emory4DCT(data_root)

    train_examples = emory4dct.get_examples(mask_roi, mesh_radius)
    test_example = train_examples.pop(test_index)

    train_data = project.data.Dataset(train_examples)
    test_data = project.data.Dataset([test_example])

    model = project.model.UNet3D(
        in_channels=1,
        out_channels=1,
        num_levels=3,
        num_conv_layers=2,
        conv_channels=4,
        conv_kernel_size=3
    ).cuda()

    trainer = project.training.Trainer(
        model, train_data, test_data, mesh_radius, batch_size, learning_rate
    )
    trainer.train(num_epochs)
