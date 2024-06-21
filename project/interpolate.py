import numpy as np
import fenics as fe
import torch
import torch_fenics


def image_to_dofs(image, resolution, V):
    '''
    Args:
        image: (n_x, n_y, n_z, n_c) torch.Tensor
        V: fenics.FunctionSpace
            defined on (mesh_size, 3) coordinates
    Returns:
        dofs: (batch_size, mesh_size, n_channels) torch.Tensor
    '''    
    if V.num_sub_spaces() == 0:
        image = image.unsqueeze(-1)

    n_x, n_y, n_z, n_c = image.shape
    
    coords = V.tabulate_dof_coordinates()
    if V.num_sub_spaces() > 0:
        coords = coords[::V.num_sub_spaces()]
    
    mesh_size, n_dims = coords.shape

    coords = torch.as_tensor(coords, dtype=image.dtype, device=image.device)

    shape = torch.as_tensor([n_x, n_y, n_z], dtype=image.dtype, device=image.device)
    resolution = torch.as_tensor(resolution, dtype=image.dtype, device=image.device)
    extent = (shape - 1) * resolution

    dofs = F.grid_sample(
        input=image[None,...].permute(0,4,3,2,1), # xyzc -> bczyx
        grid=(coords[None,None,None,...] / extent) * 2 - 1,
        align_corners=True
    )
    if V.num_sub_spaces() > 0:
        return dofs.view(n_c, mesh_size).permute(1,0)
    else:
        return dofs.view(mesh_size)


def dofs_to_image(dofs, V, image_shape, resolution):
    '''
    Args:
        dofs: (mesh_size, n_c) torch.Tensor
        V: fenics.FunctionSpace
            defined on (mesh_size, 3) coordinates
        image_shape: (n_x, n_y, n_z)
    Returns:
        image: (n_x, n_y, n_z, n_c) torch.Tensor
    '''
    if V.num_sub_spaces() > 0:
        mesh_size, n_c = dofs.shape
    else:
        mesh_size, = dofs.shape
        n_c = 1

    n_x, n_y, n_z = image_shape

    x = np.arange(n_x) * resolution[0]
    y = np.arange(n_y) * resolution[1]
    z = np.arange(n_z) * resolution[2]

    grid = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1)
    print(image_shape, grid.shape)

    func = torch_fenics.numpy_fenics.numpy_to_fenics(
        dofs.detach().cpu().numpy(), fe.Function(V)
    )
    func.set_allow_extrapolation(True)

    image = np.zeros((n_x, n_y, n_z, n_c))

    for i in range(n_x):
        for j in range(n_y):
            for k in range(n_z):
                func.eval(image[i,j,k], grid[i,j,k])

    if V.num_sub_spaces() == 0:
        return image.squeeze(-1)
    else:
        return image
