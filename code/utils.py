# functions for converting between image-like arrays 
#   and vectors of dofs for a finite element basis

def image_to_dofs(image, V):
    '''
    Args:
        image: (batch_size, image_size) torch.Tensor
        V: (mesh_size, 1) fenics.FunctionSpace
    Returns:
        dofs: (batch_size, mesh_size) torch.Tensor
    '''
    batch_size, image_size = image.shape

    coords = V.tabulate_dof_coordinates()
    coords = np.c_[coords, coords*0]
    coords = torch.as_tensor(coords, dtype=image.dtype, device=image.device)
    
    mesh_size, n_dims = coords.shape
    coords = coords[None,None,:,:]

    return F.grid_sample(
        input=image.view(batch_size,1,1,image_size),
        grid=coords.expand(batch_size,1,mesh_size,n_dims) * 2 - 1,
        align_corners=True
    ).view(batch_size, mesh_size)


def dofs_to_image(dofs, V, image_size):
    '''
    Args:
        dofs: (batch_size, mesh_size) torch.Tensor
        V: (mesh_size, 1) fenics.FunctionSpace
        image_size: int
    Returns:
        image: (batch_size, image_size) torch.Tensor
    '''
    batch_size, mesh_size = dofs.shape
    
    coords = np.linspace(0, 1, image_size)

    image = np.zeros((batch_size, image_size))
    for i in range(batch_size):
        func = torch_fenics.numpy_fenics.numpy_to_fenics(
            dofs[i].detach().cpu().numpy(), fe.Function(V)
        )
        for j in range(image_size):
            func.eval(image[i,j,None], x[j,None])

    return torch.as_tensor(image, device=dofs.device)
