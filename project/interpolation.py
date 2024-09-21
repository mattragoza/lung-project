import numpy as np
import torch
import torch.nn.functional as F
import fenics as fe
import torch_fenics


def interpolate_image(image, points, resolution=1.0, radius=1, sigma=None):
    '''
    Image interpolation method that takes a weighted sum
    of the image values in the neighborhood of each point.

    Args:
        image: (C, X, Y, Z) image tensor
        points: (N, 3) sampling points
        radius: neighborhood size (int)
        resolution: image resolution (float)
        sigma: Gaussian parameter (float)
    '''
    C, X, Y, Z = image.shape
    N, D = points.shape
    assert D == 3, 'points must be 3D'
    
    if isinstance(radius, int):
        radius = [radius] * D
    assert len(radius) == D

    if sigma is None:
        sigma = [r/2 for r in radius]
    
    zeros = torch.zeros(D, device=image.device)
    shape = torch.as_tensor([X, Y, Z], device=image.device)
    resolution = torch.as_tensor(resolution, device=image.device) 
    
    x_offsets = torch.arange(-radius[0], radius[0], device=image.device) + 1
    y_offsets = torch.arange(-radius[1], radius[1], device=image.device) + 1
    z_offsets = torch.arange(-radius[2], radius[2], device=image.device) + 1
    
    offsets = torch.meshgrid(x_offsets, y_offsets, z_offsets)
    offsets = torch.stack(offsets, dim=-1).reshape(-1, D)
    
    interpolated_values = torch.zeros(N, C, device=image.device)

    for i, point in enumerate(points):
        nearest_voxel = (point / resolution).floor().long()
        
        neighbor_voxels = nearest_voxel.unsqueeze(0) + offsets
        neighbor_voxels = neighbor_voxels.clamp(min=zeros, max=shape-1).long()
        
        neighbor_values = image[
            :,
            neighbor_voxels[:,0],
            neighbor_voxels[:,1],
            neighbor_voxels[:,2],
        ].T
        
        neighbor_points = neighbor_voxels * resolution.unsqueeze(0)
        distance = torch.norm(neighbor_points - point.unsqueeze(0), dim=1)
        
        #weights = torch.exp(-(distance**2) / (2*sigma**2))
        weights = torch.clamp(1 - torch.abs(distance / (2*sigma)), 0)
        weighted_sum = (weights.unsqueeze(1) * neighbor_values).sum(dim=0) 

        interpolated_values[i] = weighted_sum / weights.sum()
    
    return interpolated_values


def image_to_dofs(image, resolution, V, radius, sigma):
    '''
    Args:
        image: (C, X, Y, Z) torch.Tensor
        V: fenics.FunctionSpace
            with dofs on (N, 3) coordinates
    Returns:
        dofs: (N, C) torch.Tensor
    '''
    if V.num_sub_spaces() == 0 and image.ndim == 3:
        image = image.unsqueeze(0)

    C, X, Y, Z = image.shape
    
    dof_coords = V.tabulate_dof_coordinates()
    if V.num_sub_spaces() > 0:
        dof_coords = dof_coords[::V.num_sub_spaces(),:]
    
    N, D = dof_coords.shape

    dof_coords = torch.as_tensor(dof_coords, dtype=image.dtype, device=image.device)

    dofs = interpolate_image(
        image=image,
        points=dof_coords,
        resolution=resolution,
        radius=radius,
        sigma=sigma,
    ).double()

    if V.num_sub_spaces() > 0:
        return dofs.view(N, C)
    else:
        return dofs.view(N)


def dofs_to_image(dofs, V, image_shape, resolution):
    '''
    Args:
        dofs: (N, C) torch.Tensor
        V: fenics.FunctionSpace
            defined on (N, 3) coordinates
        image_shape: (X, Y, Z)
    Returns:
        image: (C, X, Y, Z) torch.Tensor
    '''
    if V.num_sub_spaces() > 0:
        N, C = dofs.shape
    else:
        N, = dofs.shape
        C = 1

    X, Y, Z = image_shape

    x = np.arange(X) * resolution[0]
    y = np.arange(Y) * resolution[1]
    z = np.arange(Z) * resolution[2]

    grid = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1)

    func = torch_fenics.numpy_fenics.numpy_to_fenics(
        dofs.detach().cpu().numpy(), fe.Function(V)
    )
    func.set_allow_extrapolation(True)

    image = np.zeros((X, Y, Z, C))

    for i in range(X):
        for j in range(Y):
            for k in range(Z):
                func.eval(image[i,j,k], grid[i,j,k])

    return torch.as_tensor(image).permute(3,0,1,2)
