import numpy as np
import torch


def interpolate_image(
    image: torch.Tensor,
    points: torch.Tensor,
    mode: str='bilinear',
    padding: str='border',
    align_corners: bool=True,
    reshape: bool=True
):
    '''
    Args:
        image:  (C, I, J, K) input image tensor
        points: (N, 3) tensor of voxel coordinates,
            where component order == image dim order
    Returns:
        (N, C) tensor of interpolated image values
    '''
    from . import transforms
    import torch.nn.functional as F

    points = transforms.normalize_voxel_coords(
        points,
        image.shape[1:],
        align_corners=align_corners,
        flip_order=True
    )
    output = F.grid_sample(
        input=image[None,:,:,:,:],       # (B, C, I, J, K)
        grid=points[None,None,None,:,:], # (B, L, M, N, 3)
        mode=mode,
        padding_mode=padding,
        align_corners=align_corners
    ) # (B, C, L, M, N)
    if reshape:
        return output[0,:,0,0,:].T # (N, C)
    else:
        return output[0] # (C, L, M, N)


def deform_image(
    image: torch.Tensor,
    disp: torch.Tensor,
    mode: str='bilinear',
    padding: str='border'
):
    '''
    Args:
        image: (C, I, J, K) input image tensor
        disp:  (L, M, N, 3) displacement tensor,
            in terms of voxel coordinates
    Returns:
        (C, L, M, N)
    '''
    import torch

    grid = torch.cartesian_prod(
        torch.arange(disp.shape[0]),
        torch.arange(disp.shape[1]),
        torch.arange(disp.shape[2]),
    )
    points = (grid + disp).view(-1, 3)

    return interpolate_image(
        image, points, mode, padding, align_corners=True, reshape=False
    )


## DEPRECATED


def my_interpolate_image(
    image, mask, resolution, points, kernel_radius,
    kernel_size=None,
    kernel_type='tent',
):
    '''
    Image interpolation method that takes a weighted sum
    of the image values in the neighborhood of each point.

    Args:
        image: (C, X, Y, Z) image tensor
        mask: (C, X, Y, Z) mask tensor
        resolution: image resolution (float)
        points: (N, 3) sampling points
        kernel_radius: kernel shape parameter (N,)
        kernel_size: kernel window half-size
        kernel_type: 'flat', 'tent', or 'bell'
    '''
    import torch

    C, X, Y, Z = image.shape
    N, D = points.shape
    assert D == 3, 'points must be 3D'
    
    zeros = torch.zeros(D, device=image.device)
    shape = torch.as_tensor([X, Y, Z], device=image.device)
    resolution = torch.as_tensor(resolution, device=image.device) 

    if kernel_size is None:
        kernel_size = (kernel_radius / resolution).floor().long().max()

    x_offsets = torch.arange(-kernel_size, kernel_size, device=image.device) + 1
    y_offsets = torch.arange(-kernel_size, kernel_size, device=image.device) + 1
    z_offsets = torch.arange(-kernel_size, kernel_size, device=image.device) + 1
    
    offsets = torch.meshgrid(x_offsets, y_offsets, z_offsets, indexing='ij')
    offsets = torch.stack(offsets, dim=-1).reshape(-1, D) # (K, D)

    nearest_voxel = (points / resolution.unsqueeze(0)).floor().long() # (N, D)
    
    neighbor_voxels = nearest_voxel.unsqueeze(1) + offsets.unsqueeze(0) # (N, K, D)
    neighbor_voxels = neighbor_voxels.clamp(min=zeros, max=(shape - 1)).long()
    
    neighbor_values = image[
        :,
        neighbor_voxels[:,:,0],
        neighbor_voxels[:,:,1],
        neighbor_voxels[:,:,2],
    ].permute(1,2,0) # (N, K, C)

    neighbor_mask = mask[
        0,
        neighbor_voxels[:,:,0],
        neighbor_voxels[:,:,1],
        neighbor_voxels[:,:,2],
    ] # (N, K)
  
    neighbor_points = neighbor_voxels * resolution.unsqueeze(0).unsqueeze(0) # (N, K, D)
    
    displacement = (neighbor_points - points.unsqueeze(1)) # (N, K, D)
    distance = torch.norm(displacement, dim=-1) # (N, K)
    
    if kernel_type == 'flat':
        weights = (distance <= kernel_radius).float()
    elif kernel_type == 'tent':
        weights = torch.clamp(1 - torch.abs(distance / kernel_radius), 0) # (N, K)
    elif kernel_type == 'bell':
        kernel_sigma = kernel_radius / 2
        weights = torch.exp(-(distance / kernel_sigma)**2) # (N, K)
    elif kernel_type == 'nearest':
        weights = (distance == distance.min(dim=-1, keepdims=True).values).float()
    else:
        raise ValueError(f"invalid kernel type '{kernel_type}\'")
    
    weights = weights * neighbor_mask
    
    weighted_sum = (weights.unsqueeze(-1) * neighbor_values).sum(dim=1) # (N, C)
    total_weight = weights.sum(dim=-1, keepdims=True) + 1e-8 # (N, 1)
    assert (total_weight > 0).all(), 'zero total weight'
    
    interpolated_values = weighted_sum / total_weight # (N, C)

    return interpolated_values


def image_to_dofs(image, resolution, V, kernel_radius):
    '''
    Args:
        image: (C, X, Y, Z) torch.Tensor
        V: fenics.FunctionSpace
            defined on (N, D) coordinates
    Returns:
        dofs: (N, C) torch.Tensor
    '''
    import torch
    C, X, Y, Z = image.shape
    
    points = V.tabulate_dof_coordinates()
    if V.num_sub_spaces() > 0:
        points = points[::V.num_sub_spaces(),:]
    
    N, D = points.shape

    points = torch.as_tensor(points, dtype=image.dtype, device=image.device)

    values = interpolate_image(
        image=image,
        resolution=resolution,
        points=points,
        kernel_radius=kernel_radius
    ).double()

    if V.num_sub_spaces() > 0:
        return values.view(N, C)
    else:
        return values.view(N)


def dofs_to_image(dofs, V, image_shape, resolution):
    '''
    Args:
        dofs: (N, C) torch.Tensor
        V: fenics.FunctionSpace
            defined on (mesh_size, 3) coordinates
        image_shape: (int, int, int) tuple
    Returns:
        image: (C, X, Y, Z) torch.Tensor
    '''
    import numpy as np
    import fenics as fe
    import torch_fenics

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

