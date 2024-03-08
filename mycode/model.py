import numpy as np
import torch

from . import utils


class PDENet(torch.nn.Module):

    def __init__(self, n_inputs, n_filters, kernel_size, activ_fn, pde_solver):
        super().__init__()
        self.conv1 = conv1d(n_inputs,  n_filters, kernel_size)
        self.conv2 = conv1d(n_filters, n_filters, kernel_size)
        self.conv3 = conv1d(n_filters, 1, kernel_size)

        self.activ_fn = get_activ_fn(activ_fn)
        self.pde_solver = pde_solver

    def forward(self, a, ub):

        # apply convolution layers
        z1 = self.activ_fn(self.conv1(a))
        z2 = self.activ_fn(self.conv2(z1))
        mu = self.conv3(z2)[:,0,:]
        
        # convert image to FEM coefficients
        mu_dofs = utils.image_to_dofs(mu, self.pde_solver.V)

        # solve PDE forward problem
        u_dofs = self.pde_solver.forward(
            mu_dofs.to('cpu', dtype=torch.float64),
            ub.to('cpu', dtype=torch.float64)
        ).to('cuda', dtype=torch.float64)

        return mu, u_dofs


def conv1d(n_inputs, n_outputs, kernel_size):
    k = kernel_size
    return torch.nn.Conv1d(
        n_inputs, n_outputs, k, padding=k//2, padding_mode='reflect'
    )


def get_activ_fn(name):
    try:
        return getattr(torch.nn.functional, name)
    except AttributeError:
        return getattr(torch, name)
