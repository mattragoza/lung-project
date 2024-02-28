import sys, os
import numpy as np
import scipy as sp

import fenics as fe
import fenics_adjoint as fa

import torch
import torch.nn.functional as F
import torch_fenics

import pandas as pd
import matplotlib.pyplot as plt
import tqdm.notebook as tqdm
import xarray as xr
import hvplot.xarray


class PDESolver(torch_fenics.FEniCSModule):
    
    def __init__(self, n_nodes):
        from fenics import grad, dot, dx, ds
        super().__init__()
        
        # create function space
        n_elements = n_nodes - 1
        self.mesh = fe.UnitIntervalMesh(n_elements)
        self.V = fe.FunctionSpace(self.mesh, 'P', 1)
        
        # create trial and test functions
        self.u = fe.TrialFunction(self.V)
        self.v = fe.TestFunction(self.V)
        
        # construct bilinear form
        self.a = inner(grad(self.u), grad(self.v)) * dx
        
    def solve(self, f, ub):
        from fenics import grad, dot, dx, ds
        
        # construct linear form
        L = f * self.v * dx
        
        # construct boundary condition
        bc = fa.DirichletBC(self.V, ub, 'on_boundary')
        
        # solve the Poisson equation
        u = fa.Function(self.V)
        fa.solve(self.a == L, u, bc)
        
        return u
    
    def input_templates(self):
        return fa.Function(self.V), fa.Constant(0)


class PDEDataset(torch.utils.data.Dataset):

    def __init__(self, a, mu, u, ub, device):
        super().__init__()
        self.a = a
        self.mu = mu
        self.u = u
        self.ub = ub
        self.device = device
        
    @classmethod
    def generate(
        cls, n_samples, n_nodes, n_freqs, r=0.7, mu_range=(0, 1), device='cuda'
    ):
        # define spatial and frequency domain
        x = np.linspace(0, 1, n_nodes)
        f = np.arange(1, n_freqs + 1)
        
        # construct wavelet basis for each sample
        shift = np.random.uniform(0.2, 0.8, (n_samples, 1, 1))
        width = np.random.uniform(0.1, 0.6, (n_samples, 1, 1))
        phase = np.random.uniform(0, 1, (n_samples, n_freqs, 1))
        basis = wavelet(
            x[None,None,:] - shift, f[None,:,None], width, phase
        )
        
        # randomly sample basis coefficients and normalize
        coefs = np.random.normal(0, 1, (n_samples, n_freqs))
        coefs *= (r**f[None:])
        coefs /= (r / (1 - r))
        
        # compute a features by weighting basis functions
        a = np.einsum('if,ifx->ifx', coefs, basis)

        # compute mu as nonlinear combination of basis functions
        mu_min, mu_max = (1, 10)
        mu = (mu_max - mu_min) * a.sum(axis=1)**2 + mu_min
        
        # create boundary conditions and solve forward PDE       
        a  = torch.tensor(a, dtype=torch.float64)
        mu = torch.tensor(mu, dtype=torch.float64)
        ub = torch.zeros(n_samples, 1, dtype=torch.float64)
        u  = PDESolver(n_nodes).forward(mu, ub)
        
        return cls(a, mu, u, ub, device)

    def __len__(self):
        return len(self.a)

    def __getitem__(self, idx):
        return (
            torch.as_tensor(self.a[idx]).to(self.device, dtype=torch.float32),
            torch.as_tensor(self.mu[idx]).to(self.device, dtype=torch.float32),
            torch.as_tensor(self.u[idx]).to(self.device, dtype=torch.float32),
            torch.as_tensor(self.ub[idx]).to(self.device, dtype=torch.float32)
        )
    
    def select(self,inds):
        a  = [self.a[i]  for i in inds]
        mu = [self.mu[i] for i in inds]
        u  = [self.u[i]  for i in inds]
        ub = [self.ub[i] for i in inds]
        return PDEDataset(a, mu, u, ub, self.device)
    
    def sample(self, n, seed=None):
        np.random.seed(seed)
        shuffled_inds = np.random.permutation(len(self))
        sampled_inds = shuffled_inds[:n]
        return self.select(sampled_inds)
    
    def split(self, n, seed=None):
        np.random.seed(seed)
        shuffled_inds = np.random.permutation(len(self))
        train_inds, test_inds = np.split(shuffled_inds, [n])
        train_data = self.select(train_inds)
        test_data = self.select(test_inds)
        return train_data, test_data 


class PDENet(torch.nn.Module):

    def __init__(self, n_nodes, n_inputs, n_filters, kernel_size, activ_fn):
        super().__init__()
        k = kernel_size
        self.conv1 = torch.nn.Conv1d(
            n_inputs,  n_filters, k, padding=k//2, padding_mode='reflect'
        )
        self.conv2 = torch.nn.Conv1d(
            n_filters, n_filters, k, padding=kernel_size//2, padding_mode='reflect'
        )
        self.conv3 = torch.nn.Conv1d(
            n_filters, 1, k, padding=k//2, padding_mode='reflect'
        )
        try:
            self.activ_fn = getattr(torch.nn.functional, activ_fn)
        except AttributeError:
            self.activ_fn = getattr(torch, activ_fn)
        self.solve_pde = PDESolver(n_nodes)

    def forward(self, a, ub):
        z1 = self.activ_fn(self.conv1(a))
        z2 = self.activ_fn(self.conv2(z1))
        mu = self.conv3(z2)[:,0,:]
        u  = self.solve_pde(
            mu.to('cpu', dtype=torch.float64),
            ub.to('cpu', dtype=torch.float64)
        ).to('cuda', dtype=torch.float32)
        return mu, u


class TrainingPlot(object):
    '''
    Interactive training plot.
    '''
    def __init__(self):
        
        # create subplots for evaluation metrics
        self.fig, ax = plt.subplots(1, 2, figsize=(8,4))
        ax[0].set_ylabel('u_loss')
        ax[1].set_ylabel('mu_loss')
        for ax_ in ax:
            ax_.set_axisbelow(True)
            ax_.grid(linestyle=':')
            ax_.set_xlabel('iteration')
        self.fig.tight_layout()
        
        # store data and artists for interactive ploting
        self.data = pd.DataFrame(columns=['iter', 'phase', 'u_loss', 'mu_loss'])

        self.train_u_loss_line = ax[0].plot([], [], label='train')[0]
        self.test_u_loss_line  = ax[0].plot([], [], label='test')[0]
        self.train_mu_loss_line = ax[1].plot([], [], label='train')[0]
        self.test_mu_loss_line  = ax[1].plot([], [], label='test')[0]
        
    def draw(self, pad=1e-8):
        ax = self.fig.get_axes()
        ax[0].set_xlim(0, self.data.iter.max() * 1.1 + pad)
        ax[0].set_ylim(0, self.data.u_loss.max() * 1.1 + pad)
        ax[1].set_xlim(0, self.data.iter.max() * 1.1 + pad)
        ax[1].set_ylim(0, self.data.mu_loss.max() * 1.1 + pad)
        self.fig.canvas.draw()
        
    def update_train(self, iteration, u_loss, mu_loss):
        self.data.loc[len(self.data)] = [iteration, 'train', u_loss.item(), mu_loss.item()]
        
        data = self.data.groupby(['phase', 'iter']).mean()
        train = data.loc['train'].reset_index()
        if isinstance(train, pd.Series): # need > 1 rows
            return
        
        self.train_u_loss_line.set_xdata(train.iter)
        self.train_u_loss_line.set_ydata(train.u_loss)

        self.train_mu_loss_line.set_xdata(train.iter)
        self.train_mu_loss_line.set_ydata(train.mu_loss)

        self.draw()
        
    def update_test(self, iteration, u_loss, mu_loss):
        self.data.loc[len(self.data)] = [iteration, 'test', u_loss.item(), mu_loss.item()]
        
        data = self.data.groupby(['phase', 'iter']).mean()
        test = data.loc['test'].reset_index()
        if isinstance(test, pd.Series): # need > 1 rows
            return
        
        self.test_u_loss_line.set_xdata(test.iter)
        self.test_u_loss_line.set_ydata(test.u_loss) 
        
        self.test_mu_loss_line.set_xdata(test.iter)
        self.test_mu_loss_line.set_ydata(test.mu_loss)
        
        self.draw()


class Trainer(object):

    def __init__(self, train_set, val_set, model, optim, batch_size):
        self.train_loader = DataLoader(train_set, batch_size, shuffle=True)
        self.val_loader = DataLoader(val_set, batch_size, shuffle=True)
        self.model = model
        self.optim = optim
        self.plot = TrainingPlot()

    def train_one_epoch(self):
        self.model.train()
        for i, (a, mu, u, ub) in enumerate(self.train_loader):
            mu_hat, u_hat = self.model.forward(a, ub)
            mu_loss = self.loss_fn(mu_hat, mu)
            u_loss = self.loss_fn(u_hat, u)
            u_loss.backward()
            self.optim.step()

    def eval_one_epoch(self):
        self.model.eval()
        for i, (a, mu, u, ub) in enumerate(self.val_loader):
            with torch.no_grad():
                mu_hat, u_hat = self.model.forward(a, ub)
                mu_loss = self.loss_fn(mu_hat, mu)
                u_loss = self.loss_fn(u_hat, u)

    def train(self, n_epochs, verbose=False):
        for epoch in range(n_epochs):
            self.train_one_epoch()
            self.eval_one_epoch()


if __name__ == '__main__':

    # configuration
    n_samples = 10000
    n_freqs = 32
    n_nodes = 128
    n_filters = 16
    kernel_size = 5
    activ_fn = 'leaky_relu'

    dataset = PDEDataset.generate(n_samples, n_freqs, n_nodes)
    train_set, val_set = dataset.split(n=int(n_samples*0.9))
    model = PDENet(n_nodes, n_freqs, n_filters, kernel_size, activ_fn)

