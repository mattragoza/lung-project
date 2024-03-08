import numpy as np
import torch

from . import utils


class PDEDataset(torch.utils.data.Dataset):

    @classmethod
    def generate(
        cls,
        n_samples,
        n_freqs,
        image_size,
        pde_solver,
        r=0.7,
        mu_lim=(1, 10),
        device='cuda'
    ):
        # define spatial and frequency domain
        x = np.linspace(0, 1, image_size)
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
        coefs *= (r**f[None,:])
        coefs /= (r / (1 - r))
        
        # compute a features by weighting basis functions
        a = np.einsum('if,ifx->ifx', coefs, basis)
        assert a.shape == (n_samples, n_freqs, image_size)

        # compute mu as nonlinear combination of basis functions
        mu = (mu_lim[1] - mu_lim[0]) * a.sum(axis=1)**2 + mu_lim[0]
        assert mu.shape == (n_samples, image_size)
        
        # convert a, mu, and u boundary condition to tensors
        a  = torch.tensor(a, dtype=torch.float64)
        mu = torch.tensor(mu, dtype=torch.float64)
        ub = torch.zeros(n_samples, 1, dtype=torch.float64)
        
        # solve forward PDE and convert result to image
        mu_dofs = utils.image_to_dofs(mu, pde_solver.V)
        u_dofs = pde_solver.forward(mu_dofs, ub)

        return cls(a, mu, u_dofs, ub, device)

    def __init__(self, a, mu, u, ub, device='cuda'):
        super().__init__()
        self.a  = a
        self.mu = mu
        self.u  = u
        self.ub = ub
        self.device = device

    def __len__(self):
        return len(self.a)

    def __getitem__(self, idx):
        return (
            torch.as_tensor(self.a[idx]).to(self.device,  dtype=torch.float64),
            torch.as_tensor(self.mu[idx]).to(self.device, dtype=torch.float64),
            torch.as_tensor(self.u[idx]).to(self.device,  dtype=torch.float64),
            torch.as_tensor(self.ub[idx]).to(self.device, dtype=torch.float64)
        )
    
    def select(self, inds):
        a  = [self.a[i]  for i in inds]
        mu = [self.mu[i] for i in inds]
        u  = [self.u[i]  for i in inds]
        ub = [self.ub[i] for i in inds]
        return PDEDataset(a, mu, u, ub, self.device)
    
    def sample(self, n, seed=None):
        n = as_index(n, len(self))
        np.random.seed(seed)
        shuffled_inds = np.random.permutation(len(self))
        sampled_inds  = shuffled_inds[:n]
        return self.select(sampled_inds)
    
    def split(self, n, seed=None):
        n = as_index(n, len(self))
        np.random.seed(seed)
        shuffled_inds = np.random.permutation(len(self))
        train_inds, test_inds = np.split(shuffled_inds, [n])
        train_data = self.select(train_inds)
        test_data  = self.select(test_inds)
        return train_data, test_data


def as_index(n, length):
    return int(n * length) if isinstance(n, float) else n


def wavelet(x, f, width, phase):
    '''
    Args:
        x, f, width, phase
    Returns:
        e^[-4 (x / width)^2] sin(2 pi (f x + phase))
    '''
    gaussian = np.exp(-4 * (x / width)**2)
    sine = np.sin(2 * np.pi * (f * x + phase))
    return gaussian * sine

