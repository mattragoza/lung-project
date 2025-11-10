import numpy as np
import torch

from .core import utils, transforms, interpolation
from . import solvers


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        train_loader,
        test_loader=None,
        val_loader=None,
        evaluator=None,
        supervised=False,
        solver_kws=None,
        device='cuda'
    ):
        self.model = model.to(device)
        self.optimizer = optimizer

        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.evaluator = evaluator

        self.supervised = supervised
        self.solver_kws = solver_kws or {}
        self.device = device

        # physics settings
        self.rho_known = False
        self.rho_bias = 1000.
        self.nu_value = 0.4
        self.scalar_degree = 1
        self.vector_degree = 1

        self.epoch = 0
        self.cache = {}

    def train(self, num_epochs, val_every=1, save_every=10, save_path=None):
        utils.log('Start training loop')

        for _ in range(num_epochs):
            utils.log(f'Epoch {self.epoch}/{self.epoch + num_epochs}')
            if val_every and self.epoch % val_every == 0:
                self.run_val_phase()
            if save_every and self.epoch % save_every == 0:
                self.save_state(save_path)
            self.run_train_phase()
            self.epoch += 1

        if self.val_loader:
            self.run_val_phase()
        if self.test_loader:
            self.run_test_phase()
        if save_path:
            self.save_state(save_path)

    def run_train_phase(self):
        utils.log('Start train phase')
        self.model.train()
        for i, batch in enumerate(self.train_loader):
            utils.log(f'Train batch {i+1}/{len(self.train_loader)}')
            self.optimizer.zero_grad()
            loss = self.forward(batch, phase='train')
            loss.backward()
            self.optimizer.step()

    @torch.no_grad()
    def run_test_phase(self):
        utils.log('Start test phase')
        self.model.eval()
        for i, batch in enumerate(self.test_loader):
            utils.log(f'Test batch {i+1}/{len(self.test_loader)}')
            self.forward(batch, phase='test')

    @torch.no_grad()
    def run_val_phase(self):
        utils.log('Start val phase')
        self.model.eval()
        for i, batch in enumerate(self.val_loader):
            utils.log(f'Val batch {i+1}/{len(self.val_loader)}')
            self.forward(batch, phase='val')

    def forward(self, batch, phase):
        image = batch['image'].to(self.device)
        batch_size = image.shape[0]

        # predict elastic modulus from image
        E_pred = self.model.forward(image)

        if self.supervised: # train using E error
            E_true = batch['elast'].to(self.device)
            mask = batch['mask'].to(self.device) > 0
            return nrmse_loss(E_pred, E_true, mask)

        # train using simulated displacement error via PDE solver
        loss = torch.zeros(batch_size, device=self.device, dtype=image.dtype)

        for k in range(batch_size):
            pde_module, points = self.get_pde_module(batch['example'][k], batch['mesh'][k])
            E_nodes = interpolation.interpolate_image(E_pred[k], points)
            mu_nodes, lam_nodes = transforms.compute_lame_parameters(E_nodes, self.nu_value) 
            loss[k] = pde_module.forward(mu_nodes, lam_nodes)

        return loss.mean()

    def get_pde_module(self, ex, mesh, device='cpu'):
        subj = ex.subject
        if subj in self.cache:
            return self.cache[subj]

        unit = ex.metadata['unit'] # meters per world unit
        verts = mesh.points # world coords
        cells = mesh.cells_dict['tetra']

        if self.rho_known:
            if self.scalar_degree == 0:
                rho_values = mesh.cell_data_dict['rho']['tetra']
            elif self.scalar_degree == 1:
                rho_values = mesh.point_data['rho']
        else:
            img_nodes = mesh.point_data['image']
            img_cells = mesh.cell_data_dict['image']['tetra']
            if self.scalar_degree == 0:
                img_values = (img_cells + img_nodes[cells].mean(axis=1)) / 2
            elif self.scalar_degree == 1:
                img_values = (img_nodes + transforms.cell_to_node_values(verts, cells, img_cells)) / 2
            rho_values = self.rho_bias * img_values

        if self.vector_degree == 0:
            u_obs_values = mesh.cell_data_dict['u'] # world units
        elif self.vector_degree == 1:
            u_obs_values = mesh.point_data['u']

        # convert to tensors on CPU (for caching)
        verts = torch.as_tensor(verts * unit_m, dtype=dtype, device=device) # to meters
        cells = torch.as_tensor(cells, dtype=torch.int, device=device)

        rho_values = torch.as_tensor(rho_values, dtype=dtype, device=device)
        u_obs_nodes = torch.as_tensor(u_obs_nodes, dtype=dtype, device=device) * unit_m # to metes

        solver = solvers.warp.WarpFEMSolver(
            unit_m=unit_m,
            scalar_degree=self.scalar_degree,
            vector_degree=self.vector_degree,
            **(self.solver_kws or {})
        )
        module = solvers.base.PDESolverModule(
            solver, verts, cells, rho_values, u_obs_values
        )

        if self.scalar_degree == 0:
            points = verts
        elif self.scalar_degree == 1:
            points = verts[cells].mean(axis=1)

        self.cache[key] = (module, points)
        return module, points

    def save_state(self, path):
        torch.save({
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'optim': self.optimizer.state_dict()
        }, path)

    def load_state(self, path):
        state = torch.load(path)
        self.epoch = state['epoch']
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optim'])


def rmse_loss(pred, target, weights):
    err = torch.norm((pred - target) * weights, axis=1)
    return torch.sqrt(torch.mean(err**2))


def nrmse_loss(pred, target, weights, eps=1e-12):
    err = torch.norm((pred - target) * weights, axis=1)
    mag = torch.norm(target * weights, axis=1)
    num = torch.sqrt(torch.mean(err**2))
    den = torch.sqrt(torch.mean(mag**2)) + eps
    return num / den

