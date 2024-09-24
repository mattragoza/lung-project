import time
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

from . import pde, interpolation, evaluation, visual, utils


class Trainer(object):

    def __init__(
        self, model, train_data, test_data, batch_size, learning_rate
    ):
        self._model = model
        self.train_loader = torch.utils.data.DataLoader(
            train_data, batch_size, shuffle=True, collate_fn=collate_fn
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_data, batch_size, shuffle=True, collate_fn=collate_fn
        )
        self.pde_class = pde.LinearElasticPDE
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.epoch = 0

        self.evaluator = evaluation.Evaluator(
            index_cols=['epoch', 'batch', 'example', 'phase', 'rep']
        )
        self.array_viewers = {}
        self.metric_viewer = None

    @property
    def model(self):
        return self._model
    
    @property
    def train_dataset(self):
        return self.train_loader.dataset

    @property
    def test_dataset(self):
        return self.test_loader.dataset
        
    @property
    def batch_size(self):
        return self.train_loader.batch_sampler.batch_size
    
    @property
    def learning_rate(self):
        return self.optimizer.param_groups[0]['lr']

    def __repr__(self):
        return f'{type(self).__name__}(epoch={self.epoch})'

    def get_data_loader(self, phase):
        if phase == 'train':
            return self.train_loader
        elif phase == 'test':
            return self.test_loader

    def train(self, num_epochs):
        print('Training...')
        start_epoch = self.epoch
        stop_epoch = self.epoch + num_epochs
        for i in range(start_epoch, stop_epoch):
            print(f'Epoch {i+1}/{stop_epoch}')
            self.run_epoch(phase='train', epoch=i+1)
            self.run_epoch(phase='test', epoch=i+1)
            self.epoch += 1

    def run_epoch(self, phase, epoch):
        print(f'Running {phase} phase')
        for j, batch in enumerate(self.get_data_loader(phase)):
            self.run_batch(batch, phase, epoch, batch_num=j+1)
    
    def run_batch(self, batch, phase, epoch, batch_num):
        anat_image, u_true_image, mask, resolution, mesh, radius, example = batch
        print(f'{example}', end='', flush=True)

        # predict elasticity from anatomical image
        mu_pred_image = self.model.forward(anat_image)
        mu_pred_image = torch.exp(mu_pred_image) * 1000

        # physical FEM simulation
        total_loss = 0
        batch_size = len(example)
        for k in range(batch_size):
            print('.', end='', flush=True)
            pde = self.pde_class(mesh[k])

            # convert tensors to FEM basis coefficients
            u_true_dofs = interpolation.image_to_dofs(
                u_true_image[k], resolution[k], pde.V,
                radius=radius[k],
                sigma=radius[k]/2
            ).cpu()
            mu_pred_dofs = interpolation.image_to_dofs(
                mu_pred_image[k], resolution[k], pde.S,
                radius=radius[k],
                sigma=radius[k]/2
            ).cpu()
            anat_dofs = interpolation.image_to_dofs(
                anat_image[k], resolution[k], pde.S,
                radius=radius[k], 
                sigma=radius[k]/2
            ).cpu()
            rho_dofs = (1 + anat_dofs/1000) * 1000

            # solve FEM for simulated displacement coefficients
            u_pred_dofs = pde.forward(
                u_true_dofs.unsqueeze(0),
                mu_pred_dofs.unsqueeze(0),
                rho_dofs.unsqueeze(0),
            )[0]

            # compute loss and evaluation metrics
            loss = self.evaluator.evaluate(
                anat_dofs.unsqueeze(1),
                mu_pred_dofs.unsqueeze(1),
                u_pred_dofs,
                u_true_dofs,
                mask=torch.ones_like(anat_dofs, dtype=int),
                index=(epoch, batch_num, example[k], phase, 'dofs')
            )
            total_loss += loss

            if phase == 'test': # evaluate in image domain     
                u_pred_image = interpolation.dofs_to_image(
                    u_pred_dofs, pde.V, u_true_image[k].shape[-3:], resolution[k]
                )
                u_pred_image = torch.as_tensor(u_pred_image).cuda()

                self.evaluator.evaluate(
                    anat_image[k].permute(1,2,3,0),
                    mu_pred_image[k].permute(1,2,3,0),
                    u_pred_image.permute(1,2,3,0),
                    u_true_image[k].permute(1,2,3,0),
                    mask[k,0].to(dtype=int),
                    index=(epoch, batch_num, example[k], phase, 'image')
                )
                alpha = 0.5
                alpha_mask = (1 - alpha * (1 - mask[k]))
                emph = (
                    mask[k] +
                    (anat_image[k] < -850) +
                    (anat_image[k] < -900) +
                    (anat_image[k] < -950)
                )
                self.update_viewers(
                    anat=anat_image[k] * alpha_mask,
                    emph=emph * mask[k] - 1,
                    mu_pred=mu_pred_image[k] * alpha_mask,
                    u_pred=u_pred_image * alpha_mask,
                    u_true=u_true_image[k] * alpha_mask
                )

        loss = total_loss / batch_size
        print(f'{loss:.4f}', flush=True)

        if phase == 'train': # update parameters
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def update_viewers(self, **kwargs):

        if self.metric_viewer is None:
            self.metric_viewer = visual.DataViewer(
                self.evaluator.long_format_metrics,
                x='epoch', y='value', hue='phase', row='rep', col='metric',
                levels={
                    'phase': ['train', 'test'],
                    'rep': ['dofs', 'image'],
                    'metric': self.evaluator.metric_cols,
                },
                ylabel_var='row'
            )
        else:
            self.metric_viewer.update_data(self.evaluator.long_format_metrics)

        # update xarray viewers
        for key, value in kwargs.items():
            array = utils.as_xarray(value, dims=['c', 'x', 'y', 'z'], name=key)
            if key not in self.array_viewers:
                self.array_viewers[key] = visual.XArrayViewer(array)
            else:
                self.array_viewers[key].update_array(array)

    def save_state(self, prefix):
        pass

    def load_state(self, prefix):
        pass


def collate_fn(batch):
    # we need a custom collate_fn bc mesh is not a tensor
    anat = torch.stack([ex[0] for ex in batch])
    mask = torch.stack([ex[1] for ex in batch])
    disp = torch.stack([ex[2] for ex in batch])
    resolution = [ex[3] for ex in batch]
    mesh = [ex[4] for ex in batch]
    radius = [ex[5] for ex in batch]
    example = [ex[6] for ex in batch]
    return anat, mask, disp, resolution, mesh, radius, example
