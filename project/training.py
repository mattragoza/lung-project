import sys, os, time
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

from . import pde, interpolation, evaluation, visual, utils


class Trainer(object):

    def __init__(
        self,
        model,
        train_data,
        test_data,
        batch_size,
        learning_rate,
        save_every,
        save_prefix,
        interp_radius=None,
        interp_sigma=None,
        sync_cuda=False
    ):
        self.model = model

        self.train_loader = torch.utils.data.DataLoader(
            train_data, batch_size, shuffle=True, collate_fn=collate_fn
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=1, shuffle=True, collate_fn=collate_fn
        )
        self.train_iterator = iter(enumerate(self.train_loader))
        self.test_iterator = iter(enumerate(self.test_loader))

        self.pde_class = pde.LinearElasticPDE
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.epoch = 0

        self.evaluator = evaluation.Evaluator(
            index_cols=['epoch', 'batch', 'example', 'phase', 'rep']
        )
        self.timer = evaluation.Timer(
            index_cols=['epoch', 'batch', 'example', 'phase', 'event'],
            sync_cuda=sync_cuda
        )
        self.array_viewers = {}
        self.metric_viewer = None

        self.interp_radius = interp_radius
        self.interp_sigma = interp_sigma

        self.save_every = save_every
        self.save_prefix = save_prefix

        if save_prefix:
            os.makedirs(self.viewer_dir, exist_ok=True)
            os.makedirs(self.state_dir, exist_ok=True)

    @property
    def save_dir(self):
        return os.path.split(self.save_prefix)[0]

    @property
    def save_name(self):
        return os.path.split(self.save_prefix)[1]

    @property
    def viewer_dir(self):
        return os.path.join(self.save_dir, 'viewers')

    @property
    def state_dir(self):
        return os.path.join(self.save_dir, 'state')

    def __repr__(self):
        return f'{type(self).__name__}(epoch={self.epoch})'

    def get_data_loader(self, phase):
        if phase == 'train':
            return self.train_loader
        elif phase == 'test':
            return self.test_loader

    def get_data_iterator(self, phase):
        if phase == 'train':
            return self.train_iterator
        elif phase == 'test':
            return self.test_iterator

    def reset_data_iterator(self, phase):
        if phase == 'train':
            self.train_iterator = iter(enumerate(self.train_loader))
        elif phase == 'test':
            self.test_iterator = iter(enumerate(self.test_loader))

    def get_next_batch(self, phase):
        try:
            return next(self.get_data_iterator(phase))
        except StopIteration:
            self.reset_data_iterator(phase)
            return next(self.get_data_iterator(phase))

    def train(self, num_epochs):
        print('Training...')
        start_epoch = self.epoch
        stop_epoch = self.epoch + num_epochs
        self.timer.start()
        for i in range(start_epoch, stop_epoch):
            print(f'Epoch {i+1}/{stop_epoch}')
            self.run_epoch(phase='train', epoch=i+1)
            self.run_next_batch(phase='test', epoch=i+1)
            self.epoch += 1
            if (self.epoch % self.save_every) == 0:
                self.save_metrics()
                self.save_viewers()
                self.save_state()
            self.timer.tick((i+1, -1, -1, 'test', 'save_state'))

    def run_epoch(self, phase, epoch):
        print(f'Running {phase} phase')
        for j, batch in self.get_data_iterator(phase):
            batch_num = j + 1
            self.timer.tick((epoch, batch_num, -1, phase, 'get_next_batch'))
            self.run_batch(batch, phase, epoch, batch_num)
        self.reset_data_iterator(phase)

    def run_next_batch(self, phase, epoch):
        j, batch = self.get_next_batch(phase)
        batch_num = j + 1
        self.timer.tick((epoch, batch_num, -1, phase, 'get_next_batch'))
        self.run_batch(batch, phase, epoch, batch_num)
    
    def run_batch(self, batch, phase, epoch, batch_num):
        anat_image, mu_image, u_image, mask, resolution, mesh, radius, example = batch
        print(f'{example}', end='', flush=True)

        # move tensors to GPU
        anat_image = anat_image.to('cuda')
        mu_image = mu_image.to('cuda')
        u_image = u_image.to('cuda')
        mask = mask.to('cuda')

        # predict elasticity from anatomical image
        mu_pred_image = self.model.forward(anat_image) * 1000
        self.timer.tick((epoch, batch_num, -1, phase, 'model_forward'))

        # physical FEM simulation
        total_loss = 0
        batch_size = len(example)
        for k in range(batch_size):
            print('.', end='', flush=True)
            pde = self.pde_class(mesh[k])
            exam_num = k + 1

            # convert tensors to FEM basis coefficients
            anat_dofs = interpolation.image_to_dofs(
                anat_image[k], resolution[k], pde.S,
                radius=self.interp_radius or radius[k],
                sigma=self.interp_sigma or radius[k]/2
            ).cpu()
            rho_dofs = (1 + anat_dofs/1000) * 1000
            mu_pred_dofs = interpolation.image_to_dofs(
                mu_pred_image[k], resolution[k], pde.S,
                radius=self.interp_radius or radius[k],
                sigma=self.interp_sigma or radius[k]/2
            ).cpu()
            mu_true_dofs = interpolation.image_to_dofs(
                mu_image[k], resolution[k], pde.S,
                radius=self.interp_radius or radius[k],
                sigma=self.interp_sigma or radius[k]/2
            ).cpu()
            u_true_dofs = interpolation.image_to_dofs(
                u_image[k], resolution[k], pde.V,
                radius=self.interp_radius or radius[k],
                sigma=self.interp_sigma or radius[k]/2
            ).cpu()
            self.timer.tick((epoch, batch_num, exam_num, phase, 'image_to_dofs'))

            # solve FEM for simulated displacement coefficients
            u_pred_dofs = pde.forward(
                u_true_dofs.unsqueeze(0),
                mu_pred_dofs.unsqueeze(0),
                rho_dofs.unsqueeze(0),
            )[0]
            self.timer.tick((epoch, batch_num, exam_num, phase, 'pde_forward'))

            # compute loss and evaluation metrics
            loss = self.evaluator.evaluate(
                anat_dofs.unsqueeze(1),
                mu_pred_dofs.unsqueeze(1),
                mu_true_ofs.unsqueeze(1),
                u_pred_dofs,
                u_true_dofs,
                mask=torch.ones_like(anat_dofs, dtype=int),
                index=(epoch, batch_num, example[k], phase, 'dofs')
            )
            total_loss += loss
            self.timer.tick((epoch, batch_num, exam_num, phase, 'dof_metrics'))

            if phase == 'test': # evaluate in image domain     
                u_pred_image = interpolation.dofs_to_image(
                    u_pred_dofs, pde.V, u_true_image[k].shape[-3:], resolution[k]
                ).to('cuda')
                u_pred_image = torch.as_tensor(u_pred_image)
                self.timer.tick((epoch, batch_num, exam_num, phase, 'dofs_to_image'))

                self.evaluator.evaluate(
                    anat_image[k].permute(1,2,3,0),
                    mu_pred_image[k].permute(1,2,3,0),
                    mu_true_image[k].permute(1,2,3,0),
                    u_pred_image.permute(1,2,3,0),
                    u_true_image[k].permute(1,2,3,0),
                    (mask[k,0] > 0).to(dtype=int),
                    index=(epoch, batch_num, example[k], phase, 'image')
                )
                self.timer.tick((epoch, batch_num, exam_num, phase, 'image_metrics'))

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
                self.timer.tick((epoch, batch_num, exam_num, phase, 'update_viewers'))

        loss = total_loss / batch_size
        print(f'{loss:.4f}', flush=True)

        if phase == 'train': # update parameters
            loss.backward()
            self.timer.tick((epoch, batch_num, -1, phase, 'loss_backward'))

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.timer.tick((epoch, batch_num, -1, phase, 'optimizer_step'))

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
                z_mid = array.z.median().values.astype(int)
                self.array_viewers[key] = visual.XArrayViewer(
                    array, x='x', y='y', col='c', label_cols=False
                )
                self.array_viewers[key].update_index(z=z_mid)
            else:
                self.array_viewers[key].update_array(array)

    def save_metrics(self):
        csv_path = f'{self.save_prefix}_metrics.csv'
        png_path = f'{self.save_prefix}_metrics.png'
        self.evaluator.save_metrics(csv_path)
        self.metric_viewer.fig.savefig(png_path)

    def save_viewers(self):
        for key, array_viewer in self.array_viewers.items():
            viewer_path = os.path.join(self.viewer_dir, f'{key}_{self.epoch}.png')
            array_viewer.fig.savefig(viewer_path)

    def save_state(self):
        model_path = os.path.join(self.state_dir, f'model_{self.epoch}.pt')
        optim_path = os.path.join(self.state_dir, f'optim_{self.epoch}.pt')
        model_state = self.model.state_dict()
        optim_state = self.optimizer.state_dict()
        torch.save(model_state, model_path)
        torch.save(optim_state, optim_path)

    def load_state(self, epoch):
        model_path = os.path.join(self.state_dir, f'model_{epoch}.pt')
        optim_path = os.path.join(self.state_dir, f'optim_{epoch}.pt')
        model_state = torch.load(model_path)
        optim_state = torch.load(optim_path)
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optim_state)
        self.epoch = epoch


def collate_fn(batch):
    # we need a custom collate_fn b/c mesh is not a tensor
    anat = torch.stack([ex[0] for ex in batch])
    elast = torch.stack([ex[1] for ex in batch])
    disp = torch.stack([ex[2] for ex in batch])
    mask = torch.stack([ex[3] for ex in batch])
    resolution = [ex[4] for ex in batch]
    mesh = [ex[5] for ex in batch]
    radius = [ex[6] for ex in batch]
    example = [ex[7] for ex in batch]
    return anat, elast, disp, mask, resolution, mesh, radius, example
