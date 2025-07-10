import sys, os, time
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

from . import interpolation, evaluation, visual, utils


class Trainer(object):

    def __init__(
        self,
        model,
        train_data,
        test_data,
        batch_size,
        learning_rate,
        interp_size,
        interp_type,
        rho_value,
        test_every,
        save_every,
        save_prefix,
        sync_cuda=False,
        input_anat=True,
        input_coords=False
    ):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data

        self.train_loader = torch.utils.data.DataLoader(
            train_data, batch_size, shuffle=True, collate_fn=collate_fn
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=1, shuffle=True, collate_fn=collate_fn
        )
        self.train_iterator = iter(enumerate(self.train_loader))
        self.test_iterator = iter(enumerate(self.test_loader))

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

        self.test_every = test_every
        self.save_every = save_every
        self.save_prefix = save_prefix

        if save_prefix:
            os.makedirs(self.viewer_dir, exist_ok=True)
            os.makedirs(self.state_dir, exist_ok=True)

        self.interp_size = interp_size
        self.interp_type = interp_type
        self.rho_value = rho_value

        self.input_anat = input_anat
        self.input_coords = input_coords

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

    def get_learning_rate(self):
        return self.optimizer.param_groups[0]['lr']

    def set_learning_rate(self, lr):
        self.optimizer.param_groups[0]['lr'] = lr

    def is_test_epoch(self):
        return self.test_every > 0 and (self.epoch % self.test_every) == 0

    def is_save_epoch(self):
        return self.save_every > 0 and (self.epoch % self.save_every) == 0

    def train(self, num_epochs):
        print('Training...')
        start_epoch = self.epoch
        stop_epoch = self.epoch + num_epochs
        self.timer.start()
        for i in range(start_epoch, stop_epoch):
            print(f'Epoch {i+1}/{stop_epoch}')
            self.run_epoch(phase='train', epoch=i+1)
            if self.is_test_epoch():
                self.run_next_batch(phase='test', epoch=i+1)
            self.epoch += 1
            if self.is_save_epoch():
                self.save_all()
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
        self.timer.tick((epoch, j+1, -1, phase, 'get_next_batch'))
        self.run_batch(batch, phase, epoch, j+1)

    def run_batch(self, batch, phase, epoch, batch_num):
        a_image, e_image, u_image, mask, disease_mask, resolution, pde, name = batch
        print(f'{name}', end='', flush=True)

        # move CPU tensors to GPU
        a_image = a_image.to('cuda')
        e_image = e_image.to('cuda')
        u_image = u_image.to('cuda')
        region_mask = mask.to('cuda')
        binary_mask = (region_mask > 0)
        disease_mask = disease_mask.to('cuda')
        resolution = resolution.to('cuda')

        if self.input_coords:
            coords = get_input_coords(binary_mask, resolution)
            if self.input_anat:
                model_input = torch.cat([a_image, coords], dim=1)
            else:
                model_input = coords
        elif self.input_anat:
            model_input = a_image

        # predict elasticity from anatomical image
        e_pred_image = self.model.forward(model_input) * 1000 # kPa -> Pa
        e_pred_image = e_pred_image.clamp(min=1, max=1e12)

        self.timer.tick((epoch, batch_num, -1, phase, 'model_forward'))

        # physical FEM simulation
        total_loss = 0
        batch_size = len(batch[0])
        for k in range(batch_size):
            print('.', end='', flush=True)
            points_k = pde[k].points.to('cuda')
            radius_k = pde[k].radius.to('cuda')

            assert not torch.isnan(a_image[k]).any(), \
                f'a_image contains nan ({name[k]})'
            assert not torch.isnan(e_image[k]).any(), \
                f'e_image contains nan ({name[k]})'
            assert not torch.isnan(u_image[k]).any(), \
                f'u_image contains nan ({name[k]})'
            assert not torch.isnan(e_pred_image[k]).any(), \
                f'e_pred_image contains nan ({name[k]})'

            # convert tensors to FEM coefficients
            kernel_size = self.interp_size
            a_dofs = interpolation.interpolate_image(
                a_image[k], binary_mask[k], resolution[k], points_k, radius_k,
                kernel_size=self.interp_size,
                kernel_type=self.interp_type,
            ).to(dtype=torch.float64, device='cpu')

            if self.rho_value == 'anat':
                rho_dofs = (a_dofs + 1000)
            else:
                rho_dofs = torch.full_like(a_dofs, float(self.rho_value))

            e_true_dofs = interpolation.interpolate_image(
                e_image[k], binary_mask[k], resolution[k], points_k, radius_k,
                kernel_size=self.interp_size,
                kernel_type=self.interp_type,
            ).to(dtype=torch.float64, device='cpu')

            e_pred_dofs = interpolation.interpolate_image(
                e_pred_image[k], binary_mask[k], resolution[k], points_k, radius_k,
                kernel_size=self.interp_size,
                kernel_type=self.interp_type,
            ).to(dtype=torch.float64, device='cpu')

            u_true_dofs = interpolation.interpolate_image(
                u_image[k], binary_mask[k], resolution[k], points_k, radius_k,
                kernel_size=self.interp_size,
                kernel_type=self.interp_type,
            ).to(dtype=torch.float64, device='cpu')

            dof_region_mask = interpolation.interpolate_image(
                region_mask[k], binary_mask[k], resolution[k], points_k, radius_k,
                kernel_size=1,
                kernel_type='nearest',
            ).to(dtype=torch.int32, device='cpu')
            self.timer.tick((epoch, batch_num, k+1, phase, 'image_to_dofs'))

            dof_disease_mask = interpolation.interpolate_image(
                disease_mask[k], binary_mask[k], resolution[k], points_k, radius_k,
                kernel_size=1,
                kernel_type='nearest',
            ).to(dtype=torch.int32, device='cpu')
            self.timer.tick((epoch, batch_num, k+1, phase, 'image_to_dofs'))

            # solve pde for simulated displacement field
            u_pred_dofs = pde[k].forward(
                u_true_dofs[None,:,:],
                e_pred_dofs[None,:,0],
                rho_dofs[None,:,0],
            )[0]
            assert not torch.isnan(u_pred_dofs).any(), \
                f'u_pred_dofs contains nan ({name[k]})'
    
            self.timer.tick((epoch, batch_num, k+1, phase, 'pde_forward'))

            # compute loss and evaluation metrics
            loss = self.evaluator.evaluate(
                anat=a_dofs,
                e_pred=e_pred_dofs,
                e_true=e_true_dofs,
                u_pred=u_pred_dofs,
                u_true=u_true_dofs,
                mask=dof_region_mask[:,0],
                disease_mask=dof_disease_mask,
                index=(epoch, batch_num, name[k], phase, 'dofs')
            )
            self.update_metric_viewer()

            assert not torch.isnan(loss).any(), f'loss is nan ({name[k]})'

            total_loss += loss
            self.timer.tick((epoch, batch_num, k+1, phase, 'dof_metrics'))

            if phase == 'test': # evaluate in image domain
                binary_mask_k = binary_mask[k,0].to(dtype=torch.int32)
                region_mask_k = region_mask[k,0].to(dtype=torch.int32)
                disease_mask_k = disease_mask[k].to(dtype=torch.int32)
                resolution_k = resolution[k].detach().cpu().numpy()

                u_pred_image = interpolation.dofs_to_image(
                    u_pred_dofs, pde[k].V, u_image[k].shape[-3:], resolution_k
                ).to('cuda')
                self.timer.tick((epoch, batch_num, k+1, phase, 'dofs_to_image'))

                self.evaluator.evaluate(
                    anat=a_image[k].permute(1,2,3,0),
                    e_pred=e_pred_image[k].permute(1,2,3,0),
                    e_true=e_image[k].permute(1,2,3,0),
                    u_pred=u_pred_image.permute(1,2,3,0),
                    u_true=u_image[k].permute(1,2,3,0),
                    mask=region_mask_k,
                    disease_mask=disease_mask_k.permute(1,2,3,0),
                    index=(epoch, batch_num, name[k], phase, 'image')
                )
                self.timer.tick((epoch, batch_num, k+1, phase, 'image_metrics'))
                self.update_metric_viewer()

                alpha = 1.0
                alpha_mask = (1 - alpha * (1 - binary_mask_k))
                emph_mask = (
                    binary_mask_k +
                    (a_image[k] < -850) +
                    (a_image[k] < -900) +
                    (a_image[k] < -950)
                )
                self.update_xarray_viewers(
                    resolution_k,
                    anat=a_image[k] * alpha_mask,
                    emph=emph_mask * binary_mask_k - 1,
                    e_pred=e_pred_image[k] * alpha_mask,
                    e_true=e_image[k] * alpha_mask,
                    u_pred=u_pred_image * alpha_mask,
                    u_true=u_image[k] * alpha_mask
                )
                self.timer.tick((epoch, batch_num, k+1, phase, 'update_viewers'))

        loss = total_loss / batch_size
        print(f'{loss:.8f}', flush=True)

        if phase == 'train': # update parameters
            loss.backward()
            self.timer.tick((epoch, batch_num, -1, phase, 'loss_backward'))

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.timer.tick((epoch, batch_num, -1, phase, 'optimizer_step'))

    def fit(self, num_epochs):
        example = self.test_data[0] # fit single example
        a_image, e_image, u_image, mask, disease_mask, resolution, pde, name = example
        print(f'Fitting {name}...')

        # move CPU tensors to GPU
        a_image = a_image.to('cuda')
        e_image = e_image.to('cuda')
        u_image = u_image.to('cuda')
        region_mask = mask.to('cuda')
        binary_mask = (region_mask > 0)
        disease_mask = disease_mask.to('cuda')

        coords = get_input_coords(binary_mask, resolution)

        pde.points = pde.points.to('cuda')
        pde.radius = pde.radius.to('cuda')

        # convert tensors to FEM coefficients
        a_dofs = interpolation.interpolate_image(
            a_image, binary_mask, resolution, pde.points, pde.radius,
            kernel_size=self.interp_size,
            kernel_type=self.interp_type,
        ).to(dtype=torch.float64, device='cpu')

        e_true_dofs = interpolation.interpolate_image(
            e_image, binary_mask, resolution, pde.points, pde.radius,
            kernel_size=self.interp_size,
            kernel_type=self.interp_type,
            ).to(dtype=torch.float64, device='cpu')

        u_true_dofs = interpolation.interpolate_image(
            u_image, binary_mask, resolution, pde.points, pde.radius,
            kernel_size=self.interp_size,
            kernel_type=self.interp_type,
        ).to(dtype=torch.float64, device='cpu')

        dof_region_mask = interpolation.interpolate_image(
            region_mask, binary_mask, resolution, pde.points, pde.radius,
            kernel_size=1,
            kernel_type='nearest',
        ).to(dtype=torch.int32, device='cpu')

        dof_disease_mask = interpolation.interpolate_image(
            disease_mask, binary_mask, resolution, pde.points, pde.radius,
            kernel_size=1,
            kernel_type='nearest',
        ).to(dtype=torch.int32, device='cpu')

        images = (a_image, e_image, u_image, region_mask, binary_mask, disease_mask, coords)
        dofs = (a_dofs, e_true_dofs, u_true_dofs, dof_region_mask, dof_disease_mask)

        start_epoch = self.epoch
        stop_epoch = self.epoch + num_epochs
        for i in range(start_epoch, stop_epoch):
            print(f'[{i+1}/{stop_epoch}] train', end=' ')
            self.fit_epoch(images, resolution, pde, name, dofs, 'train', epoch=i+1)
            if self.is_test_epoch():
                print(f'[{i+1}/{stop_epoch}] test', end=' ')
                self.fit_epoch(
                    images, resolution, pde, name, dofs, 'test', epoch=i+1
                )
            self.epoch += 1
            if self.is_save_epoch():
                self.save_all()

    def fit_epoch(self, images, resolution, pde, name, dofs, phase, epoch):
        a_image, e_image, u_image, region_mask, binary_mask, disease_mask, coords = images
        a_dofs, e_true_dofs, u_true_dofs, dof_region_mask, dof_disease_mask = dofs
        batch_num = 0

        if self.input_coords and self.input_anat:
            model_input = torch.cat([a_image, coords], dim=0)
        elif self.input_coords:
            model_input = coords
        elif self.input_anat:
            model_input = a_image

        # get elasticity parameter map
        e_pred_image = self.model.forward(model_input.unsqueeze(0))[0] * 1000
        e_pred_image = e_pred_image.clamp(min=1, max=1e12)

        if self.rho_value == 'anat':
            rho_dofs = (a_dofs + 1000)
        else:
            rho_dofs = torch.full_like(a_dofs, float(self.rho_value))

        e_pred_dofs = interpolation.interpolate_image(
            e_pred_image, binary_mask, resolution, pde.points, pde.radius,
            kernel_size=self.interp_size,
            kernel_type=self.interp_type,
        ).to(dtype=torch.float64, device='cpu')

        # solve pde for simulated displacement field
        u_pred_dofs = pde.forward(
            u_true_dofs[None,:,:],
            e_pred_dofs[None,:,0],
            rho_dofs[None,:,0],
        )[0]
        assert not torch.isnan(u_pred_dofs).any(), \
            f'u_pred_dofs contains nan ({name})'
        
        # compute loss and evaluation metrics
        loss = self.evaluator.evaluate(
            anat=a_dofs,
            e_pred=e_pred_dofs,
            e_true=e_true_dofs,
            u_pred=u_pred_dofs,
            u_true=u_true_dofs,
            mask=dof_region_mask[:,0],
            disease_mask=dof_disease_mask,
            index=(epoch, 0, name, 'train', 'dofs')
        )
        self.update_metric_viewer()

        assert not torch.isnan(loss).any(), f'loss is nan ({name})'

        if phase == 'test': # evaluate in image domain
            binary_mask = binary_mask[0].to(dtype=torch.int32)
            region_mask = region_mask[0].to(dtype=torch.int32)
            disease_mask = disease_mask.to(dtype=torch.int32)

            u_pred_image = interpolation.dofs_to_image(
                u_pred_dofs, pde.V, u_image.shape[-3:], resolution
            ).to('cuda')

            self.evaluator.evaluate(
                anat=a_image.permute(1,2,3,0),
                e_pred=e_pred_image.permute(1,2,3,0),
                e_true=e_image.permute(1,2,3,0),
                u_pred=u_pred_image.permute(1,2,3,0),
                u_true=u_image.permute(1,2,3,0),
                mask=region_mask,
                disease_mask=disease_mask.permute(1,2,3,0),
                index=(epoch, 0, name, phase, 'image')
            )
            self.update_metric_viewer()

            alpha = 1.0
            alpha_mask = (1 - alpha * (1 - binary_mask))
            emph_mask = (
                binary_mask +
                (a_image < -850) +
                (a_image < -900) +
                (a_image < -950)
            )
            self.update_xarray_viewers(
                resolution,
                anat=a_image * alpha_mask,
                emph=emph_mask * binary_mask - 1,
                e_pred=e_pred_image * alpha_mask,
                e_true=e_image * alpha_mask,
                u_pred=u_pred_image * alpha_mask,
                u_true=u_image * alpha_mask
            )

        print(f'{loss:.8f}', flush=True)

        if phase == 'train': # update parameters
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def update_metric_viewer(self):
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

    def update_xarray_viewers(self, resolution, **kwargs):
        # update xarray viewers
        for key, value in kwargs.items():
            array = utils.as_xarray(
                value, dims=['c', 'x', 'y', 'z'], name=key, resolution=resolution
            )
            if key not in self.array_viewers:
                slice_dim = 'y'
                slice_mid = len(array[slice_dim]) // 2
                self.array_viewers[key] = visual.XArrayViewer(
                    array, x='x', y='z', col='c', label_cols=False
                )
                self.array_viewers[key].update_index(**{slice_dim: slice_mid})
            else:
                self.array_viewers[key].update_array(array)

    def save_all(self):
        self.save_metrics()
        self.save_viewers()
        self.save_state()

    def save_metrics(self):
        self.evaluator.save_metrics(f'{self.save_prefix}_metrics.csv')
        self.metric_viewer.fig.savefig(f'{self.save_prefix}_metrics.png')
        self.timer.save_benchmarks(f'{self.save_prefix}_benchmarks.csv')

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
    a_image = torch.stack([ex[0] for ex in batch])
    e_image = torch.stack([ex[1] for ex in batch])
    u_image = torch.stack([ex[2] for ex in batch])
    mask = torch.stack([ex[3] for ex in batch])
    disease_mask = torch.stack([ex[4] for ex in batch])
    res  = torch.as_tensor([ex[5] for ex in batch])
    pde  = [ex[6] for ex in batch]
    name = [ex[7] for ex in batch]
    return a_image, e_image, u_image, mask, disease_mask, res, pde, name


def get_input_coords(binary_mask, resolution):
    shape = binary_mask.shape[-3:]
    device = binary_mask.device
    resolution = torch.as_tensor(resolution, device=device)
    x = torch.arange(shape[0], device=device)
    y = torch.arange(shape[1], device=device)
    z = torch.arange(shape[2], device=device)
    coords = torch.stack(torch.meshgrid(x, y, z), dim=0) # ndim == 4
    if binary_mask.ndim == 5: # add batch dim to coords
        coords = coords.unsqueeze(0)
    elif binary_mask.ndim == 3: # add channel dim to mask
        binary_mask = binary_mask.unsqueeze(0)
    center = evaluation.weighted_mean(coords, binary_mask, dim=(-3,-2,-1))
    coords = coords - center[...,None,None,None]
    coords = coords * resolution[...,None,None,None]
    return coords
