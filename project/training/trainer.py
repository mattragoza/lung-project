from __future__ import annotations
from pathlib import Path
import numpy as np
import torch

from ..core import utils


class Trainer:

    def __init__(
        self,
        task: TaskSpec,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        phys_adapter: project.physics.PhysicsAdapter,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader = None,
        val_loader: torch.utils.data.DataLoader = None,
        callbacks: torch.utils.data.DataLoader = None,
        output_dir: str = 'checkpoints',
        device: str = 'cuda'
    ):
        self.task = task
        self.model = model.to(device)
        self.optimizer = optimizer
        self.phys_adapter = phys_adapter

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.callbacks = callbacks or []

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.device = device

        self.epoch = 0 # number of complete epochs
        self.step  = 0 # number of optimizer steps

        self.timer = utils.Timer()

    # ----- training loop / phases -----

    def check_epoch(self, interval: int):
        return interval and self.epoch % interval == 0

    def train(self, num_epochs, val_every=1, test_every=5, save_every=5):
        self.start_train()
        start_step = self.step

        while self.epoch < num_epochs:
            self.start_epoch()

            if self.output_dir and self.check_epoch(save_every):
                self.save_state()

            if self.val_loader and self.check_epoch(val_every):
                self.run_val_phase()

            if self.test_loader and self.check_epoch(test_every):
                self.run_test_phase()

            self.run_train_phase()
            self.end_epoch()

        if self.output_dir and self.step > start_step:
            self.save_state()

        if self.val_loader:
            self.run_val_phase()

        if self.test_loader:
            self.run_test_phase()

        self.end_train()

    def run_train_phase(self):
        self.start_phase(phase='train')

        for i, batch in enumerate(self.train_loader):
            self.start_batch(phase='train', batch=i)

            self.start_forward()
            outputs = self.forward(batch, eval_mode=False)
            self.end_forward()

            loss = outputs['loss']
            if not torch.isfinite(loss):
                raise RuntimeError(f'Invalid loss: {loss.item()}')

            self.start_backward()
            loss.backward()
            self.end_backward()

            grad_norm = param_grad_norm(self.model)
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f'Invalid grad_norm: {grad_norm.item()}')
            outputs['grad_norm'] = grad_norm

            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.step += 1

            self.end_batch(phase='train', batch=i, outputs=outputs)

        self.end_phase(phase='train')

    @torch.no_grad()
    def run_test_phase(self):
        self.start_phase(phase='test')

        for i, batch in enumerate(self.test_loader):
            self.start_batch(phase='test', batch=i)

            self.start_forward()
            outputs = self.forward(batch, eval_mode=True)
            self.end_forward()

            outputs['grad_norm'] = param_grad_norm(self.model)

            self.end_batch(phase='test', batch=i, outputs=outputs)

        self.end_phase(phase='test')

    @torch.no_grad()
    def run_val_phase(self):
        self.start_phase(phase='val')

        for i, batch in enumerate(self.val_loader):
            self.start_batch(phase='val', batch=i)

            self.start_forward()
            outputs = self.forward(batch, eval_mode=True)
            self.end_forward()

            outputs['grad_norm'] = param_grad_norm(self.model)

            self.end_batch(phase='val', batch=i, outputs=outputs)

        self.end_phase(phase='val')

    # ----- forward pass -----

    def forward(self, batch, eval_mode=False):
        batch_size = len(batch['example'])

        device = self.device
        mask = batch['mask'].to(device, dtype=torch.bool)

        input_keys = [self.task.input_key(inp) for inp in self.task.inputs]
        input_vals = [batch[k].to(device, dtype=torch.float) for k in input_keys]
        inputs_cat = torch.cat(input_vals, dim=1)

        preds = self.model.forward(inputs_cat)

        outputs = {
            'example':  batch['example'],
            'mask':     batch['mask'].cpu(),
            'mat_true': batch['mat_true'].cpu()
        }
        for k in input_keys:
            if k not in outputs:
                outputs[k] = batch[k].cpu()

        # decide whether to run physics simulation
        need_physics_loss = self.task.has_physics_loss
        need_physics_eval = self.task.has_physics_output and eval_mode
        run_physics = need_physics_loss or need_physics_eval

        sim_loss = None
        if run_physics: # compute displacement error via physics simulation

            sim_loss = torch.zeros(batch_size, device=self.device, dtype=torch.float)
            sim_outputs = [None] * batch_size

            for k in range(batch_size):
                sim_params = {
                    name: preds[self.task.output_key(name)][k]
                    for name in self.task.physics_outputs
                }
                sim_loss[k], sim_outputs[k] = self.phys_adapter.voxel_simulation_loss(
                    mesh=batch['mesh'][k],
                    unit_m=batch['example'][k].metadata['unit'],
                    affine=batch['affine'][k],
                    params=sim_params,
                    bc_spec=None,
                    ret_outputs=True
                )

            outputs['sim'] = sim_outputs

        # compute multi-task loss
        total_loss = torch.zeros((), device=device)
        total_base = torch.zeros((), device=device)

        for tgt in self.task.targets:
            output_key = self.task.output_key(tgt)
            target_key = self.task.target_key(tgt)

            y_pred = preds[output_key].to(device)

            if tgt in self.task.losses:
                y_true = batch[target_key].to(device)

                loss_name = self.task.losses[tgt].lower()
                loss_weight = self.task.weights.get(tgt, 1.0)

                if loss_name == 'ce':
                    y_base = torch.zeros_like(y_pred) # uniform logits
                    loss = masked_cross_entropy(y_pred, y_true, mask)
                    base = masked_cross_entropy(y_base, y_true, mask)

                elif loss_name == 'mse':
                    y_base = torch.full_like(y_pred, y_true.mean())
                    loss = mean_squared_error(y_pred, y_true, mask)
                    base = mean_squared_error(y_base, y_true, mask)

                elif loss_name == 'msre':
                    y_base = torch.full_like(y_pred, y_true.mean())
                    loss = mean_squared_relative_error(y_pred, y_true, mask)
                    base = mean_squared_relative_error(y_base, y_true, mask)

                elif loss_name == 'sim':
                    loss = sim_loss.mean()
                    base = 1.0 # TODO
                else:
                    raise ValueError(loss_name)

                total_loss = total_loss + loss_weight * loss
                total_base = total_base + loss_weight * base

            outputs[output_key] = preds[output_key].cpu()
            m = mask.expand(-1, y_pred.shape[1], -1, -1, -1)
            outputs[output_key + '.mean'] = torch.mean(y_pred[m].float()).detach().cpu()
            outputs[output_key + '.std']  = torch.std(y_pred[m].float()).detach().cpu()

            if target_key in batch:
                outputs[target_key] = batch[target_key].cpu()
                m = mask.expand(-1, y_true.shape[1], -1, -1, -1)
                outputs[target_key + '.mean'] = torch.mean(y_true[m].float()).detach().cpu()
                outputs[target_key + '.std']  = torch.std(y_true[m].float()).detach().cpu()

        outputs['loss'] = total_loss
        outputs['loss_base'] = total_base.detach().cpu()
        outputs['loss_ratio'] = (total_loss / total_base.clamp_min(1e-12)).detach().cpu()
        return outputs

    # ----- callback hooks -----

    def start_train(self):
        for cb in self.callbacks:
            cb.on_train_start()

    def end_train(self):
        for cb in self.callbacks:
            cb.on_train_end()

    def start_epoch(self):
        for cb in self.callbacks:
            cb.on_epoch_start(self.epoch)

    def end_epoch(self):
        for cb in self.callbacks:
            cb.on_epoch_end(self.epoch)
        self.epoch += 1

    def start_phase(self, phase: str):
        if phase.lower() == 'train':
            self.model.train()
        else:
            self.model.eval()
        for cb in self.callbacks:
            cb.on_phase_start(self.epoch, phase)

    def end_phase(self, phase: str):
        for cb in self.callbacks:
            cb.on_phase_end(self.epoch, phase)

    def start_batch(self, phase: str, batch: int):
        for cb in self.callbacks:
            cb.on_batch_start(self.epoch, phase, batch, self.step)

    def end_batch(self, phase: str, batch: int, outputs):
        for cb in self.callbacks:
            self.timer.tick(sync=False)
            cb.on_batch_end(self.epoch, phase, batch, self.step, outputs=outputs)
            stats = self.timer.tick(sync=False)
            utils.log(f'{cb.name}: {stats}')

    def start_forward(self):
        for cb in self.callbacks:
            cb.on_forward_start()

    def end_forward(self):
        for cb in self.callbacks:
            cb.on_forward_end()

    def start_backward(self):
        for cb in self.callbacks:
            cb.on_backward_start()

    def end_backward(self):
        for cb in self.callbacks:
            cb.on_backward_end()

    # ----- saving / loading state -----

    def save_state(self, path=None):
        if path is None:
            path = self._checkpoint_path(self.epoch)
        path = Path(path)
        utils.log(f'Saving {path}')
        torch.save({
            'epoch': self.epoch,
            'step':  self.step,
            'model': self.model.state_dict(),
            'optim': self.optimizer.state_dict()
        }, path)

    def load_state(self, path=None, epoch=None, by_mtime=False):
        if path is not None:
            path = Path(path)
        elif epoch is not None:
            path = self._checkpoint_path(epoch)
        else:
            path = self._latest_checkpoint(by_mtime)
        if path is None:
            raise FileNotFoundError(f'No checkpoints found in {self.output_dir}')

        utils.log(f'Loading {path}')
        state = torch.load(path, map_location=self.device)

        self.epoch = int(state.get('epoch', 0))
        self.step  = int(state.get('step', 0))

        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optim'])

    def _checkpoint_path(self, epoch: int):
        return self.output_dir / f'checkpoint{epoch:05d}.pt'

    def _latest_checkpoint(self, by_mtime: bool = False):
        ckpts = self._list_checkpoints(by_mtime)
        if not ckpts:
            return None
        return ckpts[-1]

    def _list_checkpoints(self, by_mtime: bool = False):
        if not self.output_dir.exists():
            return []

        import re
        pat = re.compile(r'^checkpoint(\d{5})\.pt$')

        def is_ckpt(p):
            return p.is_file() and pat.match(p.name)

        paths = [p for p in self.output_dir.iterdir() if is_ckpt(p)]

        if by_mtime:
            sort_key = lambda p: p.stat().st_mtime
        else:
            sort_key = lambda p: p.name

        return sorted(paths, key=sort_key)



@torch.no_grad()
def param_grad_norm(model):
    norm2 = torch.zeros((), device='cpu')
    for p in model.parameters():
        if p.grad is not None:
            norm2 += p.grad.pow(2).sum().cpu()
    return norm2.pow(0.5)

