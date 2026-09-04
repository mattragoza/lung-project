from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from ..core import utils

EPS = 1e-12


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
        device: str = 'cuda',
        bc_spec: Any = None
    ):
        self.task = task

        self.model = model.to(device)
        self.optimizer = optimizer

        self.phys_adapter = phys_adapter
        self.bc_spec = bc_spec

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader

        self.callbacks = callbacks or []

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.device = device

        self.epoch = 0 # number of complete epochs
        self.step = 0 # number of optimizer steps

    # ----- training loop / epochs / phases -----

    def train(self, num_epochs, val_every=1, test_every=5, save_every=5):
        self.start_training()
        init_step = self.step

        while self.epoch < num_epochs:
            self.start_epoch()

            if self.output_dir and self._check_epoch(save_every):
                self.save_state()

            if self.test_loader and self._check_epoch(test_every):
                self.run_test_phase()

            if self.val_loader and self._check_epoch(val_every):
                self.run_val_phase()

            self.run_train_phase()
            self.end_epoch()

        if self.output_dir and self.step > init_step:
            self.save_state()

        if self.val_loader:
            self.run_val_phase()

        if self.test_loader:
            self.run_test_phase()

        self.end_training()

    def _check_epoch(self, interval: int) -> bool:
        return interval > 0 and self.epoch % interval == 0

    def run_train_phase(self):
        return self.run_phase(self.train_loader, phase='train', train_mode=True)

    @torch.no_grad()
    def run_test_phase(self):
        return self.run_phase(self.test_loader, phase='test')

    @torch.no_grad()
    def run_val_phase(self):
        return self.run_phase(self.val_loader, phase='val')

    def run_phase(self, data_loader, phase: str, train_mode: bool = False):
        self.start_phase(phase)

        for batch_idx, batch in enumerate(data_loader):
            self.start_batch(phase, batch_idx)

            loss, outputs = self.run_batch(batch, eval_mode=not train_mode)

            if train_mode:
                self.optimizer.zero_grad(set_to_none=True)

                if not torch.isfinite(loss):
                    raise RuntimeError(f'Invalid training loss: {loss.item()}')

                loss.backward()
                grad_norm = _compute_grad_norm(self.model)

                if not torch.isfinite(grad_norm):
                    raise RuntimeError(f'Invalid gradient norm: {grad_norm.item()}')

                outputs['grad_norm'] = grad_norm

                self.optimizer.step()
                self.step += 1

            self.end_batch(phase, batch_idx, outputs)

        self.end_phase(phase)

    # ----- batch pipeline -----

    def run_batch(self, batch, eval_mode: bool = False):

        preds = self.run_model(batch)
        sim_loss, sim_outputs = self.run_physics(batch, preds, eval_mode)

        loss, loss_base = self.compute_loss(batch, preds, sim_loss)

        outputs = self._package_outputs(
            batch, preds, sim_outputs, loss, loss_base, eval_mode
        )
        return loss, outputs

    def run_model(self, batch):
        inputs = self._prepare_inputs(batch)
        return self.model(inputs)

    def run_physics(self, batch, preds, eval_mode: bool = False):
        batch_size = len(batch['example'])

        if not self._requires_physics(eval_mode):
            return None, None

        loss = torch.zeros(batch_size, device=self.device, dtype=torch.float)
        outputs = [None] * batch_size

        for idx in range(batch_size):
            loss[idx], outputs[idx] = self.phys_adapter.voxel_simulation_loss(
                mesh=batch['mesh'][idx],
                unit_m=batch['example'][idx].metadata['unit'],
                affine=batch['affine'][idx],
                params=self._prepare_params(batch, preds, idx),
                bc_spec=self.bc_spec,
                ret_outputs=True # no extra work
            )

        return loss, outputs

    def compute_loss(self, batch, preds, sim_loss=None):
        mask = batch['mask'].to(self.device)

        total_loss = torch.zeros((), device=self.device)
        total_base = torch.zeros((), device=self.device)

        for target in self.task.loss_targets:
            output_key = self.task.output_key(target)
            target_key = self.task.target_key(target)

            y_pred = preds[output_key].to(self.device)

            loss_name = self.task.losses[target].lower()
            loss_weight = self.task.weights.get(target, 1.0)

            if loss_name == 'ce':
                y_true = batch[target_key].to(self.device)
                y_base = torch.zeros_like(y_pred) # uniform logits
                loss = masked_cross_entropy(y_pred, y_true, mask)
                base = masked_cross_entropy(y_base, y_true, mask)

            elif loss_name == 'mse':
                y_true = batch[target_key].to(self.device)
                y_base = torch.full_like(y_pred, y_true.mean())
                loss = mean_squared_error(y_pred, y_true, mask)
                base = mean_squared_error(y_base, y_true, mask)

            elif loss_name == 'msre':
                y_true = batch[target_key].to(self.device)
                y_base = torch.full_like(y_pred, y_true.mean())
                loss = mean_squared_relative_error(y_pred, y_true, mask)
                base = mean_squared_relative_error(y_base, y_true, mask)

            elif loss_name == 'sim':
                if sim_loss is None:
                    raise RuntimeError('Simulation loss not provided')
                loss = sim_loss.mean()
                base = 1.0 # TODO

            else:
                raise ValueError(f'Unknown loss: {loss_name}')

            total_loss = total_loss + loss_weight * loss
            total_base = total_base + loss_weight * base

        return total_loss, total_base

    # ----- task-specific internals -----

    def _prepare_inputs(self, batch: Dict[str, torch.Tensor]):
        input_names = self.task.inputs
        input_keys = [self.task.input_key(name) for name in input_names]
        input_vals = [batch[key].to(self.device) for key in input_keys]
        return torch.cat(input_vals, dim=1)

    def _requires_physics(self, eval_mode: bool):
        need_sim_loss = self.task.has_physics_loss
        need_sim_outputs = self.task.has_physics_output and eval_mode
        return need_sim_loss or need_sim_outputs

    def _prepare_params(self, preds: Dict[str, torch.Tensor], idx: int):
        param_names = self.task.physics_outputs
        param_keys = [self.task.output_key(name) for name in param_names]
        param_vals = [preds[key][idx] for key in param_keys]
        return dict(zip(param_names, param_vals))

    def _package_outputs(
        self,
        batch,
        preds,
        sim_outputs,
        loss,
        loss_base,
        eval_mode: bool = False
    ):
        outputs = {
            'example': batch['example'],
            'loss': loss.detach().cpu(),
            'loss_base': loss_base.detach().cpu()
        }

        if not eval_mode:
            return outputs

        outputs['mask'] = batch['mask'].detach().cpu()
        outputs['mat_true'] = batch['mat_true'].detach().cpu()

        for input_ in self.task.inputs:
            input_key = self.task.input_key(input_)
            outputs[input_key] = batch[input_key].detach().cpu()

        for target in self.task.targets:
            output_key = self.task.output_key(target)
            outputs[output_key] = preds[output_key].detach().cpu()

            target_key = self.task.target_key(target)
            if target_key in batch:
                outputs[target_key] = batch[target_key].detach().cpu()

        if sim_outputs is not None:
            outputs['sim'] = sim_outputs

        return outputs

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

    # ----- callback hooks -----

    def start_training(self):
        for cb in self.callbacks:
            cb.on_train_start()

    def end_training(self):
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
        self.model.train() if phase.lower() == 'train' else self.model.eval()
        for cb in self.callbacks:
            cb.on_phase_start(self.epoch, phase)

    def end_phase(self, phase: str):
        for cb in self.callbacks:
            cb.on_phase_end(self.epoch, phase)

    def start_batch(self, phase: str, batch_idx: int):
        for cb in self.callbacks:
            cb.on_batch_start(self.epoch, phase, batch_idx)

    def end_batch(self, phase: str, batch_idx: int, outputs: dict):
        for cb in self.callbacks:
            cb.on_batch_end(self.epoch, phase, batch_idx, outputs)


@torch.no_grad()
def _compute_grad_norm(model):
    norm2 = torch.zeros((), device='cpu')
    for p in model.parameters():
        if p.grad is not None:
            norm2 += p.grad.pow(2).sum().cpu()
    return norm2.pow(0.5)

