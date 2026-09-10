# training/trainer.py

from typing import List, Dict, Tuple, Optional, Any

from pathlib import Path
import numpy as np
import torch

from ..core import utils
from .. import physics, evaluation
from . import tasks, losses


class Trainer:

    def __init__(
        self,
        task: tasks.TaskSpec,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        adapter: physics.adapter.PhysicsAdapter,
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

        self.adapter = adapter
        self.bc_spec = bc_spec

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader

        self.callbacks = callbacks or []

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.device = device

        self.epoch = 0  # number of complete epochs
        self.step = 0   # number of optimizer steps

    # ----- training loop / epochs / phases -----

    def train(
        self,
        num_epochs: int,
        val_interval: int = 1,
        test_interval: int = 5,
        save_interval: int = 5,
    ):
        self._start_train()

        while self.epoch < num_epochs:
            self._start_epoch()

            if self._check_epoch(save_interval):
                self.save_checkpoint()

            if self._check_epoch(test_interval):
                self.run_test_phase()

            if self._check_epoch(val_interval):
                self.run_val_phase()

            self.run_train_phase()
            self._end_epoch()

        self.save_checkpoint()
        self.run_test_phase()
        self.run_val_phase()

        self._end_train()

    def _check_epoch(self, interval: int) -> bool:
        return interval > 0 and self.epoch % interval == 0

    def run_train_phase(self):
        if self.train_loader:
            return self.run_phase(self.train_loader, phase='train', train_mode=True)

    @torch.no_grad()
    def run_test_phase(self):
        if self.test_loader:
            return self.run_phase(self.test_loader, phase='test')

    @torch.no_grad()
    def run_val_phase(self):
        if self.val_loader:
            return self.run_phase(self.val_loader, phase='val')

    def run_phase(self, data_loader, phase: str, train_mode: bool = False):
        self._start_phase(phase, train_mode)

        for idx, batch in enumerate(data_loader):
            self._start_batch(phase, idx)

            loss, outputs = self.run_batch(batch, eval_mode=not train_mode)

            if train_mode: # backward pass and update
                self.optimizer.zero_grad(set_to_none=True)

                if not torch.isfinite(loss):
                    raise RuntimeError(f'Invalid training loss: {loss.item()}')

                loss.backward()
                grad_norm = _compute_grad_norm(self.model)

                if not torch.isfinite(grad_norm):
                    raise RuntimeError(f'Invalid gradient norm: {grad_norm.item()}')

                outputs['grad_norm'] = grad_norm.detach().cpu()

                self.optimizer.step()
                self.step += 1

            self._end_batch(phase, idx, outputs)

        self._end_phase(phase)

    # ----- batch forward pipeline -----

    def run_batch(self, batch, eval_mode: bool = False):

        preds = self.run_model(batch)
        sim_loss, sims = self.run_physics(batch, preds, eval_mode)
        loss = self.compute_loss(batch, preds, sim_loss)

        outputs = {
            'batch': batch,
            'preds': preds,
            'sims': sims,
            'loss': loss.detach().cpu()
        }

        return loss, outputs

    def run_model(self, batch):
        inputs = self._concat_inputs(batch)
        return self.model(inputs)

    def run_physics(self, batch, preds, eval_mode: bool = False):
        batch_size = len(batch['example'])

        if not self._requires_physics(eval_mode):
            return None, None

        loss = torch.zeros(batch_size, device=self.device, dtype=torch.float)
        outputs = [None] * batch_size

        for idx in range(batch_size):
            loss[idx], outputs[idx] = self.adapter.voxel_simulation_loss(
                mesh=batch['mesh'][idx],
                unit_m=batch['example'][idx].metadata['unit'],
                affine=batch['affine'][idx],
                params=self._select_params(preds, idx),
                bc_spec=self.bc_spec,
                ret_outputs=eval_mode
            )

        return loss, outputs

    def compute_loss(self, batch, preds, sim_loss=None):
        mask = batch['mask'].to(self.device)

        total_loss = torch.zeros((), device=self.device, dtype=torch.float)
        sim_added

        for target in self.task.loss_targets:
            y_pred = preds[target].to(self.device)

            loss_name = self.task.losses[target].upper()
            loss_weight = self.task.weights.get(target, 1.0)

            if loss_name != 'SIM':
                y_true = batch[target].to(self.device)

            if loss_name == 'CE':
                loss = losses.masked_cross_entropy(y_pred, y_true, mask)

            elif loss_name == 'MSE':
                loss = losses.mean_squared_error(y_pred, y_true, mask)

            elif loss_name == 'MSRE':
                loss = losses.mean_squared_relative_error(y_pred, y_true, mask)

            elif loss_name == 'SIM':
                if sim_loss is None:
                    raise RuntimeError('Expected sim_loss to be provided')
                loss = sim_loss.mean()

            else:
                raise ValueError(f'Unknown loss: {loss_name}')

            total_loss = total_loss + loss_weight * loss

        return total_loss

    # ----- task-specific internals -----

    def _concat_inputs(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_vals = [batch[key] for key in self.task.inputs]
        return torch.cat(input_vals, dim=1).to(self.device)

    def _requires_physics(self, eval_mode: bool = False) -> bool:
        need_loss = self.task.has_physics_loss
        need_eval = self.task.has_physics_target and eval_mode
        return need_loss or need_eval

    def _select_params(self, preds: Dict[str, torch.Tensor], idx: int) -> dict:
        param_keys = self.task.physics_targets
        param_vals = [preds[key][idx] for key in param_keys]
        return dict(zip(param_keys, param_vals))

    # ----- saving / loading checkpoints -----

    def save_checkpoint(self, path: Optional[Path] = None):
        if path is not None:
            path = Path(path)
        else:
            path = self._checkpoint_path(self.epoch)

        utils.log(f'Saving {path}')

        torch.save({
            'epoch': self.epoch,
            'step':  self.step,
            'model': self.model.state_dict(),
            'optim': self.optimizer.state_dict()
        }, path)

    def load_checkpoint(
        self,
        path: Optional[Path] = None,
        epoch: Optional[int] = None,
        by_mtime: bool = False
    ):
        if path is not None:
            path = Path(path)
        elif epoch is not None:
            path = self._checkpoint_path(epoch)
        else:
            path = self._last_checkpoint(by_mtime)

        if not path.is_file():
            raise FileNotFoundError(f'Checkpoint not found: {path}')

        utils.log(f'Loading {path}')
        state = torch.load(path, map_location=self.device)

        self.epoch = int(state.get('epoch', 0))
        self.step = int(state.get('step', 0))

        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optim'])

    def _checkpoint_path(self, epoch: int) -> Path:
        return self.output_dir / f'checkpoint{epoch:05d}.pt'

    def _last_checkpoint(self, by_mtime: bool = False) -> Path:
        found = self._find_checkpoints(by_mtime)
        if len(found) > 0:
            return found[-1]
        raise FileNotFoundError('No checkpoints found')

    def _find_checkpoints(self, by_mtime: bool = False) -> List[Path]:
        import re
        pat = re.compile(r'^checkpoint(\d{5})\.pt$')

        def _is_ckpt(p):
            return p.is_file() and pat.match(p.name)

        paths = [p for p in self.output_dir.iterdir() if _is_ckpt(p)]

        if by_mtime:
            sort_key = lambda p: p.stat().st_mtime
        else:
            sort_key = lambda p: p.name

        return sorted(paths, key=sort_key)

    # ----- callback hooks -----

    def _start_train(self):
        for cb in self.callbacks:
            cb.on_train_start()

    def _end_train(self):
        for cb in self.callbacks:
            cb.on_train_end()

    def _start_epoch(self):
        for cb in self.callbacks:
            cb.on_epoch_start(self.epoch)

    def _end_epoch(self):
        for cb in self.callbacks:
            cb.on_epoch_end(self.epoch)
        self.epoch += 1

    def _start_phase(self, phase: str, train_mode: str):
        self.model.train() if train_mode else self.model.eval()
        for cb in self.callbacks:
            cb.on_phase_start(self.epoch, phase)

    def _end_phase(self, phase: str):
        for cb in self.callbacks:
            cb.on_phase_end(self.epoch, phase)

    def _start_batch(self, phase: str, batch_idx: int):
        for cb in self.callbacks:
            cb.on_batch_start(self.epoch, phase, batch_idx)

    def _end_batch(self, phase: str, batch_idx: int, outputs: dict):
        for cb in self.callbacks:
            cb.on_batch_end(self.epoch, phase, batch_idx, outputs)


@torch.no_grad()
def _compute_grad_norm(model):
    norm2 = torch.zeros((), dtype=torch.float, device='cpu')
    for param in model.parameters():
        if param.grad is not None:
            norm2 += param.grad.pow(2).sum().cpu()
    return norm2.pow(0.5)

