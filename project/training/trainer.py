# training/trainer.py

from typing import List, Dict, Tuple, Optional, Any

from pathlib import Path
import numpy as np
import torch

from ..common import utils, fileio
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
        test_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        evaluator: evaluation.Evaluator,
        csv_path: Path,
        callbacks: Optional[List['Callback']] = None,
        output_dir: str = 'checkpoints',
        device: str = 'cuda',
        bc_spec: Any = None
    ):
        self.task = task

        self.model = model.to(device)
        self.optimizer = optimizer
        self.evaluator = evaluator

        self.adapter = adapter
        self.bc_spec = bc_spec

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader

        self.callbacks = callbacks or []

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(exist_ok=True, parents=True)

        self.device = device

        self.metrics = []  # accumulated eval metrics
        self.epoch = 0     # number of complete epochs
        self.step = 0      # number of optimizer steps

    # ----- training loop / epochs / phases -----

    def train(
        self,
        num_epochs: int,
        val_interval: int = 1,
        test_interval: int = 5,
        save_interval: int = 5,
    ):
        self.start_train()

        while self.epoch < num_epochs:
            self.start_epoch()

            if self.check_epoch(save_interval):
                self.save_checkpoint()

            if self.check_epoch(test_interval):
                self.run_test_phase()

            if self.check_epoch(val_interval):
                self.run_val_phase()

            self.run_train_phase()

            if self.check_epoch(save_interval):
                self.save_metrics()

            self.end_epoch()

        self.save_checkpoint()
        self.run_test_phase()
        self.run_val_phase()
        self.save_metrics()

        self.end_train()

    def check_epoch(self, interval: int) -> bool:
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
        self.start_phase(phase)

        eval_on_train = not (self.test_loader or self.val_loader)
        eval_mode = eval_on_train or not train_mode

        self.model.train(train_mode)

        for idx, batch in enumerate(data_loader):
            self.start_batch(phase, idx, batch)

            if train_mode:
                self.optimizer.zero_grad(set_to_none=True)

            loss, outputs = self.run_batch(batch, eval_mode)

            if train_mode:
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Invalid training loss: {loss.item()}')

                loss.backward()
                grad_norm = _compute_grad_norm(self.model)

                if not torch.isfinite(grad_norm):
                    raise RuntimeError(f'Invalid gradient norm: {grad_norm.item()}')

                outputs['grad_norm'] = grad_norm.detach().cpu()

                self.optimizer.step()
                self.step += 1

            base = {
                'epoch': self.epoch,
                'phase': phase,
                'batch': idx,
                'loss': loss.item()
            }
            if train_mode:
                base['grad_norm'] = grad_norm.item()

            if eval_mode:
                for vals in outputs['metrics']:
                    self.metrics.append({**base, **vals})
            else:
                self.metrics.append(base)

            self.end_batch(phase, idx, outputs)

            del loss, outputs, batch

        self.end_phase(phase)

    # ----- batch forward pipeline -----

    def run_example(self, ex):
        raise NotImlementedError

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

        if eval_mode and self.evaluator is not None:
            outputs['metrics'] = self.evaluator.evaluate_batch(batch, preds, sims)

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

    # ----- saving / loading state -----

    def save_metrics(self):
        fileio.save_csv(self.csv_path, self.metrics, index=False)

    def load_metrics(self):
        self.metrics = fileio.load_csv(self.csv_path).to_dict('records')

    def save_checkpoint(self, path: Optional[Path] = None):
        if path is not None:
            path = Path(path)
        else:
            path = self.checkpoint_path(self.epoch)
        utils.log(f'Saving {path}')
        torch.save({
            'model': self.model.state_dict(),
            'optim': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'step':  self.step
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
            path = self.checkpoint_path(epoch)
        else:
            path = self.last_checkpoint(by_mtime)

        if not path.is_file():
            raise FileNotFoundError(f'Checkpoint not found: {path}')

        utils.log(f'Loading {path}')
        state = torch.load(path, map_location=self.device)

        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optim'])
        self.epoch = int(state.get('epoch', 0))
        self.step = int(state.get('step', 0))

    def checkpoint_path(self, epoch: int) -> Path:
        return self.output_dir / f'checkpoint{epoch:05d}.pt'

    def last_checkpoint(self, by_mtime: bool = False) -> Path:
        found = self.find_checkpoints(by_mtime)
        if len(found) > 0:
            return found[-1]
        raise FileNotFoundError('No checkpoints found')

    def find_checkpoints(self, by_mtime: bool = False) -> List[Path]:
        import re
        name_regex = re.compile(r'^checkpoint(\d{5})\.pt$')
        def match(path):
            return path.is_file() and name_regex.match(path.name)
        found = [p for p in self.output_dir.iterdir() if match(p)]
        if by_mtime:
            sort_key = lambda path: path.stat().st_mtime
        else:
            sort_key = lambda path: path.name
        return sorted(found, key=sort_key)

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
        for cb in self.callbacks:
            cb.on_phase_start(self.epoch, phase)

    def end_phase(self, phase: str):
        for cb in self.callbacks:
            cb.on_phase_end(self.epoch, phase)

    def start_batch(self, phase: str, batch_idx: int, batch: dict):
        for cb in self.callbacks:
            cb.on_batch_start(self.epoch, phase, batch_idx, batch)

    def end_batch(self, phase: str, batch_idx: int, outputs: dict):
        for cb in self.callbacks:
            cb.on_batch_end(self.epoch, phase, batch_idx, outputs)


@torch.no_grad()
def _compute_grad_norm(model):
    norm2 = torch.zeros((), dtype=torch.float, device='cpu')
    for param in model.parameters():
        if param.grad is not None:
            norm2 += param.grad.pow(2).sum().cpu()
    return norm2.pow(0.5)

