from __future__ import annotations
from typing import List, Dict, Iterable
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

from .core import utils, transforms

def _to_numpy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


class TaskSpec:
    image_channels = 1
    material_labels = 5

    def __init__(
        self,
        inputs: List[str],
        targets: List[str],
        losses: Dict[str, str],
        weights: Dict[str, float]=None
    ):
        self.inputs  = inputs
        self.targets = targets
        self.losses  = losses
        self.weights = weights or {}
        self._validate()

    def _validate(self):
        for input_ in self.inputs:
            assert input_ in {'image', 'material'}, input_
        for target in self.targets:
            assert target in {'image', 'material', 'E', 'logE'}, target
        for target, loss in self.losses.items():
            assert target in self.targets, (target, loss)
            assert loss.lower() in {'ce', 'mse', 'msre', 'sim'}, (target, loss)
        for target, weight in self.weights.items():
            assert target in self.targets, (target, weight)
            assert weight >= 0.0, (target, weight)

    @property
    def in_channels(self) -> int:
        total = 0
        for input_ in self.inputs:
            if input_ == 'image':
                total += self.image_channels
            elif input_ == 'material':
                total += self.material_labels + 1
            else:
                raise ValueError(input_)
        return total

    def out_channels(self, target: str) -> int:
        if target == 'image':
            return self.image_channels
        elif target == 'material':
            return self.material_labels + 1
        elif target in {'E', 'logE'}:
            return 1
        raise ValueError(target)

    def input_key(self, input_: str, visual: bool=False) -> str:
        if input_ == 'image':
            return 'img_true'
        elif input_ == 'material':
            return 'mat_true' if visual else 'mat_onehot'
        raise ValueError(input_)

    def input_keys(self, *args, **kwargs) -> List[str]:
        return [self.input_key(x, *args, **kwargs) for x in self.inputs]

    def output_key(self, target: str, visual: bool=False) -> str:
        if target == 'E':
            return 'E_pred'
        elif target == 'logE':
            return 'logE_pred'
        elif target == 'image':
            return 'img_pred'
        elif target == 'material':
            return 'mat_pred' if visual else 'mat_logits'
        raise ValueError(target)

    def target_key(self, target: str, visual: bool=False) -> str:
        assert target in self.targets, target
        if target == 'E':
            return 'E_true'
        elif target == 'logE':
            return 'logE_true'
        elif target == 'image':
            return 'img_true'
        elif target == 'material':
            return 'mat_true'
        raise ValueError(target)

    def base_value(self, target: str) -> float:
        if target == 'E':
            return 10**3.4863
        elif target == 'logE':
            return 3.4863
        return 0.0

    @property
    def plotter_keys(self) -> List[str]:
        return ['loss', 'loss_ratio']

    @property
    def viewer_keys(self) -> List[str]:
        keys = self.input_keys(visual=True)
        for t in self.targets:
            output_key = self.output_key(t, visual=True)
            target_key = self.target_key(t, visual=True)
            keys.extend([output_key, target_key])
        return keys


class Trainer:

    def __init__(
        self,
        task: TaskSpec,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        physics_adapter: project.physics.PhysicsAdapter,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader=None,
        val_loader: torch.utils.data.DataLoader=None,
        callbacks: torch.utils.data.DataLoader=None,
        eval_physics=True,
        output_dir='checkpoints',
        device='cuda'
    ):
        self.task = task
        self.model = model.to(device)
        self.optimizer = optimizer
        self.physics_adapter = physics_adapter
        self.eval_physics = eval_physics

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.callbacks = callbacks or []

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.device = device

        self.epoch = 0 # number of complete epochs
        self.step  = 0 # number of optimizer steps

    # ----- training loop / phases -----

    def train(self, num_epochs, val_every=1, save_every=10):
        self.start_train()

        for ep in range(num_epochs):
            self.start_epoch()

            if self.output_dir and save_every and self.epoch % save_every == 0:
                self.save_state()

            if self.val_loader and val_every and self.epoch % val_every == 0:
                self.run_val_phase()

            self.run_train_phase()
            self.end_epoch()

        if self.output_dir:
            self.save_state()

        if self.val_loader:
            self.run_val_phase()

        if self.test_loader:
            self.run_test_phase()

        self.end_train()

    def run_train_phase(self):
        self.start_phase(phase='train')

        for i, batch in enumerate(self.train_loader):
            self.start_batch(phase='train', batch=i, loader=self.train_loader)

            self.optimizer.zero_grad()
            outputs = self.forward(batch)

            loss = outputs['loss']
            if not torch.isfinite(loss):
                raise RuntimeError(f'Invalid loss: {loss.item()}')

            loss.backward()
            self.optimizer.step()
            self.step += 1

            self.end_batch(phase='train', batch=i, outputs=outputs)

        self.end_phase(phase='train')

    @torch.no_grad()
    def run_test_phase(self):
        self.start_phase(phase='test')

        for i, batch in enumerate(self.test_loader):
            self.start_batch(phase='test', batch=i, loader=self.test_loader)

            outputs = self.forward(batch, eval_mode=True)

            self.end_batch(phase='test', batch=i, outputs=outputs)

        self.end_phase(phase='test')

    @torch.no_grad()
    def run_val_phase(self):
        self.start_phase(phase='val')

        for i, batch in enumerate(self.val_loader):
            self.start_batch(phase='val', batch=i, loader=self.val_loader)

            outputs = self.forward(batch, eval_mode=True)

            self.end_batch(phase='val', batch=i, outputs=outputs)

        self.end_phase(phase='val')

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

    def start_batch(self, phase: str, batch: int, loader):
        for cb in self.callbacks:
            cb.on_batch_start(self.epoch, phase, batch, self.step, loader=loader)

    def end_batch(self, phase: str, batch: int, outputs):
        for cb in self.callbacks:
            cb.on_batch_end(self.epoch, phase, batch, self.step, outputs=outputs)

    # ----- forward pass -----

    def forward(self, batch, eval_mode=False):
        device = self.device
        batch_size = len(batch['example'])

        mask = batch['mask'].to(device, dtype=torch.bool)

        input_keys = self.task.input_keys()
        input_vals = [batch[k].to(device, dtype=torch.float) for k in input_keys]
        inputs_cat = torch.cat(input_vals, dim=1)

        preds = self.model.forward(inputs_cat)

        outputs = {
            'example':  batch['example'],
            'mask':     batch['mask'].cpu(),
            'mat_true': batch['mat_true'].cpu()
        }
        for k in input_keys:
            outputs[k] = batch[k].cpu()

        need_physics_loss = ('sim' in self.task.losses)
        need_physics_eval = (self.eval_physics and eval_mode)
        run_physics = need_physics_loss or need_physics_eval

        sim_loss = None
        if run_physics: # compute displacement error via physics simulation
            sim_loss = torch.zeros(batch_size, device=self.device, dtype=torch.float)
            pde_outputs = [None] * batch_size

            for k in range(batch_size):
                sim_loss[k], pde_outputs[k] = self.physics_adapter.voxel_simulation_loss(
                    mesh=batch['mesh'][k],
                    unit_m=batch['example'][k].metadata['unit'],
                    affine=batch['affine'][k],
                    E_vox=preds['E_pred'][k],
                    bc_spec=None
                )

            outputs['sim_loss'] = sim_loss.mean().detach().cpu()
            outputs['pde'] = pde_outputs

        total_loss = torch.zeros((), device=device)
        total_base = torch.zeros((), device=device)

        for t in self.task.targets:
            output_key  = self.task.output_key(t)
            target_key  = self.task.target_key(t)

            x_pred = preds[output_key].to(device)
            x_true = batch[target_key].to(device)
            x_base = torch.full_like(x_pred, self.task.base_value(t), dtype=torch.float)

            loss_name   = self.task.losses[t].lower()
            loss_weight = self.task.weights.get(t, 1.0)

            if loss_name == 'ce':
                loss = masked_cross_entropy(x_pred, x_true, mask)
                base = masked_cross_entropy(x_base, x_true, mask)

            elif loss_name == 'mse':
                loss = mean_squared_error(x_pred, x_true, mask)
                base = mean_squared_error(x_base, x_true, mask)

            elif loss_name == 'msre':
                loss = mean_squared_relative_error(x_pred, x_true, mask)
                base = mean_squared_relative_error(x_base, x_true, mask)

            elif loss_name == 'sim':
                loss = sim_loss.mean()
                base = 0.0
            else:
                raise ValueError(loss_name)

            total_loss = total_loss + loss_weight * loss
            if loss_name != 'u_sim':
                total_base = total_base + loss_weight * base

            outputs[output_key] = preds[output_key].cpu()
            outputs[target_key] = batch[target_key].cpu()

        outputs['loss'] = total_loss
        outputs['loss_base'] = total_base.detach().cpu()
        outputs['loss_ratio'] = (total_loss / total_base.clamp_min(1e-12)).detach().cpu()
        return outputs

    # ----- saving / loading state -----

    def save_state(self, path=None):
        if path is None:
            path = self.output_dir / f'checkpoint{self.epoch:05d}.pt'
        utils.log(f'Saving {path}')
        torch.save({
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'optim': self.optimizer.state_dict()
        }, path)

    def load_state(self, path=None, epoch=None):
        if path is None:
            path = self.output_dir / f'checkpoint{epoch:05d}.pt'
        utils.log(f'Loading {path}')
        state = torch.load(path)
        self.epoch = state['epoch']
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optim'])


# ----- basic loss functions -----


def mean_squared_error(pred, target, mask):
    '''
    Mean squared error.

    Args:
        pred:   (B, C, I, J, K) prediction tensor
        target: (B, C, I, J, K) target tensor
        mask:   (B, 1, I, J, K) weight tensor
    '''
    if mask.dim() == 5:
        mask = mask[:,0]
    err = torch.linalg.norm(pred - target, dim=1)
    return torch.mean(err[mask]**2)


def mean_squared_relative_error(pred, target, mask, eps=1e-12):
    '''
    Mean squared relative error.

    Args:
        pred:   (B, C, I, J, K) prediction tensor
        target: (B, C, I, J, K) target tensor
        mask:   (B, 1, I, J, K) weight tensor
    '''
    if mask.dim() == 5:
        mask = mask[:,0]

    err = torch.linalg.norm(pred - target, dim=1)
    mag = torch.linalg.norm(target, dim=1)
    rel_err = err / mag.clamp_min(eps)

    return torch.mean(rel_err[mask]**2)


def rmse(pred, target, mask):
    '''
    RMSE = RMS(||pred - target||)

    Args:
        pred:   (B, C, I, J, K) prediction tensor
        target: (B, C, I, J, K) target tensor
        mask:   (B, 1, I, J, K) weight tensor
    '''
    if mask.dim() == 5:
        mask = mask[:,0]
    err = torch.linalg.norm(pred - target, dim=1)
    return torch.sqrt(torch.mean(err[mask]**2))


def normalized_rmse(pred, target, mask, eps=1e-12):
    '''
    NRMSE = RMS(||pred - target||) / RMS(||target||)

    Args:
        pred:   (B, C, I, J, K) prediction tensor
        target: (B, C, I, J, K) target tensor
        mask:   (B, 1, I, J, K) weight tensor
    '''
    if mask.dim() == 5:
        mask = mask.squeeze(1)
    mask = mask.bool()

    err = torch.linalg.norm(pred - target, dim=1) # (B, I, J, K)
    mag = torch.linalg.norm(target, dim=1)        # (B, I, J, K)

    num = torch.sqrt(torch.mean(err[mask] ** 2))
    den = torch.sqrt(torch.mean(mag[mask] ** 2))

    return num / den.clamp_min(eps)


def masked_cross_entropy(pred, target, mask):
    '''
    Args:
        pred: (B, C, I, J, K) predicted material logits.
        target: (B, 1, I, J, K) integer material labels.
        mask: (B, 1, I, J, K) boolean foreground mask.
    '''
    assert target.min() >= 0
    assert target.max() < pred.shape[1]

    if target.dim() == 5:
        target = target.squeeze(1)
    target = target.long()

    if mask.dim() == 5:
        mask = mask.squeeze(1)
    mask = mask.bool()

    ce = F.cross_entropy(pred, target, reduction='none')
    return torch.mean(ce[mask])


# ----- cross-validation -----


def split_on_metadata(examples, key, test_ratio, val_ratio, seed=0):
    from collections import defaultdict

    utils.log('Splitting examples')

    cats_by_subj = defaultdict(set)
    subjs_by_cat = defaultdict(set)
    for ex in examples:
        subj = ex.subject
        for cat in ex.metadata[key]:
            if cat.startswith('_'):
                continue
            cats_by_subj[subj].add(cat)
            subjs_by_cat[cat].add(subj)

    subjects = list(sorted(cats_by_subj.keys()))
    n_subjects = len(subjects)
    target_test = int(round(test_ratio * n_subjects))

    categories = list(sorted(subjs_by_cat.keys()))
    rng = np.random.default_rng(seed)
    rng.shuffle(categories)

    test_cats = set()
    test_subj = set()

    for cat in categories:
        proposal = test_cats | {cat}
        eligible = {s for s, c in cats_by_subj.items() if c.issubset(proposal)}
        delta = eligible - test_subj
        if len(test_subj) < target_test:
            test_cats.add(cat)
            test_subj |= delta
        else:
            break

    utils.log(f'Test categories: {test_cats}')
    utils.log(f'Test subjects:   {test_subj}')

    train_subj = set(subjects) - test_subj

    num_val = int(round(val_ratio * n_subjects))
    train_list = list(train_subj)
    rng.shuffle(train_list)
    val_subj = set(train_list[:num_val])
    train_subj = set(train_list[num_val:])

    utils.log(f'Train subjects: {train_subj}')
    utils.log(f'Val subjects:   {val_subj}')

    train_ex = [ex for ex in examples if ex.subject in train_subj]
    test_ex  = [ex for ex in examples if ex.subject in test_subj]
    val_ex   = [ex for ex in examples if ex.subject in val_subj]

    return train_ex, test_ex, val_ex

