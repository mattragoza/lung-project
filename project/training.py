from __future__ import annotations
from typing import List, Dict, Iterable
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from .core import utils, transforms


def _to_numpy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


class TaskSpec:
    valid_inputs  = {'image', 'material', 'mask'}
    valid_physics = {'E', 'nu', 'G', 'K', 'mu', 'lam', 'rho'}
    valid_targets = {'image', 'material'} | valid_physics
    valid_losses  = {'ce', 'mse', 'msre', 'sim'}

    def __init__(
        self,
        inputs:  List[str],
        targets: List[str],
        losses:  Dict[str, str],
        weights: Dict[str, float] = None,
        n_mat_labels: int = 5,
        rgb: bool = False
    ):
        self.inputs  = list(inputs)
        self.targets = list(targets)
        self.losses  = dict(losses)
        self.weights = dict(weights or {})

        self.n_mat_labels = n_mat_labels
        self.rgb = rgb

        self._validate()

    def _validate(self):

        utils.log(f'Inputs:  {self.inputs}')
        utils.log(f'Targets: {self.targets}')
        utils.log(f'Losses:  {self.losses}')

        assert len(self.inputs)  > 0
        assert len(self.targets) > 0
        assert len(self.losses)  > 0

        for input_ in self.inputs:
            if input_ not in self.valid_inputs:
                raise ValueError(f'Invalid input: {input_}')

        for target in self.targets:
            if target not in self.valid_targets:
                raise ValueError(f'Invalid target: {target}')

        for target, loss in self.losses.items():
            loss_l = loss.lower()
            if target not in self.targets:
                raise ValueError(f'Invalid loss target: {target}')
            if loss_l not in self.valid_losses:
                raise ValueError(f'Invalid loss function: {loss}')
            if loss_l == 'sim' and target not in self.valid_physics:
                raise ValueError(f'Invalid physics target: {target}')

        for target, weight in self.weights.items():
            if target not in self.losses:
                raise ValueError(f'Invalid weight target: {target}')
            if weight < 0.0:
                raise ValueError(f'Invalid weight value: {weight}')

    @property
    def physics_outputs(self) -> List[str]:
        return [t for t in self.targets if t in self.valid_physics]

    @property
    def has_physics_output(self) -> bool:
        return len(self.physics_outputs) > 0

    @property
    def has_physics_loss(self) -> bool:
        return any(l.lower() == 'sim' for l in self.losses.values())

    @property
    def image_channels(self) -> int:
        return 3 if self.rgb else 1

    @property
    def material_labels(self) -> int:
        return self.n_mat_labels

    @property
    def in_channels(self) -> int:
        total = 0
        for input_ in self.inputs:
            if input_ == 'image':
                total += self.image_channels
            elif input_ == 'material':
                total += self.material_labels + 1
            elif input_ == 'mask':
                total += 1
            else:
                raise ValueError(input_)
        return total

    def out_channels(self, target: str) -> int:
        if target == 'image':
            return self.image_channels
        elif target == 'material':
            return self.material_labels + 1
        elif target in self.valid_physics:
            return 1
        raise ValueError(target)

    def input_key(self, input_: str, visual: bool=False) -> str:
        if input_ == 'image':
            return 'img_true'
        elif input_ == 'material':
            return 'mat_true' if visual else 'mat_onehot'
        elif input_ == 'mask':
            return 'mask'
        raise ValueError(input_)

    def output_key(self, target: str, visual: bool=False) -> str:
        if target == 'image':
            return 'img_pred'
        elif target == 'material':
            return 'mat_pred' if visual else 'mat_logits'
        elif target in self.valid_physics:
            return f'{target}_pred'
        raise ValueError(target)

    def target_key(self, target: str, visual: bool=False) -> str:
        if target == 'image':
            return 'img_true'
        elif target == 'material':
            return 'mat_true'
        elif target in self.valid_physics:
            return f'{target}_true'
        raise ValueError(target)

    def base_value(self, target: str) -> float:
        if target == 'E':
            return 3.4863 # logE mean
        return 0.0

    @property
    def metric_keys(self) -> List[str]:
        keys = ['loss', 'loss_base', 'loss_ratio', 'grad_norm']
        for tgt in self.targets:
            output_key = self.output_key(tgt)
            target_key = self.target_key(tgt)
            keys.append(output_key + '.mean')
            keys.append(output_key + '.std')
            keys.append(target_key + '.mean')
            keys.append(target_key + '.std')
        return keys 

    @property
    def viewer_keys(self) -> List[str]:
        keys = []
        for inp in self.inputs:
            input_key = self.input_key(inp, visual=True)
            keys.append(input_key)
        for tgt in self.targets:
            output_key = self.output_key(tgt, visual=True)
            target_key = self.target_key(tgt, visual=True)
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
        output_dir='checkpoints',
        device='cuda'
    ):
        self.task = task
        self.model = model.to(device)
        self.optimizer = optimizer
        self.physics_adapter = physics_adapter

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
                sim_loss[k], sim_outputs[k] = self.physics_adapter.voxel_simulation_loss(
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


@torch.no_grad()
def param_grad_norm(model):
    norm2 = torch.zeros((), device='cpu')
    for p in model.parameters():
        if p.grad is not None:
            norm2 += p.grad.pow(2).sum().cpu()
    return norm2.pow(0.5)


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
    Root mean squared error.

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
    Normalized root mean squared error.

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
    Masked cross entropy.

    Args:
        pred: (B, C, I, J, K) predicted material logits.
        target: (B, 1, I, J, K) integer material labels.
        mask: (B, 1, I, J, K) boolean foreground mask.
    '''
    assert target.min() >= 0
    assert target.max() < pred.shape[1], pred.shape

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

