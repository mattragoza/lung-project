from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

from .core import utils, transforms

def _to_numpy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


class Trainer:

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        physics_adapter: project.physics.PhysicsAdapter,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader=None,
        val_loader: torch.utils.data.DataLoader=None,
        callbacks: torch.utils.data.DataLoader=None,
        train_mode: str='u_sim',
        output_dir='checkpoints',
        device='cuda'
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.physics_adapter = physics_adapter

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.callbacks = callbacks or []

        assert train_mode in {'u_sim', 'E_reg', 'logE_reg', 'mat_seg'}
        self.train_mode = train_mode

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

        image = batch['image'].to(device, dtype=torch.float)
        mask  = batch['mask'].to(device, dtype=torch.bool)

        preds = self.model.forward(image)

        outputs = {
            'example':  batch['example'],
            'image':    batch['image'].cpu(),
            'mask':     batch['mask'].cpu(),
            'mat_true': batch['material'].cpu()
        }

        need_physics_loss = (self.train_mode == 'u_sim')
        need_physics_eval = (self.train_mode != 'mat_seg') and eval_mode
        run_physics = need_physics_loss or need_physics_eval

        sim_loss = None
        if run_physics: # compute displacement error via physics simulation
            sim_loss = torch.zeros(batch_size, device=self.device, dtype=image.dtype)
            pde_outputs = [None] * batch_size

            for k in range(batch_size):
                sim_loss[k], pde_outputs[k] = self.physics_adapter.voxel_simulation_loss(
                    mesh=batch['mesh'][k],
                    unit_m=batch['example'][k].metadata['unit'],
                    affine=batch['affine'][k],
                    E_vox=preds['E'][k],
                    bc_spec=None
                )

            outputs['sim_loss'] = sim_loss.mean().detach().cpu()
            outputs['pde'] = pde_outputs

        # mode determines loss function + eval outputs
        if self.train_mode == 'u_sim':
            outputs['loss'] = sim_loss.mean()
            outputs['E_pred'] = preds['E'].detach().cpu()
            if 'E' in batch:
                outputs['E_true'] = batch['E'].detach().cpu()

        elif self.train_mode == 'E_reg':
            E_pred = preds['E'].to(device)
            E_true = batch['E'].to(device)
            outputs['loss'] = normalized_rmse(E_pred, E_true, mask)
            outputs['E_pred'] = E_pred.detach().cpu()
            outputs['E_true'] = E_true.detach().cpu()

        elif self.train_mode == 'logE_reg':
            logE_pred = preds['logE'].to(device)
            logE_true = batch['logE'].to(device)
            outputs['loss'] = normalized_rmse(logE_pred, logE_true, mask)
            outputs['E_pred'] = preds['E'].detach().cpu()
            outputs['E_true'] = batch['E'].detach().cpu()

        elif self.train_mode == 'mat_seg':
            mat_logits = preds['logits'].to(device)
            mat_true = batch['material'].to(device)
            outputs['loss'] = masked_cross_entropy(mat_logits, mat_true, mask)
            outputs['mat_logits'] = mat_logits.detach().cpu()

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


def masked_cross_entropy(pred, target, mask, ignore_index=-1):
    '''
    Args:
        pred: (B, C, I, J, K) foreground material logits.
            Channel i corresponds to label (i+1) in the target.

        target: (B, 1, I, J, K) integer material labels.
            Value 0 indicates background, 1..C are material IDs.

        mask: (B, 1, I, J, K) boolean foreground mask.
            True for voxels inside the domain (i.e. target > 0).
    '''
    assert target.min() >= 0
    assert target.max() <= pred.shape[1]

    if mask.dim() == 5:
        mask = mask.squeeze(1)
    mask = mask.bool()

    if target.dim() == 5:
        target = target.squeeze(1)

    shifted = (target - 1).long()
    shifted[~mask] = ignore_index

    loss_vox = F.cross_entropy(
        pred, shifted, reduction='none', ignore_index=ignore_index
    )
    return loss_vox[mask].mean()


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

