import time
from pathlib import Path
import numpy as np
import torch

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

from .core import utils

def _to_numpy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        train_loader,
        physics_adapter,
        test_loader=None,
        val_loader=None,
        callbacks=None,
        supervised=False,
        output_dir='checkpoints',
        device='cuda'
    ):
        self.model = model.to(device)
        self.optimizer = optimizer

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.callbacks = callbacks or []

        self.physics_adapter = physics_adapter
        self.supervised = supervised
        self.device = device

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.epoch = 0 # number of complete epochs
        self.step  = 0 # number of optimizer steps

    # ----- training loop / phases -----

    def train(self, num_epochs, val_every=1, save_every=5):
        self.start_train()

        for _ in range(num_epochs):
            self.start_epoch()

            if self.output_dir and save_every and self.epoch % save_every == 0:
                self.save_state()

            if self.val_loader and val_every and self.epoch % val_every == 0:
                self.run_val_phase()

            self.run_train_phase()
            self.end_epoch()

        if self.val_loader:
            self.run_val_phase()

        if self.output_dir:
            self.save_state()

        self.end_train()

    def run_train_phase(self):
        self.start_phase(phase='train')

        for i, batch in enumerate(self.train_loader):
            self.start_batch(phase='train', batch=i, loader=self.train_loader)

            self.optimizer.zero_grad()
            outputs = self.forward(batch, run_physics=not self.supervised)

            loss = outputs['loss']
            if not torch.isfinite(loss):
                raise RuntimeError(f'Invalid loss: {loss.item()}')

            loss.backward()
            self.optimizer.step()
            self.step += 1

            self.end_batch(phase='train', batch=i, outputs=outputs)

        self.end_phase(phase='train')

    @torch.no_grad()
    def run_test_phase(self, sync_cuda=False):
        self.start_phase(phase='test')

        for i, batch in enumerate(self.test_loader):
            self.batch_start(phase='test', batch=i, loader=self.test_loader)

            outputs = self.forward(batch, run_physics=True)

            self.batch_end(phase='test', batch=i, outputs=outputs)

        self.end_phase(phase='test')

    @torch.no_grad()
    def run_val_phase(self, sync_cuda=False):
        self.start_phase(phase='val')

        for i, batch in enumerate(self.val_loader):
            self.start_batch(phase='val', batch=i, loader=self.val_loader)

            outputs = self.forward(batch, run_physics=True)

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

    def forward(self, batch, run_physics):

        image = batch['image'].to(self.device)
        mask  = batch['mask'].to(self.device)
        batch_size = image.shape[0]

        # predict elastic modulus from image
        E_pred = self.model.forward(image)

        outputs = {
            'example': batch['example'],
            'image':   batch['image'].cpu(),
            'mask':    batch['mask'].cpu(),
            'E_pred':  E_pred.detach().cpu()
        }
        if 'elast' in batch:
            E_true = batch['elast'].to(self.device)
            outputs['E_true'] = batch['elast'].cpu()

        if run_physics: # compute displacement error via physics simulation
            sim_loss = torch.zeros(batch_size, device=self.device, dtype=image.dtype)
            outputs['pde'] = []

            for k in range(batch_size):
                sim_loss[k], pde_outputs = self.physics_adapter.voxel_simulation_loss(
                    mesh=batch['mesh'][k],
                    unit_m=batch['example'][k].metadata['unit'],
                    affine=batch['affine'][k],
                    E_vox=E_pred[k],
                    bc_spec=None
                )
                outputs['pde'].append(pde_outputs)

        if self.supervised: # minimize error wrt true elasticity field
            outputs['loss'] = normalized_rmse_loss(E_pred, E_true, mask > 0)

        else: # minimize error in simulated displacement field
            outputs['loss'] = sim_loss.mean()

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


def rmse_loss(pred, target, weights):
    err = torch.linalg.norm((pred - target) * weights, axis=1)
    return torch.sqrt(torch.mean(err**2))


def normalized_rmse_loss(pred, target, weights, eps=1e-12):
    assert weights.sum() > 0
    err = torch.linalg.norm((pred - target) * weights, axis=1)
    mag = torch.linalg.norm(target * weights, axis=1)
    num = torch.sqrt(torch.mean(err**2))
    den = torch.sqrt(torch.mean(mag**2)) + eps
    return num / den


# ----- cross-validation -----


def split_on_metadata(examples, key, test_ratio, val_ratio, seed=0):
    from collections import defaultdict

    cats_by_subj = defaultdict(set)
    subjs_by_cat = defaultdict(set)
    for ex in examples:
        subj = ex.subject
        for cat in ex.metadata[key]:
            if cat.startswith('_'):
                continue
            cats_by_subj[subj].add(cat)
            subjs_by_cat[cat].add(subj)

    subjects = list(cats_by_subj.keys())
    n_subjects = len(subjects)
    target_test = int(round(test_ratio * n_subjects))

    categories = list(subjs_by_cat.keys())
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

    print( len(test_subj) / len(subjects) )
    train_subj = set(subjects) - test_subj

    num_val = int(round(val_ratio * n_subjects))
    train_list = list(train_subj)
    rng.shuffle(train_list)
    val_subj = set(train_list[:num_val])
    train_subj = set(train_list[num_val:])

    train_ex = [ex for ex in examples if ex.subject in train_subj]
    test_ex  = [ex for ex in examples if ex.subject in test_subj]
    val_ex   = [ex for ex in examples if ex.subject in val_subj]

    return train_ex, test_ex, val_ex

