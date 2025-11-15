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
        evaluator=None,
        supervised=False,
        output_dir='checkpoints',
        device='cuda'
    ):
        self.model = model.to(device)
        self.optimizer = optimizer

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.evaluator = evaluator

        self.physics_adapter = physics_adapter
        self.supservised = supervised
        self.device = device

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.epoch = 0
        self.cache = {}

    def train(self, num_epochs, val_every=1, save_every=10, eval_on_train=False):

        for _ in range(num_epochs):
            if self.val_loader and val_every and self.epoch % val_every == 0:
                self.run_val_phase()

            if self.output_dir and save_every and self.epoch % save_every == 0:
                self.save_state()

            self.run_train_phase(eval_on_train)
            self.epoch += 1

        if self.val_loader:
            self.run_val_phase()

        if self.output_dir:
            self.save_state()

    def run_train_phase(self, run_eval=False):
        self.model.train()

        for i, batch in enumerate(self.train_loader):
            utils.log(f'[Epoch {self.epoch} | Train batch {i+1}/{len(self.train_loader)}]', end=' ')
            t0 = time.time()

            self.optimizer.zero_grad()
            outputs = self.forward(batch, run_physics=not self.supervised)

            loss = outputs['loss']
            if not torch.isfinite(loss):
                raise RuntimeError(f'Invalid loss: {loss.item()}')

            loss.backward()
            self.optimizer.step()

            if run_eval:
                self.evaluator.evaluate(outputs, self.epoch, phase='train', batch=i)

            t1 = time.time()
            utils.log(f'loss = {loss.item():.4e} | time = {t1 - t0:.4f}')

        if run_eval:
            metrics = self.evaluator.phase_end(self.epoch, phase='train')
            utils.log(f'Train metrics @ epoch {self.epoch}: \n{metrics}')

    @torch.no_grad()
    def run_test_phase(self):
        self.model.eval()

        for i, batch in enumerate(self.test_loader):
            utils.log(f'[Epoch {self.epoch} | Test batch {i+1}/{len(self.test_loader)}]', end=' ')
            t0 = time.time()

            outputs = self.forward(batch, run_physics=True)
            if self.evaluator:
                self.evaluator.evaluate(outputs, self.epoch, phase='test', batch=i)

            t1 = time.time()
            utils.log(f'loss = {outputs["loss"].item():.4e} | time = {t1 - t0:.4f}')

        if self.evaluator:
            metrics = self.evaluator.phase_end(self.epoch, phase='test')
            utils.log(f'Test metrics @ epoch {self.epoch}: \n{metrics}')

    @torch.no_grad()
    def run_val_phase(self):
        self.model.eval()

        for i, batch in enumerate(self.val_loader):
            utils.log(f'[Epoch {self.epoch} | Val batch {i+1}/{len(self.val_loader)}]', end=' ')
            t0 = time.time()

            outputs = self.forward(batch, run_physics=True)
            if self.evaluator:
                self.evaluator.evaluate(outputs, self.epoch, phase='val', batch=i)

            t1 = time.time()
            utils.log(f'loss = {outputs["loss"].item():.4e} | time = {t1 - t0:.4f}')

        if self.evaluator:
            metrics = self.evaluator.phase_end(self.epoch, phase='val')
            utils.log(f'Val metrics @ epoch {self.epoch}: \n{metrics}')

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
                sim_loss[k], pde_outputs = self.physics_adapter.simulate(
                    mesh=batch['mesh'][k],
                    unit_m=batch['example'][k].metadata['unit'],
                    affine=batch['affine'][k],
                    E_pred=E_pred[k]
                )
                outputs['pde'].append(pde_outputs)

        if self.supervised: # minimize error wrt true elasticity field
            outputs['loss'] = normalized_rmse_loss(E_pred, E_true, mask > 0)

        else: # minimize error in simulated displacement field
            outputs['loss'] = sim_loss.mean()

        return outputs

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

