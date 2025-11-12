import sys, os, time
from pathlib import Path
import numpy as np
import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

from .core import utils, transforms, interpolation
from . import solvers


def _to_numpy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        train_loader,
        test_loader=None,
        val_loader=None,
        evaluator=None,
        supervised=False,
        solver_kws=None,
        nu_value=0.4,
        rho_bias=1000.,
        rho_known=False,
        scalar_degree=1,
        vector_degree=1,
        output_dir='checkpoints',
        device='cuda'
    ):
        self.model = model.to(device)
        self.optimizer = optimizer

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.evaluator = evaluator

        self.supervised = supervised
        self.solver_kws = solver_kws or {}
        self.device = device

        # physics adapter settings
        self.nu_value = nu_value
        self.rho_bias = rho_bias
        self.rho_known = rho_known
        self.scalar_degree = scalar_degree
        self.vector_degree = vector_degree

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
            self.run_train_phase(run_eval=eval_on_train)
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
            outputs = self.forward(batch, phase='train', run_solver=not self.supervised)

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
            ex_df = self.evaluator.phase_end(self.epoch, phase='train')
            utils.log(f'Train metrics @ epoch {self.epoch}: \n{ex_df}')

    @torch.no_grad()
    def run_test_phase(self):
        self.model.eval()
        for i, batch in enumerate(self.test_loader):
            utils.log(f'[Epoch {self.epoch} | Test batch {i+1}/{len(self.test_loader)}]', end=' ')
            t0 = time.time()

            outputs = self.forward(batch, phase='test', run_solver=True)
            if self.evaluator:
                self.evaluator.evaluate(outputs, self.epoch, phase='test', batch=i)

            t1 = time.time()
            loss = outputs['loss']
            utils.log(f'loss = {loss.item():.4e} | time = {t1 - t0:.4f}')

        if self.evaluator:
            ex_df = self.evaluator.phase_end(self.epoch, phase='test')
            utils.log(f'Test metrics @ epoch {self.epoch}: \n{ex_df}')

    @torch.no_grad()
    def run_val_phase(self):
        self.model.eval()
        for i, batch in enumerate(self.val_loader):
            utils.log(f'[Epoch {self.epoch} | Val batch {i+1}/{len(self.val_loader)}]', end=' ')
            t0 = time.time()

            outputs = self.forward(batch, phase='val', run_solver=True)
            if self.evaluator:
                self.evaluator.evaluate(outputs, self.epoch, phase='val', batch=i)

            t1 = time.time()
            loss = outputs['loss']
            utils.log(f'loss = {loss.item():.4e} | time = {t1 - t0:.4f}')

        if self.evaluator:
            ex_df = self.evaluator.phase_end(self.epoch, phase='val')
            utils.log(f'Val metrics @ epoch {self.epoch}: \n{ex_df}')

    def forward(self, batch, phase, run_solver):

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

        if self.supervised: # train to directly minimize E error
            outputs['loss'] = normalized_rmse_loss(E_pred, E_true, mask > 0)

        if run_solver: # compute simulated displacement error via PDE solver
            pde_loss = torch.zeros(batch_size, device=self.device, dtype=image.dtype)
            pde_outputs = []

            for k in range(batch_size):
                pde_context = self.get_pde_context(batch['example'][k], batch['mesh'][k])

                pts_world = pde_context['pts_world'].to(self.device)
                pts_voxel = transforms.world_to_voxel_coords(pts_world, batch['affine'][k])

                E_interp = interpolation.interpolate_image(E_pred[k], pts_voxel)[...,0]
                mu_values, lam_values = transforms.compute_lame_parameters(E_interp, self.nu_value)

                pde_module = pde_context['module']
                pde_loss[k], res_values, u_sim_values = pde_module.forward(mu_values, lam_values)

                pde_outputs.append({
                    'scalar_degree': int(self.scalar_degree),
                    'vector_degree': int(self.vector_degree),
                    'verts':         pde_context['verts'].cpu(),
                    'cells':         pde_context['cells'].cpu(),
                    'vol_cells':     pde_context['vol_cells'].cpu(),
                    'mat_cells':     pde_context['mat_cells'].cpu(),
                    'E_true_cells':  pde_context['E_cells'].cpu(),
                    'u_true_values': pde_context['u_obs_values'].cpu(),
                    'E_pred_values': E_interp.detach().cpu(),
                    'u_pred_values': u_sim_values.detach().cpu(),
                    'res_values':    res_values.detach().cpu(),
                })

            outputs['pde']  = pde_outputs

        if not self.supervised: # train to minimize simulation error
            outputs['loss'] = pde_loss.mean()

        return outputs

    def get_pde_context(self, ex, mesh):
        key = ex.subject
        if key not in self.cache:
            self.cache[key] = self.init_pde_context(ex, mesh)
        return self.cache[key]

    def init_pde_context(self, ex, mesh):

        unit_m = ex.metadata['unit'] # meters per world unit
        verts  = mesh.points # world coords
        cells  = mesh.cells_dict['tetra']

        vol_cells = transforms.compute_cell_volume(verts, cells)

        # materials are originally defined on mesh cells
        mat_cells = mesh.cell_data_dict['material']['tetra']
        rho_cells = mesh.cell_data_dict['rho']['tetra'] # kg/m^3
        E_cells   = mesh.cell_data_dict['E']['tetra']   # Pa

        if self.scalar_degree == 0: # cell data
            pts_world  = verts[cells].mean(axis=1)
            rho_values = rho_cells
        elif self.scalar_degree == 1: # node data
            pts_world  = verts
            rho_values = mesh.point_data['rho']

        if not self.rho_known: # estimate from image
            img_cells  = mesh.cell_data_dict['image']['tetra']
            img_nodes  = mesh.point_data['image']
            img_values = transforms.smooth_mesh_values(
                verts, cells, img_nodes, img_cells, degree=self.scalar_degree
            )
            rho_values = img_values * self.rho_bias

        if self.vector_degree == 0: # cell data
            u_obs_values = mesh.cell_data_dict['u'] # u_obs has world units
        elif self.vector_degree == 1: # node data
            u_obs_values = mesh.point_data['u']

        # convert to CPU tensors; solver will handle device management
        def _as_cpu_tensor(a, dtype=None):
            return torch.as_tensor(a, dtype=dtype or torch.float, device='cpu')

        verts_t = _as_cpu_tensor(verts * unit_m) # to meters
        cells_t = _as_cpu_tensor(cells, torch.int)

        vol_cells_t = _as_cpu_tensor(vol_cells * unit_m**3) # to meters^3
        mat_cells_t = _as_cpu_tensor(mat_cells, torch.int)
        rho_cells_t = _as_cpu_tensor(rho_cells)
        E_cells_t   = _as_cpu_tensor(E_cells)

        pts_world_t    = _as_cpu_tensor(pts_world)
        rho_values_t   = _as_cpu_tensor(rho_values)
        u_obs_values_t = _as_cpu_tensor(u_obs_values * unit_m) # to meters

        solver = solvers.warp.WarpFEMSolver(
            scalar_degree=self.scalar_degree,
            vector_degree=self.vector_degree,
            device=self.device,
            **(self.solver_kws or {})
        )
        module = solvers.base.PDESolverModule(
            solver, verts_t, cells_t, rho_values_t, u_obs_values_t
        )
        context = {
            'module': module,
            'verts':  verts_t, # meters
            'cells':  cells_t,
            'vol_cells': vol_cells_t,
            'mat_cells': mat_cells_t,
            'rho_cells': rho_cells_t,
            'E_cells':   E_cells_t,
            'pts_world':    pts_world_t,
            'rho_values':   rho_values_t,
            'u_obs_values': u_obs_values_t, # meters
        }
        return context

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
    err = torch.linalg.norm((pred - target) * weights, axis=1)
    mag = torch.linalg.norm(target * weights, axis=1)
    num = torch.sqrt(torch.mean(err**2))
    den = torch.sqrt(torch.mean(mag**2)) + eps
    return num / den


# ---- cross validation splits ----


def split_by_category(examples, test_ratio, val_ratio, seed=0):
    from collections import defaultdict

    cats_by_subj = defaultdict(set)
    subjs_by_cat = defaultdict(set)
    for ex in examples:
        subj = ex.subject
        for cat in ex.metadata['category']:
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

