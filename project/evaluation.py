from typing import List, Dict, Any
from collections import defaultdict
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from .core import utils, transforms
from .core import metrics as mm


def _to_numpy(t):
    return t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)


def _eval(pred, target, weight=None, name=None, profile=None):
    if profile is None:
        profile = name.split('_')[0]
    metrics = mm.evaluate_metrics(pred, target, weight, profile)
    return utils.namespace(metrics, name) if name else metrics


class Callback:

    def on_train_start(self, **kwargs):
        return

    def on_train_end(self, **kwargs):
        return

    def on_epoch_start(self, epoch, **kwargs):
        return

    def on_epoch_end(self, epoch, **kwargs):
        return

    def on_phase_start(self, epoch, phase, **kwargs):
        return

    def on_phase_end(self, epoch, phase, **kwargs):
        return

    def on_batch_start(self, epoch, phase, batch, step, **kwargs):
        return

    def on_batch_end(self, epoch, phase, batch, step, **kwargs):
        return


class Logger(Callback):

    def __init__(self, sync_cuda: bool=False):
        self.sync_cuda = sync_cuda

    def on_train_start(self):
        utils.log(f'Start training loop')

    def on_epoch_start(self, epoch):
        utils.log(f'===== Epoch {epoch} =====')

    def on_batch_start(self, epoch, phase, batch, step, loader):
        self.t_start = time.time()
        torch.cuda.reset_peak_memory_stats()
        utils.log(f'[Epoch {epoch} | {phase.capitalize()} batch {batch}/{len(loader)}]', end=' ')

    def on_batch_end(self, epoch, phase, batch, step, outputs, unit_b=2**20):
        loss = outputs['loss'].item()
        if self.sync_cuda:
            torch.cuda.synchronize()
        curr = torch.cuda.memory_allocated()     / unit_b
        peak = torch.cuda.max_memory_allocated() / unit_b
        res  = torch.cuda.memory_reserved()      / unit_b
        memory = f'{curr:.0f} / {peak:.0f} / {res:.0f}'
        t_delta = time.time() - self.t_start
        utils.log(f'loss = {loss:.4e} | time = {t_delta:.4f}s | memory = {memory} MiB')


class Plotter(Callback):

    def __init__(self):
        self.records = defaultdict(list) # (phase, step) -> List[loss]
        self.output_dir = Path('./outputs')
        self.output_dir.mkdir(exist_ok=True, parents=True)

        plt.ion()
        self.fig, self.ax = plt.subplots()

    def on_batch_end(self, epoch, phase, batch, step, outputs):
        loss_val = float(outputs['loss'].item())
        self.records[(phase, step)].append(loss_val)
        self.update_plot()

    def on_phase_end(self, epoch, phase):
        self.save_plot()

    def update_plot(self):
        self.ax.clear()
        for phase in ['train', 'val', 'test']:
            phase_items = [
                (step, vals) for (p, step), vals in self.records.items()
                if p == phase
            ]
            if not phase_items:
                continue
            steps = []
            means = []
            for step, vals in sorted(phase_items, key=lambda x: x[0]):
                steps.append(step)
                means.append(float(np.mean(vals)))
            self.ax.plot(steps, means, marker='o', linestyle='-', label=phase)
        self.ax.set_xlabel('Optimizer step')
        self.ax.set_ylabel('Loss')
        self.ax.grid(True)
        self.ax.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def save_plot(self):
        out = self.output_dir / 'training_plot.png'
        self.fig.savefig(out, bbox_inches='tight')


class Evaluator(Callback):

    def __init__(self, eval_on_train: bool=False):
        self.eval_on_train = eval_on_train
        self.example_rows  = defaultdict(list)
        self.material_rows = defaultdict(list)

        self.output_dir = Path('./outputs')
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def on_batch_end(self, epoch, phase, batch, step, outputs):
        if self.eval_on_train or phase.lower() != 'train':
            self.evaluate(epoch, phase, batch, step, outputs)

    @torch.no_grad()
    def evaluate(self, epoch, phase, batch, step, outputs):
        batch_size = len(outputs['example'])
        base = {
            'epoch': int(epoch),
            'phase': str(phase),
            'batch': int(batch),
            'step': int(step),
            'loss': float(outputs['loss'].item())
        }
        for k in range(batch_size):
            ex = outputs['example'][k]
            ex_base = {**base, 'subject': ex.subject}
            ex_metrics = self.compute_metrics(outputs, index=k)
            self.example_rows[phase].append(ex_base | ex_metrics)

            for l in self.get_material_labels(outputs, index=k):
                mat_base = {**ex_base, 'material': int(l)}
                mat_metrics = self.compute_metrics(outputs, index=k, label=l)
                self.material_rows[phase].append(mat_base | mat_metrics)

    def get_material_labels(self, outputs, index):
        labels = set()
        if 'mask' in outputs:
            mat_mask = _to_numpy(outputs['mask'][index]).reshape(-1, 1)
            labels |= set(np.unique(mat_mask[mat_mask > 0]))
        if 'pde' in outputs:
            pde_output = outputs['pde'][index]
            mat_cells = _to_numpy(pde_output['material'].cells)
            labels |= set(np.unique(mat_cells[mat_cells > 0]))
        return sorted(labels)

    def compute_metrics(self, outputs, index, label=None):
        ret = {}
        if 'mask' in outputs:
            ret |= self.compute_voxel_metrics(outputs, index, label)
        if 'pde' in outputs:
            ret |= self.compute_mesh_metrics(outputs, index, label)
        return ret

    def compute_voxel_metrics(self, outputs, index, label=None):
        ex = outputs['example'][index]

        mat_mask = _to_numpy(outputs['mask'][index]).reshape(-1, 1)
        E_true = _to_numpy(outputs['E_true'][index]).reshape(-1, 1) # Pa
        E_pred = _to_numpy(outputs['E_pred'][index]).reshape(-1, 1) # Pa

        if label is None:
            sel = (mat_mask != 0)
        else:
            sel = (mat_mask == label)

        num_voxels = int(np.count_nonzero(sel))
        if num_voxels == 0:
            utils.warn(f'WARNING: Mask is empty for subject {ex.subject} (material {label})')
            return {'num_voxels': num_voxels}

        ret = {'num_voxels': num_voxels}
        ret |= _eval(E_pred[sel], E_true[sel], name='E_vox')

        return ret

    def compute_mesh_metrics(self, outputs, index, label=None):
        ex = outputs['example'][index]
        pde_output = outputs['pde'][index]

        vol_cells = _to_numpy(pde_output['volume'])
        mat_cells = _to_numpy(pde_output['material'].cells)

        rho_true = _to_numpy(pde_output['rho_true'].cells)
        rho_pred = _to_numpy(pde_output['rho_pred'].cells)

        E_true = _to_numpy(pde_output['E_true'].cells) # Pa
        E_pred = _to_numpy(pde_output['E_pred'].cells) # Pa

        u_true = _to_numpy(pde_output['u_true'].cells) # meters
        u_pred = _to_numpy(pde_output['u_pred'].cells) # meters

        residual = _to_numpy(pde_output['residual'].cells)

        if label is None:
            sel = (mat_cells != 0)
        else:
            sel = (mat_cells == label)

        num_cells = int(np.count_nonzero(sel))
        if num_cells == 0:
            return {'num_cells': 0}

        vol_sel = vol_cells[sel]
        vol_sum = float(np.sum(vol_sel))
        if vol_sum <= 0:
            utils.warn(f'WARNING: Invalid cell volume for subject {ex.subject} (material {label}); skipping')
            return {'num_cells': 0, 'volume': vol_sum}

        ret = {'num_cells': num_cells, 'volume': vol_sum}
        ret |= _eval(rho_pred[sel], rho_true[sel], vol_sel, name='rho_cell')
        ret |= _eval(E_pred[sel], E_true[sel], vol_sel, name='E_cell')
        ret |= _eval(u_pred[sel], u_true[sel], vol_sel, name='u_cell')
        ret |= _eval(residual[sel], None, vol_sel, name='res_cell')

        return ret

    def on_phase_end(self, epoch, phase):
        ex_path  = self.output_dir / 'example_metrics.csv'
        mat_path = self.output_dir / 'material_metrics.csv'

        ex_df_all = pd.concat(
            [pd.DataFrame(rows) for rows in self.example_rows.values()],
            ignore_index=True
        )
        mat_df_all = pd.concat(
            [pd.DataFrame(rows) for rows in self.material_rows.values()],
            ignore_index=True
        )
        for df, path in [(ex_df_all, ex_path), (mat_df_all, mat_path)]:
            if df.empty:
                continue
            tmp = path.with_suffix(path.suffix + '.tmp')
            try:
                utils.log(f'Saving {path}')
                df.to_csv(tmp, index=False)
                tmp.replace(path)
            finally:
                tmp.unlink(missing_ok=True)

        # current phase metrics
        m = pd.DataFrame(self.example_rows[phase])
        utils.log(f'{phase.capitalize()} metrics @ epoch {epoch}: \n{m}')

