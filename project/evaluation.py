from typing import List, Dict, Any
from collections import defaultdict
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from .core import utils, transforms
from .core import metrics as mm
from .visual import matplotlib as mpl_viz


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


class LoggerCallback(Callback):

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


class PlotterCallback(Callback):

    def __init__(self):
        self.loss_history = defaultdict(list) # (phase, step) -> List[loss]
        self.norm_history = defaultdict(list) # (phase, step) -> List[norm]

        self.output_dir = Path('./outputs')
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self._init_plot()

    def on_batch_end(self, epoch, phase, batch, step, outputs):
        phase = str(phase).lower()

        loss_val = float(outputs['loss'].item())
        norm_val = float(outputs['E_pred'].norm().item())

        self.loss_history[(phase, step)].append(loss_val)
        self.norm_history[(phase, step)].append(norm_val)

        self._update_plot()

    def on_phase_end(self, epoch, phase):
        self._save_plot()

    def _init_plot(self):
        plt.ion()
        self.fig, self.axes = mpl_viz.subplot_grid(
            1, 2, ax_height=3, ax_width=3,
            spacing=(0.5, 1.5),
            padding=(0.75, 0.75, 0.5, 0.25)
        )
        self.loss_ax = self.axes[0,0]
        self.norm_ax = self.axes[0,1]

    def _phase_data(self, data, phase):
        items = [
            (step, vals) for (p, step), vals in data.items()
            if p == phase
        ]
        if not items:
            return None, None
        
        items.sort(key=lambda x: x[0])

        steps = [step for step, _ in items]
        means = [float(np.mean(vals)) for _, vals in items]

        return steps, means

    def _update_plot(self):
        for ax in self.axes.flatten():
            ax.clear()

        for phase in ['train', 'val', 'test']:
            steps, means = self._phase_data(self.loss_history, phase)
            if steps is not None:
                self.loss_ax.plot(steps, means, label=phase)

            steps, means = self._phase_data(self.norm_history, phase)
            if steps is not None:
                self.norm_ax.plot(steps, means, label=phase)

        self.loss_ax.set_ylabel('loss')
        self.norm_ax.set_ylabel('norm')

        for ax in self.axes.flatten():
            ax.set_xlabel('step')
            ax.set_yscale('log')
            ax.grid(True)
            ax.legend()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.fig.tight_layout()

    def _save_plot(self):
        out = self.output_dir / 'training_plot.png'
        self.fig.savefig(out, bbox_inches='tight')


class ViewerCallback(Callback):

    def __init__(self, n_labels=5, **kwargs):
        from .visual.matplotlib import SliceViewer
        self.output_dir = Path('./outputs')
        self.output_dir.mkdir(exist_ok=True, parents=True)

        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        colors = plt.get_cmap('tab10').colors
        cmap = ListedColormap(colors[:n_labels])
        cmap.set_under('white')
        cmap.set_over('black')
        cmap.set_bad('black')

        self.viewers = {
            'image': SliceViewer(cmap='gray'),
            'E_pred': SliceViewer(cmap='jet', clim=(0, 1e4)),
            'E_true': SliceViewer(cmap='jet', clim=(0, 1e4)),
            'mask': SliceViewer(cmap=cmap, clim=(1, n_labels)),
            'mat_pred': SliceViewer(cmap=cmap, clim=(1, n_labels)),
        }

    def on_batch_end(self, epoch, phase, batch, step, outputs):
        for key, viewer in self.viewers.items():
            if key == 'mat_pred':
                E_pred = _to_numpy(outputs['E_pred'][0,0])
                E_true = _to_numpy(outputs['E_true'][0,0])
                mat_mask = _to_numpy(outputs['mask'][0,0])
                array = predict_material_map(E_pred, E_true, mat_mask)
            else:
                array = _to_numpy(outputs[key][0,0])
            viewer.update_array(array)

    def on_phase_end(self, epoch, phase):
        for key, viewer in self.viewers.items():
            out = self.output_dir / f'{key}_viewer.png'
            viewer.fig.savefig(out, bbox_inches='tight')


class EvaluatorCallback(Callback):

    def __init__(self, eval_on_train: bool=False):
        self.eval_on_train = eval_on_train

        self.example_rows  = defaultdict(list)
        self.material_rows = defaultdict(list)

        self.output_dir = Path('./outputs')
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def on_batch_end(self, epoch, phase, batch, step, outputs):
        if self.eval_on_train or phase.lower() != 'train':
            self.evaluate(epoch, phase, batch, step, outputs)

    def on_phase_end(self, epoch, phase):
        if self.eval_on_train or phase.lower() != 'train':
            self.summarize(epoch, phase)

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

            # pre-compute predicted material maps
            if 'mask' in outputs:
                mat_mask = _to_numpy(outputs['mask'][k])
                E_true   = _to_numpy(outputs['E_true'][k])
                E_pred   = _to_numpy(outputs['E_pred'][k])
                mat_pred = predict_material_map(E_pred, E_true, mat_mask)
                if 'mat_pred' not in outputs:
                    outputs['mat_pred'] = [None] * batch_size
                outputs['mat_pred'][k] = mat_pred

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

        if label is not None:
            mat_pred = _to_numpy(outputs['mat_pred'][index]).reshape(-1, 1)
            ret |= _eval(mat_pred == label, mat_mask == label, name='mat_vox')

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

    def summarize(self, epoch, phase):
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


def predict_material_map(E_pred, E_true, mat_mask, background=0, eps=1e-8):
    assert E_pred.shape == E_true.shape == mat_mask.shape

    mat_labels = np.unique(mat_mask)
    mat_labels = mat_labels[mat_labels != background]
    if mat_labels.size == 0:
        raise RuntimeError('no foreground in material mask')

    E_levels = np.zeros_like(mat_labels, dtype=np.float64)
    for i, label in enumerate(mat_labels):
        vals = E_true[mat_mask == label]
        if vals.size > 0:
            E_levels[i] = np.median(vals)

    log_dist = np.abs(
        np.log10(np.maximum(E_pred.reshape(-1, 1), eps)) - 
        np.log10(np.maximum(E_levels.reshape(1, -1), eps))
    )
    nearest_inds = np.argmin(log_dist, axis=1)
    mat_pred = mat_labels[nearest_inds].reshape(mat_mask.shape).astype(mat_mask.dtype)

    # preserve background label
    mat_pred = np.where(mat_mask == background, background, mat_pred)

    return mat_pred

