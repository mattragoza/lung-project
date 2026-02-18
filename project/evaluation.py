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


def _evaluate(pred, target, weight=None, name=None, profile=None):
    if profile is None:
        profile = name.split('_')[0]
    metrics = mm.evaluate_metrics(pred, target, weight, profile)
    return utils.namespace(metrics, name) if name else metrics


class Callback:

    @property
    def name(self):
        return self.__class__.__name__[:-8].lower() or None

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

    def on_forward_start(self, *args, **kwargs):
        return

    def on_forward_end(self, *args, **kwargs):
        return

    def on_backward_start(self, *args, **kwargs):
        return

    def on_backward_end(self, *args, **kwargs):
        return


class TimerCallback(Callback):

    def __init__(self):
        self.timer = utils.Timer()

    def on_phase_start(self, *args, **kwargs):
        self.timer.tick(sync=False)

    def on_batch_start(self, *args, **kwargs):
        stats = self.timer.tick(sync=False)
        utils.log(f'load_data: {stats}')

    def on_forward_start(self, *args, **kwargs):
        self.timer.tick(sync=False)

    def on_forward_end(self, *args, **kwargs):
        stats = self.timer.tick(sync=True)
        utils.log(f'forward:   {stats}')

    def on_backward_start(self, *args, **kwargs):
        self.timer.tick(sync=False)

    def on_backward_end(self, *args, **kwargs):
        stats = self.timer.tick(sync=True)
        utils.log(f'backward:  {stats}')

    def on_batch_end(self, *args, **kwargs):
        self.timer.tick(sync=False)


class LoggerCallback(Callback):

    def __init__(self, keys):
        self.keys = keys

    def on_train_start(self, *args, **kwargs):
        utils.log('Start training')

    def on_epoch_start(self, epoch, *args, **kwargs):
        utils.log(f'Start epoch {epoch}')

    def on_phase_start(self, epoch, phase, *args, **kwargs):
        utils.log(f'Start epoch {epoch} {phase} phase')

    def on_batch_start(self, epoch, phase, batch, *args, **kwargs):
        utils.log(f'[Epoch {epoch} | {phase.capitalize()} batch {batch}] start')

    def on_batch_end(self, epoch, phase, batch, step, outputs):
        metrics = {k: round(outputs[k].item(), 4) for k in self.keys if k in outputs}
        utils.log(f'[Epoch {epoch} | {phase.capitalize()} batch {batch}] {metrics}')

    def on_phase_end(self, epoch, phase, *args, **kwargs):
        utils.log(f'End epoch {epoch} {phase} phase')

    def on_epoch_end(self, epoch, *args, **kwargs):
        utils.log(f'End epoch {epoch}')

    def on_train_end(self, *args, **kwargs):
        utils.log('Training complete')


class PlotterCallback(Callback):

    def __init__(self, keys, update_interval=1):
        self.update_interval = update_interval

        # history[key][phase][step] = [values]
        self.history = {
            key: {p: defaultdict(list) for p in ['train', 'test', 'val']}
                for key in keys
        }
        self.output_dir = Path('./outputs')
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self._init_plot()

    def on_batch_end(self, epoch, phase, batch, step, outputs):
        phase = str(phase).lower()
        for key in self.history:
            if key == 'mat_pred':
                outputs = ensure_material_map(outputs)
            if key in outputs:
                val = float(outputs[key].float().norm().item())
                self.history[key][phase][step].append(val)
        if batch % self.update_interval == 0:
            self._update_plot()

    def on_phase_end(self, epoch, phase):
        self._save_plot()

    def _init_plot(self, n_cols=4):
        import math
        n_axes = len(self.history.keys())
        n_rows = int(math.ceil(n_axes / n_cols))
        self.fig, self.axes = mpl_viz.subplot_grid(
            n_rows, n_cols, ax_height=2, ax_width=1.5,
            spacing=(1.0, 1.0),
            padding=(1.0, 0.5, 0.5, 0.5) # lrbt
        )
        axes_flat = self.axes.flatten()
        for i, key in enumerate(self.history.keys()):
            ax = axes_flat[i]
            ax.set_xlabel('step')
            ax.set_title(key)

        self.fig.canvas.draw()

    def _update_plot(self):
        for ax in self.axes.flatten():
            ax.clear()

        axes_flat = self.axes.flatten()
        for i, key in enumerate(self.history):
            ax = axes_flat[i]
            for phase in ['train', 'val', 'test']:
                data = self.history[key][phase]
                if not data:
                    continue
                items = sorted(data.items(), key=lambda x: x[0])
                steps = [s for s, _ in items]
                means = [float(np.mean(v)) for _, v in items]
                ax.plot(steps, means, label=phase)
            ax.set_title(key)

        for ax in self.axes.flatten():
            ax.set_xlabel('step')
            ax.set_yscale('log')
            ax.grid(True)
            ax.legend()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _save_plot(self):
        out = self.output_dir / 'training_plot.png'
        self.fig.savefig(out, bbox_inches='tight')


class ViewerCallback(Callback):

    def __init__(
        self,
        keys,
        update_interval=10,
        apply_mask=True,
        shift_rgb=True,
        scale_rgb=1.0,
        n_labels=5,
        **kwargs
    ):
        assert len(keys) > 0
        self.update_interval = update_interval

        self.apply_mask = apply_mask
        self.shift_rgb = shift_rgb
        self.scale_rgb = scale_rgb

        self.output_dir = Path('./outputs')
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self._init_viewers(keys, n_labels)

    def on_batch_end(self, epoch, phase, batch, step, outputs):
        if batch % self.update_interval == 0:
            self._update_viewers(outputs)

    def _init_viewers(self, keys, n_labels):
        from .visual.matplotlib import SliceViewer, get_color_kws
        self.viewers = {}
        for k in keys:
            self.viewers[k] = SliceViewer(title=k, **get_color_kws(k, n_labels))

    def _update_viewers(self, outputs, k=0):
        if self.apply_mask:
            mask = _to_numpy(outputs['mask'][k])
            assert mask.ndim == 4 and mask.shape[0] == 1

        for key, viewer in self.viewers.items():
            if key.startswith('mat_pred'):
                outputs = ensure_material_map(outputs)

            if key not in outputs:
                continue

            array = _to_numpy(outputs[key][k])
            assert array.ndim == 4, array.shape

            if key in {'image', 'img_true', 'img_pred'} and array.shape[0] == 3: # RGB
                if self.scale_rgb:
                    array = array * self.scale_rgb
                if self.shift_rgb: # map [-1, 1] -> [0, 1]
                    array = (array + 1) / 2 
                if self.apply_mask:
                    array = array * mask
                # array shape: (3, I, J, K)
            else:
                if self.apply_mask:
                    array = array * mask
                array = array[0] # (I, J, K)

            viewer.update_array(array)

    def on_phase_end(self, epoch, phase):
        for key, viewer in self.viewers.items():
            out = self.output_dir / f'{key}_viewer.png'
            viewer.fig.savefig(out, bbox_inches='tight')


class EvaluatorCallback(Callback):

    def __init__(self):
        self.example_rows  = defaultdict(list)
        self.material_rows = defaultdict(list)

        self.output_dir = Path('./outputs')
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def on_batch_end(self, epoch, phase, batch, step, outputs):
        self.evaluate(epoch, phase, batch, step, outputs)

    def on_phase_end(self, epoch, phase):
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
        if 'loss_base' in outputs:
            base['loss_base'] = float(outputs['loss_base'].item()),
            base['loss_ratio'] = float(outputs['loss_ratio'].item())

        outputs = ensure_material_map(outputs)

        for k in range(batch_size):
            ex = outputs['example'][k]
            ex_base = {**base, 'subject': ex.subject}

            ex_metrics = self.compute_metrics(outputs, index=k)
            self.example_rows[phase].append(ex_base | ex_metrics)

            for l in _material_labels(outputs, index=k):
                mat_base = {**ex_base, 'material': int(l)}
                mat_metrics = self.compute_metrics(outputs, index=k, label=l)
                self.material_rows[phase].append(mat_base | mat_metrics)

    def compute_metrics(self, outputs, index, label=None):
        ret = {}
        if 'mask' in outputs:
            ret |= self.compute_voxel_metrics(outputs, index, label)
        if 'sim' in outputs:
            ret |= self.compute_mesh_metrics(outputs, index, label)
        return ret

    def compute_voxel_metrics(self, outputs, index, label=None):
        ex = outputs['example'][index]

        mask = _to_numpy(outputs['mask'][index].bool()).reshape(-1, 1)
        mat_true = _to_numpy(outputs['mat_true'][index]).reshape(-1, 1)
        mat_pred = _to_numpy(outputs['mat_pred'][index]).reshape(-1, 1)

        if label is None:
            sel = mask
        else:
            sel = mask & (mat_true == label)

        num_voxels = int(np.count_nonzero(sel))
        if num_voxels == 0:
            utils.warn(f'WARNING: Mask is empty for subject {ex.subject} (material {label})')
            return {'num_voxels': num_voxels}

        ret = {'num_voxels': num_voxels}

        for name in _voxel_param_fields(outputs):
            pred_key = f'{name}_pred'
            true_key = f'{name}_true'
            pred = _to_numpy(outputs[pred_key][index]).reshape(-1, 1)
            if true_key in outputs:
                true = _to_numpy(outputs[true_key][index]).reshape(-1, 1)
                ret |= _evaluate(pred[sel], true[sel], name=f'{name}_vox')
            else:
                ret |= _evaluate(pred[sel], None, name=f'{name}_vox')
    
        if label is not None:
            ret |= _evaluate(mat_pred == label, mat_true == label, name='mat_vox')

            for key in ['mat_pred_a', 'mat_pred_r', 'mat_pred_o']:
                if outputs.get(key) is not None:
                    mat_pred_ = _to_numpy(outputs[key][index]).reshape(-1, 1)
                    ret |= _evaluate(mat_pred_ == label, mat_true == label, name=key)
        return ret

    def compute_mesh_metrics(self, outputs, index, label=None):
        ex = outputs['example'][index]
        sim_output = outputs['sim'][index]
        if sim_output is None:
            return {}

        vol_cells = _to_numpy(sim_output['volume'])
        mat_cells = _to_numpy(sim_output['material'].cells)

        if mat_cells is None:
            sel = np.ones_like(vol_cells, dtype=bool)
        else:
            sel = (mat_cells != 0) if label is None else (mat_cells == label)

        num_cells = int(np.count_nonzero(sel))
        if num_cells == 0:
            return {'num_cells': 0}

        vol_sel = vol_cells[sel]
        vol_sum = float(np.sum(vol_sel))
        if not np.isfinite(vol_sum) or vol_sum <= 0:
            utils.warn(f'WARNING: Invalid cell volume for subject {ex.subject} (material {label}); skipping')
            return {'num_cells': 0, 'volume': vol_sum}

        ret = {'num_cells': num_cells, 'volume': vol_sum}

        if 'u_pred' in sim_output:
            u_pred = _to_numpy(sim_output['u_pred'].cells) # meters
            if 'u_true' in sim_output:
                u_true = _to_numpy(sim_output['u_true'].cells) # meters
                ret |= _evaluate(u_pred[sel], u_true[sel], vol_sel, name='u_cell')
            else:
                ret |= _evaluate(u_pred[sel], None, vol_sel, name='u_cell')

        if 'residual' in sim_output:
            residual = _to_numpy(sim_output['residual'].cells)
            ret |= _evaluate(residual[sel], None, vol_sel, name='res_cell')

        for name in _mesh_param_fields(sim_output):
            pred_key = f'{name}_pred'
            true_key = f'{name}_true'

            pred = _to_numpy(sim_output[pred_key].cells)
            if sim_output.get(true_key) is not None:
                true = _to_numpy(sim_output[true_key].cells)
                ret |= _evaluate(pred[sel], true[sel], vol_sel, name=f'{name}_cell')
            else:
                ret |= _evaluate(pred[sel], None, vol_sel, name=f'{name}_cell')

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



def _material_labels(outputs, index):
    labels = set()
    if 'mat_true' in outputs:
        mat_mask = _to_numpy(outputs['mat_true'][index]).reshape(-1, 1)
        labels |= set(np.unique(mat_mask[mat_mask > 0]))
    if outputs.get('sim') is not None:
        sim_output = outputs['sim'][index]
        if sim_output and sim_output.get('material') is not None:
            mat_cells = _to_numpy(sim_output['material'].cells)
            labels |= set(np.unique(mat_cells[mat_cells > 0]))
    return sorted(labels)


def _voxel_param_fields(outputs):
    names = []
    for k, v in outputs.items():
        if not k.endswith('_pred'):
            continue
        name = k[:-5]
        if name not in {'E', 'nu', 'G', 'K', 'mu', 'lam', 'rho'}:
            continue
        if torch.is_tensor(v) and v.ndim == 5 and v.shape[1] == 1:
            names.append(name)
    return sorted(set(names))


def _mesh_param_fields(sim_output):
    names = []
    for k, v in sim_output.items():
        if not k.endswith('_pred'):
            continue
        name = k[:-5]
        if name not in {'E', 'nu', 'G', 'K', 'mu', 'lam', 'rho'}:
            continue
        if getattr(v, 'cells') is not None:
            names.append(name)
    return sorted(set(names))


def ensure_material_map(outputs):
    if 'mat_pred' in outputs:
        return outputs

    elif 'mat_logits' in outputs:
        mat_logits = outputs['mat_logits']                # (B,C,I,J,K)
        mat_pred = torch.argmax(mat_logits, dim=1)        # (B,I,J,K)
        outputs['mat_pred'] = mat_pred.unsqueeze(1).cpu() # (B,1,I,J,K)
        return outputs

    if 'E_true' not in outputs or 'E_pred' not in outputs:
        utils.warn('Cannot estimate material map from provided outputs.')
        return outputs

    batch_size = len(outputs['example'])
    outputs['mat_pred'] = [None] * batch_size
    outputs['mat_pred_a'] = [None] * batch_size
    outputs['mat_pred_r'] = [None] * batch_size
    outputs['mat_pred_o'] = [None] * batch_size

    for k in range(batch_size):
        mat_true = _to_numpy(outputs['mat_true'][k]) # (B,1,I,J,K)
        E_true   = _to_numpy(outputs['E_true'][k])   # (B,1,I,J,K)
        E_pred   = _to_numpy(outputs['E_pred'][k])   # (B,1,I,J,K)

        mat_pred_a = predict_material_map(E_pred, E_true, mat_true, mode='absolute')
        mat_pred_r = predict_material_map(E_pred, E_true, mat_true, mode='relative')
        mat_pred_o = predict_material_map(E_pred, E_true, mat_true, mode='ordinal')

        outputs['mat_pred_a'][k] = torch.from_numpy(mat_pred_a)
        outputs['mat_pred_r'][k] = torch.from_numpy(mat_pred_r)
        outputs['mat_pred_o'][k] = torch.from_numpy(mat_pred_o)
        outputs['mat_pred'][k] = torch.from_numpy(mat_pred_a)

    return outputs


def predict_material_map(
    E_pred,
    E_true,
    mat_true,
    mode='absolute',
    use_prior=False,
    background=0,
    labels=(1, 2, 3, 4, 5),
    levels=(1e3, 2e3, 3e3, 5e3, 9e3),
    prior=(0.2, 0.2, 0.2, 0.2, 0.2),
    eps=1e-12
):
    E_pred = np.asarray(E_pred) # (I, J, K)
    E_true = np.asarray(E_true)
    mat_true = np.asarray(mat_true)

    assert E_pred.shape == E_true.shape == mat_true.shape

    assert mode in {'absolute', 'relative', 'ordinal'}
    mat_pred = np.full(mat_true.shape, background, dtype=np.int32)

    mask = (mat_true != background)
    assert mask.any()

    labels = np.asarray(labels, dtype=int)
    levels = np.asarray(levels, dtype=np.float32)
    prior = np.asarray(prior, dtype=np.float32)
    assert len(labels) == len(levels) == len(prior)

    p_sum = np.sum(prior)
    assert np.isfinite(p_sum) and abs(p_sum - 1.0) < 1e-5

    logE_pred = np.log10(np.maximum(E_pred, eps))
    logE_true = np.log10(np.maximum(E_true, eps))
    log_levels = np.log10(np.maximum(levels, eps))

    def bin_with_edges(x, edges):
        inds = np.digitize(x, edges, right=False)
        return labels[inds]

    fixed_edges = (log_levels[:-1] + log_levels[1:]) * 0.5
    assert len(fixed_edges) == len(levels) - 1

    if mode == 'absolute':
        mat_pred[mask] = bin_with_edges(logE_pred[mask], fixed_edges)

    elif mode == 'relative':
        pred_mean = logE_pred[mask].mean()
        pred_std  = logE_pred[mask].std()

        if use_prior:
            true_mean = np.sum(prior * log_levels)
            true_std  = np.sqrt(np.sum(prior * (log_levels - true_mean)**2))
        else:
            true_mean = logE_true[mask].mean()
            true_std  = logE_true[mask].std()

        standardized = (fixed_edges - true_mean) / np.maximum(true_std, eps)
        adjusted_edges = pred_mean + np.maximum(pred_std, eps) * standardized
        mat_pred[mask] = bin_with_edges(logE_pred[mask], adjusted_edges)

    elif mode == 'ordinal':
        if use_prior:
            probabilities = np.cumsum(prior)[:-1]
        else:
            counts = np.bincount(mat_true[mask], minlength=labels.max() + 1)[1:]
            posterior = counts / counts.sum()
            probabilities = np.cumsum(posterior)[:-1]

        probabilities = np.clip(probabilities, eps, 1.0 - eps)
        quantile_edges = np.quantile(logE_pred[mask], q=probabilities)
        mat_pred[mask] = bin_with_edges(logE_pred[mask], quantile_edges)

    return np.where(mask, mat_pred, background)

