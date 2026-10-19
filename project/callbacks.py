# callbacks.py

from typing import List, Dict, Optional
from pathlib import Path

from .common import utils, fileio


class Callback:

    @property
    def name(self):
        return type(self).__name__.removesuffix('Callback').lower() or None

    def on_train_start(self):
        return

    def on_train_end(self):
        return

    def on_epoch_start(self, epoch: int):
        return

    def on_epoch_end(self, epoch: int):
        return

    def on_phase_start(self, epoch: int, phase: str):
        return

    def on_phase_end(self, epoch: int, phase: str):
        return

    def on_batch_start(self, epoch: int, phase: str, batch_idx: int, batch: dict):
        return

    def on_batch_end(self, epoch: int, phase: str, batch_idx: int, outputs: dict):
        return


class LoggingCallback(Callback):

    def __init__(self):
        self.timer = utils.Timer()

    def on_train_start(self):
        utils.log('Training start')

    def on_train_end(self):
        utils.log('Training complete')

    def on_phase_start(self, epoch: int, phase: str):
        self.timer.tick()

    def on_batch_start(self, epoch: int, phase: str, batch_idx: int, batch: dict):

        stats = self.timer.tick()
        t_delta = stats['t_delta']
        mem_used = stats['mem_used']

        sid = batch['example'][0].subject
        img_size = tuple(batch['image'][0,0].shape)
        n_verts = len(batch['mesh'][0].points)

        utils.log(
            f'[Epoch {epoch} | {phase} batch {batch_idx} | load data]  '
            f't_delta = {t_delta:.2g}s  mem_used = {mem_used:.2g} GiB  '
            f'sid = {sid}  img_size = {img_size}  n_verts = {n_verts}'
        )

    def on_batch_end(self, epoch: int, phase: str, batch_idx: int, outputs: dict):

        stats = self.timer.tick()
        t_delta = stats['t_delta']
        mem_used = stats['mem_used']

        utils.log(
            f'[Epoch {epoch} | {phase} batch {batch_idx} | run batch]  '
            f't_delta = {t_delta:.2g}s  mem_used = {mem_used:.2g} GiB  '
            f'loss = {outputs["loss"].item():.4g}'
        )


## DEPRECATING


class PlotterCallback(Callback):

    def __init__(self, keys, output_dir='plotter', update_interval=1):
        self.update_interval = update_interval

        # history[key][phase][step] = [values]
        self.history = {
            key: {p: defaultdict(list) for p in ['train', 'test', 'val']}
                for key in keys
        }
        self.output_dir = Path(outputs)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self._init_plot()

    def on_batch_end(self, epoch, phase, batch_idx, outputs):
        phase = str(phase).lower()

        for key in self.history:

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
        output_dir='views',
        **kwargs
    ):
        assert len(keys) > 0
        self.update_interval = update_interval

        self.apply_mask = apply_mask
        self.shift_rgb = shift_rgb
        self.scale_rgb = scale_rgb

        self.output_dir = Path(output_dir)
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
                outputs = ensure_material_preds(outputs)

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

