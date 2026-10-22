# callbacks.py

from typing import List, Dict, Optional
from pathlib import Path
import numpy as np

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

    def on_batch_start(self, epoch: int, phase: str, batch_num: int, batch: dict):
        return

    def on_batch_end(self, epoch: int, phase: str, batch_num: int, outputs: dict):
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

    def on_batch_start(self, epoch: int, phase: str, batch_num: int, batch: dict):
        stats = self.timer.tick()

        utils.log(
            f'[Epoch {epoch} | {phase} batch {batch_num} | load data]  '
            f'dt = {stats["t_delta"]:.2g}s  '
            f'curr_alloc = {stats["curr_alloc"]:.2g} GiB  '
            f'curr_resvd = {stats["curr_resvd"]:.2g} GiB  '
            #f'peak_alloc = {stats["peak_alloc"]:.2g} GiB  '
            #f'peak_resvd = {stats["peak_resvd"]:.2g} GiB  '
            f'mem_usage = {stats["mem_used"]:.2g}/{stats["mem_total"]:.2g} GiB  '
            f'subject = {batch["example"][0].subject}  '
            f'img_size = {tuple(batch["image"][0,0].shape)}  '
            f'n_verts = {len(batch["mesh"][0].points)}'
        )

    def on_batch_end(self, epoch: int, phase: str, batch_num: int, outputs: dict):
        stats = self.timer.tick()

        utils.log(
            f'[Epoch {epoch} | {phase} batch {batch_num} | run batch]  '
            f'dt = {stats["t_delta"]:.2g}s  '
            f'curr_alloc = {stats["curr_alloc"]:.2g} GiB  '
            f'curr_resvd = {stats["curr_resvd"]:.2g} GiB  '
            #f'peak_alloc = {stats["peak_alloc"]:.2g} GiB  '
            #f'peak_resvd = {stats["peak_resvd"]:.2g} GiB  '
            f'mem_usage = {stats["mem_used"]:.2g}/{stats["mem_total"]:.2g} GiB  '
            f'loss = {float(outputs["loss"]):.4g}  '
            f'grad_norm = {float(outputs.get("grad_norm", 0.)):.4g}'
        )


class PlotterCallback(Callback):

    def __init__(
        self,
        output_dir,
        metrics=('loss',), 
        interactive=True,
        exclude_test=True
    ):
        from collections import defaultdict

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = list(metrics)
        self.interactive = interactive
        self.exclude_test = exclude_test

        self.x_history = defaultdict(list)
        self.y_history = defaultdict(list)

        self.lines = {}

        self.create_plots()

    def create_plots(self):
        from .visual.matplotlib import subplot_grid

        self.fig, axes = subplot_grid(
            n_rows=1,
            n_cols=len(self.metrics),
            ax_height=3,
            ax_width=2,
            spacing=(1.0, 1.0),
            padding=(1.0, 0.5, 0.5, 0.5) # lrbt
        )

        self.axes = dict(zip(self.metrics, axes.flat))

        expected_phases = ['train', 'val']
        if not self.exclude_test:
            expected_phases.append('test')

        for metric, ax in self.axes.items():
            for phase in expected_phases:
                self.lines[phase, metric] = ax.plot([], [], label=phase)[0]
            ax.set(xlabel='step', ylabel=metric)
            ax.grid(alpha=0.25)
            ax.legend()

        if self.interactive:
            import matplotlib.pyplot as plt
            plt.show(block=False)

    def on_batch_end(self, epoch, phase, batch_num, outputs):
        if phase == 'test' and self.exclude_test:
            return
        self.append_metrics(phase, outputs)
        self.update_plots()

    def append_metrics(self, phase, outputs):

        def _scalar(x):
            if hasattr(x, 'detach'):
                x = x.detach().item()
            return float(x)

        self.x_history[phase].append(outputs['step'])

        for metric in self.metrics:
            if metric in outputs:
                value = _scalar(outputs[metric])
            elif metric in outputs.get('metrics', {}):
                value = np.mean([
                    _scalar(x[metric]) for x in outputs['metrics']
                ])
            else:
                continue

            self.y_history[phase, metric].append(value)

    def update_plots(self):

        for phase, metric in self.y_history:
            x_vals = self.x_history[phase]
            y_vals = self.y_history[phase, metric]

            x_unique, groups = np.unique(x_vals, return_inverse=True)
            y_mean = np.bincount(groups, weights=y_vals) / np.bincount(groups)

            self.lines[phase, metric].set_data(x_unique, y_mean)
            ax = self.axes[metric]
            ax.relim()
            ax.autoscale_view()

        if self.interactive:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def on_phase_end(self, epoch: int, phase: str):
        self.save_plots()

    def save_plots(self):
        path = self.output_dir / 'training_plot.png'
        self.fig.savefig(path, bbox_inches='tight')


class ViewerCallback(Callback):

    def __init__(self, output_dir, keys):

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.keys = list(keys)

        self.create_viewers()

    def on_batch_end(self, epoch, phase, batch_num, outputs):
        self.update_viewers(outputs)

    def create_viewers(self):
        import matplotlib.pyplot as plt
        from IPython.display import display
        from .visual.matplotlib import SliceViewer, get_color_kws

        self.viewers = {}
        with plt.ioff():
            for key in self.keys:
                viewer = SliceViewer(title=key, **get_color_kws(key))
                self.viewers[key] = viewer
                display(viewer.fig.canvas)        

    def update_viewers(self, outputs):

        def _numpy(t):
            return t.detach().cpu().numpy()

        mask = _numpy(outputs['batch']['mask'][0,0])

        for key, viewer in self.viewers.items():

            if key.endswith('_true'):
                tensor = outputs['batch'][key[:-5]]
            elif key.endswith('_pred'):
                tensor = outputs['preds'][key[:-5]]
            else:
                tensor = outputs['batch'][key]

            array = _numpy(tensor[0,0]) * mask

            viewer.update_array(array)

    def on_phase_end(self, epoch, phase):
        for key, viewer in self.viewers.items():
            out = self.output_dir / f'{key}_viewer.png'
            viewer.fig.savefig(out, bbox_inches='tight')

