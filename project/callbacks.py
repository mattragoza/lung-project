# callbacks.py

from typing import List, Dict, Optional
from pathlib import Path

from .common import utils


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

    def on_batch_start(self, epoch: int, phase: str, batch_idx: int):
        return

    def on_batch_end(self, epoch: int, phase: str, batch_idx: int, outputs: dict):
        return


class LoggingCallback(Callback):

    def on_train_start(self):
        utils.log('Start training')

    def on_train_end(self):
        utils.log('End training')

    def on_epoch_start(self, epoch: int):
        utils.log(f'Start epoch {epoch}')

    def on_epoch_end(self, epoch: int):
        utils.log(f'End epoch {epoch}')

    def on_phase_start(self, epoch: int, phase: str):
        utils.log(f'Start epoch {epoch} {phase} phase')

    def on_phase_end(self, epoch: int, phase: str):
        utils.log(f'End epoch {epoch} {phase} phase')

    def on_batch_start(self, epoch: int, phase: str, batch_idx: int):
        utils.log(f'[Epoch {epoch} {phase} batch {batch_idx}] start')

    def on_batch_end(self, epoch: int, phase: str, batch_idx: int, outputs: dict):
        loss = float(outputs['loss'].item())
        utils.log(f'[Epoch {epoch} {phase} batch {batch_idx}] loss = {loss:.4f}')


class TimerCallback(Callback):

    def __init__(self):
        self.timer = utils.Timer()

    def on_phase_start(self, epoch: int):
        self.timer.tick()

    def on_batch_start(self, epoch: int, phase: str, batch_idx: int):
        self.timer.tick() # load data into batch

    def on_batch_end(self, epoch: int, phase: str, batch_idx: int, outputs: dict):
        self.timer.tick() # batch forward/backward


class EvaluatorCallback(Callback):

    def __init__(
        self,
        evaluator,
        output_dir: Path,
        group_by: Optional[str] = None,
        on_train: bool = False
    ):
        self.evaluator = evaluator
        self.group_by = group_by
        self.on_train = on_train

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = _get_new_path(self.output_dir / 'metrics.csv')

        self.rows = []

    def on_batch_end(
        self, epoch: int, phase: str, batch_idx: int, outputs: dict
    ):
        if phase == 'train' and not self.on_train:
            return # skip train eval

        for values in self.evaluator.evaluate_batch(
            outputs['data_batch'],
            outputs['model_preds'],
            outputs['sim_outputs'],
            groupby=None
        ):
            self.rows.append({
                'epoch': int(epoch),
                'phase': str(phase),
                'batch': int(batch_idx),
                **values
            })

    def on_phase_end(self, epoch: int, phase: str):
        if not self.rows:
            return
        import pandas as pd

        df = pd.DataFrame(self.rows)

        tmp_path = self.csv_path.with_suffix('.tmp')

        try:
            df.to_csv(tmp_path, index=False)
            tmp_path.replace(self.csv_path)

        finally:
            tmp_path.unlink(missing_ok=True)


def _get_new_path(path: Path) -> Path:
    while path.is_file():
        path = Path(str(path) + '.new')
    return path


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

    def on_batch_end(self, epoch, phase, batch, step, outputs):
        phase = str(phase).lower()
        for key in self.history:
            if key == 'mat_pred':
                outputs = ensure_material_preds(outputs)
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

