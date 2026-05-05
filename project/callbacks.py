from .core import utils


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


