import numpy as np
import scipy.stats

EPS = 1e-12


class MetricRegistry:

    def __init__(self):
        self._fns = {}
        self._profiles = {}

    def add_metric(self, name, fn, requires_target: bool):
        self._fns[name] = (fn, requires_target)

    def add_profile(self, name, which):
        self._profiles[name] = list(which)

    def eval(self, pred, target, which=None):
        names = self._profiles.get(which, self._fns.keys())

        pred = np.asarray(pred, dtype=float)
        if pred.ndim != 2:
            raise ValueError(f'expected (N, C) array, got {pred.shape}')
        if target is not None:
            target = np.asarray(target, dtype=float)
            if target.ndim != 2:
                raise ValueError(f'expected (N, C) array, got {target.shape}')
            if target.shape != pred.shape:
                raise ValueError(f'shape mismatch: {pred.shape} vs {target.shape}')

        out = {}
        for n in names:
            fn, requires_target = self._fns[n]
            if requires_target and target is None:
                if which is not None: # user specified invalid metric
                    raise ValueError(f'metric {n} requires a target')
                continue
            elif requires_target:
                out[n] = fn(pred, target)
            else:
                out[n] = fn(pred)
        return out


def evaluate_metrics(pred, target=None, which=None):
    return registry.eval(pred, target, which)


# ----- helper functions -----

def _mean(a, w=None):
    return np.average(a, weights=w) if w is not None else np.mean(a)


def _rms(arr, w=None):
    return np.sqrt(_mean(arr**2, w))


# ----- metric definitions -----


def l2_norm(pred: np.ndarray) -> float:
    '''
    mean(||pred||)
    '''
    mag = np.linalg.norm(pred, axis=1)
    return float(_mean(mag))


def rms_norm(pred: np.ndarray) -> float:
    '''
    sqrt(mean(||pred||^2))
    '''
    mag = np.linalg.norm(pred, axis=1)
    return float(_rms(mag))


def absolute_error(pred: np.ndarray, target: np.ndarray) -> float:
    '''
    MAE = mean(||pred - target||)
    '''
    err = np.linalg.norm(pred - target, axis=1)
    return float(_mean(err))


def relative_error(pred: np.ndarray, target: np.ndarray) -> float:
    '''
    MRE = sum(||pred - target||) / sum(||target||)
    '''
    err = np.linalg.norm(pred - target, axis=1)
    mag = np.linalg.norm(target, axis=1)
    num = np.sum(err)
    den = np.sum(mag) + EPS
    return float(num / den)


def absolute_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    '''
    RMSE = sqrt(mean(||pred - target||^2))
    '''
    err = np.linalg.norm(pred - target, axis=1)
    return float(_rms(err))


def normalized_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    '''
    NRMSE = RMS(||pred - target||) / RMS(||target||)
    '''
    err = np.linalg.norm(pred - target, axis=1)
    mag = np.linalg.norm(target, axis=1)
    num = _rms(err)
    den = _rms(mag) + EPS
    return float(num / den)


def pearson_corr(pred: np.ndarray, target: np.ndarray) -> float:
    '''
    Pearson's correlation coefficient (component-wise)
    '''
    pred, target = pred.flatten(), target.flatten()
    result = scipy.stats.pearsonr(pred, target)
    return float(result.statistic)


def spearman_corr(pred: np.ndarray, target: np.ndarray) -> float:
    '''
    Spearman's rank correlation coefficient (component-wise)
    '''
    pred, target = pred.flatten(), target.flatten()
    result = scipy.stats.spearmanr(pred, target)
    return float(result.statistic)


# ----- register metrics and profiles -----

registry = MetricRegistry()

registry.add_metric('norm', l2_norm,  requires_target=False)
registry.add_metric('rms',  rms_norm, requires_target=False)

registry.add_metric('mae',   absolute_error,  requires_target=True)
registry.add_metric('mre',   relative_error,  requires_target=True)
registry.add_metric('rmse',  absolute_rmse,   requires_target=True)
registry.add_metric('nrmse', normalized_rmse, requires_target=True)
registry.add_metric('pcorr', pearson_corr,    requires_target=True)
registry.add_metric('scorr', spearman_corr,   requires_target=True)

registry.add_profile('u', ['norm', 'rms', 'rmse', 'nrmse', 'pcorr', 'scorr'])
registry.add_profile('E', ['norm', 'rms',  'rmse', 'nrmse', 'pcorr', 'scorr'])
registry.add_profile('res', ['norm', 'rms'])

