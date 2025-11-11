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

    def eval(self, pred, target=None, weight=None, profile=None):
        names = self._profiles.get(profile, self._fns.keys())

        pred = np.asarray(pred, dtype=float)
        if pred.ndim > 2:
            raise ValueError(f'Expected (N, C) array, got {pred.shape}')
        elif pred.ndim == 1:
            pred = pred[:,None]

        if target is not None:
            target = np.asarray(target, dtype=float)
            if target.ndim > 2:
                raise ValueError(f'Expected (N, C) array, got {target.shape}')
            elif target.ndim == 1:
                target = target[:,None]
            if target.shape != pred.shape:
                raise ValueError(f'Shape mismatch: {pred.shape} vs {target.shape}')

        if weight is not None:
            weight = np.asarray(weight, dtype=float)
            if weight.ndim != 1:
                raise ValueError(f'Expected (N,) array, got {weight.shape}')
            if weight.shape[0] != pred.shape[0]:
                raise ValueError(f'Shape mismatch: {weight.shape} vs {pred.shape}')

        out = {}
        for n in names:
            fn, requires_target = self._fns[n]
            if requires_target and target is None:
                if which is not None: # user specified invalid metric
                    raise ValueError(f'metric {n} requires a target')
                continue
            elif requires_target:
                out[n] = fn(pred, target, weight)
            else:
                out[n] = fn(pred, weight)
        return out


# ----- public interface -----


def evaluate_metrics(pred, target=None, weight=None, profile=None):
    return registry.eval(pred, target, weight, profile)


# ----- helper functions -----


def _sum(a, w=None):
    return np.sum((a * w) if w is not None else a)


def _mean(a, w=None):
    return np.average(a, weights=w) if w is not None else np.mean(a)


def _rms(arr, w=None):
    return np.sqrt(_mean(arr**2, w))


# ----- metric definitions -----


def l2_norm(pred: np.ndarray, weight=None) -> float:
    '''
    mean(||pred||)
    '''
    mag = np.linalg.norm(pred, axis=1)
    return float(_mean(mag, weight))


def rms_norm(pred: np.ndarray, weight=None) -> float:
    '''
    sqrt(mean(||pred||^2))
    '''
    mag = np.linalg.norm(pred, axis=1)
    return float(_rms(mag, weight))


def absolute_error(pred: np.ndarray, target: np.ndarray, weight=None) -> float:
    '''
    MAE = mean(||pred - target||)
    '''
    err = np.linalg.norm(pred - target, axis=1)
    return float(_mean(err, weight))


def relative_error(pred: np.ndarray, target: np.ndarray, weight=None) -> float:
    '''
    MRE = sum(||pred - target||) / sum(||target||)
    '''
    err = np.linalg.norm(pred - target, axis=1)
    mag = np.linalg.norm(target, axis=1)
    num = _sum(err, weight)
    den = _sum(mag, weight) + EPS
    return float(num / den)


def absolute_rmse(pred: np.ndarray, target: np.ndarray, weight=None) -> float:
    '''
    RMSE = sqrt(mean(||pred - target||^2))
    '''
    err = np.linalg.norm(pred - target, axis=1)
    return float(_rms(err, weight))


def normalized_rmse(pred: np.ndarray, target: np.ndarray, weight=None) -> float:
    '''
    NRMSE = RMS(||pred - target||) / RMS(||target||)
    '''
    err = np.linalg.norm(pred - target, axis=1)
    mag = np.linalg.norm(target, axis=1)
    num = _rms(err, weight)
    den = _rms(mag, weight) + EPS
    return float(num / den)


def pearson_corr(pred: np.ndarray, target: np.ndarray, weight=None) -> float:
    '''
    Pearson's correlation coefficient (component-wise)
    '''
    pred, target = pred.flatten(), target.flatten()
    if len(np.unique(target)) < 2:
        return np.nan
    result = scipy.stats.pearsonr(pred, target)
    return float(result.statistic)


def spearman_corr(pred: np.ndarray, target: np.ndarray, weight=None) -> float:
    '''
    Spearman's rank correlation coefficient (component-wise)
    '''
    pred, target = pred.flatten(), target.flatten()
    if len(np.unique(target)) < 2:
        return np.nan
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

