import numpy as np
import scipy.stats

EPS = 1e-12


class MetricRegistry:

    def __init__(self):
        self._fns = {}
        self._profiles = {}

    def add_metric(self, name, fn, requires_target: bool):
        self._fns[name] = (fn, requires_target)

    def add_profile(self, name, metrics):
        assert all(name in self._fns for name in metrics)
        self._profiles[name] = list(metrics)

    def eval(self, pred, target=None, weight=None, profile=None):
        '''
        Args:
            pred:   (N, C) float array
            target: (N, C) float array
            weight: (N,) float array
            profile: str
        '''
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
                raise ValueError(f'Length mismatch: {weight.shape} vs {pred.shape}')

        fn_names = self._profiles[profile] or self._fns.keys()

        outputs = {}
        for name in fn_names:
            fn, requires_target = self._fns[name]
            if requires_target and target is None:
                if profile is not None:
                    raise ValueError(f'metric {name} requires a target')
                # with no profile specified, just skip invalid metrics
                continue
            elif requires_target:
                outputs[name] = fn(pred, target, weight)
            else:
                outputs[name] = fn(pred, weight)
        return outputs


# ----- public interface -----


def evaluate_metrics(pred, target=None, weight=None, profile=None):
    return registry.eval(pred, target, weight, profile)


# ----- helper functions -----


def _sum(a, w=None) -> float:
    a = np.asarray(a, dtype=float)
    if w is None:
        return float(np.sum(a))
    w = np.asarray(w).flatten()
    assert w.size == a.shape[0]
    assert np.isfinite(w).all()
    assert (w >= 0).all()
    assert w.sum() > 0
    return float(np.sum(a * w))


def _mean(a, w=None) -> float:
    a = np.asarray(a, dtype=float)
    if w is None:
        return float(np.mean(a))
    n, d = _sum(a, w), _sum(w)
    return float(_divide(n, d))


def _rms(a, w=None) -> float:
    return float(np.sqrt(_mean(a**2, w)))


def _std(a, w=None) -> float:
    return float(_rms(a - _mean(a, w), w))


def _divide(n: float, d: float) -> float:
    assert np.isfinite(n)
    assert np.isfinite(d)
    if np.isclose(d, 0):
        return float(np.nan)
    return float(n / d)


# ----- metric definitions -----


def mean_norm(pred: np.ndarray, weight=None) -> float:
    '''
    mean(||pred||)
    '''
    mag = np.linalg.norm(pred, axis=1)
    return _mean(mag, weight)


def std_norm(pred: np.ndarray, weight=None) -> float:
    '''
    std(||pred||)
    '''
    mag = np.linalg.norm(pred, axis=1)
    return _std(mag, weight)


def rms_norm(pred: np.ndarray, weight=None) -> float:
    '''
    sqrt(mean(||pred||^2))
    '''
    mag = np.linalg.norm(pred, axis=1)
    return _rms(mag, weight)


def absolute_error(pred: np.ndarray, target: np.ndarray, weight=None) -> float:
    '''
    mean(||pred - target||)
    '''
    err = np.linalg.norm(pred - target, axis=1)
    return _mean(err, weight)


def relative_error(pred: np.ndarray, target: np.ndarray, weight=None) -> float:
    '''
    sum(||pred - target||) / sum(||target||)
    '''
    err = np.linalg.norm(pred - target, axis=1)
    mag = np.linalg.norm(target, axis=1)
    num = _sum(err, weight)
    den = _sum(mag, weight)
    return _divide(num, den)


def absolute_rmse(pred: np.ndarray, target: np.ndarray, weight=None) -> float:
    '''
    RMSE = sqrt(mean(||pred - target||^2))
    '''
    err = np.linalg.norm(pred - target, axis=1)
    return _rms(err, weight)


def normalized_rmse(pred: np.ndarray, target: np.ndarray, weight=None) -> float:
    '''
    NRMSE = RMS(||pred - target||) / RMS(||target||)
    '''
    err = np.linalg.norm(pred - target, axis=1)
    mag = np.linalg.norm(target, axis=1)

    num = _rms(err, weight)
    den = _rms(mag, weight)

    return _divide(num, den)


def standardized_rmse(pred: np.ndarray, target: np.ndarray, weight=None) -> float:
    '''
    SRMSE = RMS(||pred - target||) / STD(||target||)
    '''
    err = np.linalg.norm(pred - target, axis=1)
    mag = np.linalg.norm(target, axis=1)

    num = _rms(err, weight)
    den = _std(mag, weight)

    return _divide(num, den)


def pearson_corr(pred: np.ndarray, target: np.ndarray, weight=None) -> float:
    '''
    Pearson's correlation coefficient (component-wise)
    '''
    pred, target = pred.flatten(), target.flatten()
    if len(np.unique(target)) < 2:
        return np.nan

    return float(scipy.stats.pearsonr(pred, target).statistic)


def spearman_corr(pred: np.ndarray, target: np.ndarray, weight=None) -> float:
    '''
    Spearman's rank correlation coefficient (component-wise)
    '''
    pred, target = pred.flatten(), target.flatten()
    if len(np.unique(target)) < 2:
        return np.nan

    return float(scipy.stats.spearmanr(pred, target).statistic)


def dice_score(pred: np.ndarray, target: np.ndarray, weight=None) -> float:
    '''
    Dice coefficient: 2|A ^ B| / (|A| + |B|)
    '''
    A, B = (pred > 0), (target > 0)
    num = 2 * (A & B).sum()
    den = A.sum() + B.sum()
    return _divide(num, den)


# ----- register metrics / profiles -----

registry = MetricRegistry()

registry.add_metric('mean', mean_norm, requires_target=False)
registry.add_metric('rms',  rms_norm,  requires_target=False)

registry.add_metric('mae',   absolute_error,  requires_target=True)
registry.add_metric('mre',   relative_error,  requires_target=True)
registry.add_metric('rmse',  absolute_rmse,   requires_target=True)
registry.add_metric('nrmse', normalized_rmse, requires_target=True)
registry.add_metric('pcorr', pearson_corr,    requires_target=True)
registry.add_metric('scorr', spearman_corr,   requires_target=True)

registry.add_metric('srmse', standardized_rmse, requires_target=True)
registry.add_metric('dice', dice_score, requires_target=True)

registry.add_profile('rho', ['mean', 'rms', 'rmse', 'nrmse', 'srmse', 'pcorr', 'scorr'])
registry.add_profile('E', ['mean', 'rms', 'rmse', 'nrmse', 'srmse', 'pcorr', 'scorr'])
registry.add_profile('u', ['mean', 'rms', 'rmse', 'nrmse', 'srmse', 'pcorr', 'scorr'])
registry.add_profile('res', ['mean', 'rms'])
registry.add_profile('mat', ['dice'])

