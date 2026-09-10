from typing import List, Dict, Callable, Optional
import numpy as np
import scipy.stats

from . import utils

EPS = 1e-12


class MetricRegistry:

    def __init__(self):
        self._metrics = {}
        self._profiles = {}

    def add_metric(self, name: str, func: Callable, uses_target: bool):
        self._metrics[name] = (func, uses_target)

    def add_profile(self, name: str, metrics: List[str]):
        missing = [m for m in metrics if m not in self._metrics]
        if missing:
            raise KeyError(f'Metrics not found in registry: {missing}')
        self._profiles[name] = list(metrics)

    def evaluate_metric(
        self,
        name: str,
        pred: np.ndarray,
        target: Optional[np.ndarray] = None,
        weight: Optional[np.ndarray] = None,
    ) -> Optional[float]:

        if name not in self._metrics:
            raise KeyError(f'Metric not in registry: {name}')
        func, uses_target = self._metrics[name]

        pred = np.asarray(pred, dtype=float)
        if pred.ndim == 1:
            pred = pred[:,np.newaxis]
        if pred.ndim != 2:
            raise ValueError(f'Expected (N, C) pred array, got {pred.shape}')

        if target is not None and uses_target:
            target = np.asarray(target, dtype=float)
            if target.ndim == 1:
                target = target[:,np.newaxis]
            if target.ndim != 2:
                raise ValueError(f'Expected (N, C) target array, got {target.shape}')
            if target.shape != pred.shape:
                raise ValueError(f'Shape mismatch: {pred.shape} vs. {target.shape}')

        if weight is not None:
            weight = np.asarray(weight, dtype=float)
            if weight.ndim != 1:
                raise ValueError(f'Expected (N,) weight array, got {weight.shape}')
            if weight.shape[0] != pred.shape[0]:
                raise ValueError(f'Shape mismatch: {weight.shape} vs. {pred.shape}')

        if uses_target and target is None:
            return None
        elif uses_target:
            return func(pred, target, weight)
        elif not uses_target:
            return func(pred, weight)

    def evaluate_metrics(
        self,
        pred: np.ndarray,
        target: Optional[np.ndarray] = None,
        weight: Optional[np.ndarray] = None,
        names: Optional[List[str]] = None
    ) -> Dict[str, float]:

        if names is None:
            names = self._metrics.keys()

        outputs = {}
        for name in names:
            value = self.evaluate_metric(name, pred, target, weight)
            if value is not None:
                outputs[name] = value

        return outputs

    def evaluate_profile(
        self,
        pred: np.ndarray,
        target: Optional[np.ndarray] = None,
        weight: Optional[np.ndarray] = None,
        profile: Optional[str] = None
    ) -> Dict[str, float]:

        if profile is None:
            names = self._metrics.keys()
        elif profile in self._profiles:
            names = self._profiles[profile]
        else:
            raise KeyError(f'Profile not in registry: {profile}')

        return self.evaluate_metrics(pred, target, weight, names)


# ---- public registry API -----


REGISTRY = MetricRegistry()
evaluate_metric = REGISTRY.evaluate_metric
evaluate_metrics = REGISTRY.evaluate_metrics
evaluate_profile = REGISTRY.evaluate_profile


# ----- basic math functions -----


def _sum(values: np.ndarray, weights: Optional[np.ndarray] = None) -> float:

    values = np.asarray(values, dtype=float)

    if values.ndim != 1:
        raise ValueError('values must be 1-d')
    if not np.isfinite(values).all():
        raise ValueError('values must be finite')

    if weights is None:
        return float(np.sum(values))

    weights = np.asarray(weights, dtype=float)

    if weights.ndim != 1:
        raise ValueError('weights must be 1-d')
    if weights.shape != values.shape:
        raise ValueError('invalid weights shape')
    if not np.isfinite(weights).all():
        raise ValueError('weights must be finite')
    if not np.all(weights >= 0):
        raise ValueError('weights must be non-negative')
    if not np.sum(weights) > 0:
        raise ValueError('total weight must be positive')

    return float(np.sum(values * weights))


def _divide(numer: float, denom: float) -> float:

    if not np.isfinite(numer):
        raise ValueError('numerator must be finite')
    if not np.isfinite(denom):
        raise ValueError('denominator must be finite')

    if abs(denom) < EPS:
        utils.warn('WARNING: division by zero')
        return float(np.nan)

    return float(numer / denom)


def _mean(values: np.ndarray, weight: Optional[np.ndarray] = None) -> float:

    values = np.asarray(values, dtype=float)

    if values.ndim != 1:
        raise ValueError('values must be 1-d')
    if not np.isfinite(values).all():
        raise ValueError('values must be finite')

    if weight is None:
        return float(np.mean(values))

    return float(_divide(_sum(values, weight), _sum(weight)))


def _rms(values: np.ndarray, weight: Optional[np.ndarray] = None) -> float:
    return float(np.sqrt(_mean(values**2, weight)))


def _std(values: np.ndarray, weight: Optional[np.ndarray] = None) -> float:
    deviation = values - _mean(values, weight)
    return float(_rms(deviation, weight))


# ----- metric definitions -----


def mean_norm(pred: np.ndarray, weight: Optional[np.ndarray] = None) -> float:
    '''mean(|pred|)'''
    return _mean(np.linalg.norm(pred, axis=1), weight)


def rms_norm(pred: np.ndarray, weight: Optional[np.ndarray] = None) -> float:
    '''rms(|pred|) = sqrt(mean(|pred|^2))'''
    return _rms(np.linalg.norm(pred, axis=1), weight)


def std_norm(pred: np.ndarray, weight: Optional[np.ndarray] = None) -> float:
    '''std(|pred|) = rms(|pred| - mean(|pred|))'''
    return _std(np.linalg.norm(pred, axis=1), weight)


def absolute_error(pred: np.ndarray, target: np.ndarray, weight=None) -> float:
    '''mean(|pred - target|)'''
    return _mean(np.linalg.norm(pred - target, axis=1), weight)


def relative_error(pred: np.ndarray, target: np.ndarray, weight=None) -> float:
    '''sum(|pred - target|) / sum(|target|)'''
    numer = _sum(np.linalg.norm(pred - target, axis=1), weight)
    denom = _sum(np.linalg.norm(target, axis=1), weight)
    return _divide(numer, denom)


def absolute_rmse(pred: np.ndarray, target: np.ndarray, weight=None) -> float:
    '''rms(|pred - target|)'''
    return _rms(np.linalg.norm(pred - target, axis=1), weight)


def normalized_rmse(pred: np.ndarray, target: np.ndarray, weight=None) -> float:
    '''rms(|pred - target|) / rms(|target|)'''
    num = _rms(np.linalg.norm(pred - target, axis=1), weight)
    den = _rms(np.linalg.norm(target, axis=1), weight)
    return _divide(num, den)


def pearson_corr(pred: np.ndarray, target: np.ndarray, weight=None) -> float:
    '''Pearson's correlation coefficient'''

    pred, target = pred.flatten(), target.flatten()

    if weight is not None:
        weight = np.repeat(weight, pred.size // weight.size)

    dev_p = pred - _mean(pred, weight) 
    dev_t = target - _mean(target, weight)

    var_p = _mean(dev_p * dev_p, weight)
    var_t = _mean(dev_t * dev_t, weight)

    if np.sqrt(var_p) < EPS:
        utils.warn('WARNING: pearson_corr is undefined for constant pred')
        return np.nan

    if np.sqrt(var_t) < EPS:
        utils.warn('WARNING: pearson_corr is undefined for constant target')
        return np.nan

    cov = _mean(dev_p * dev_t, weight)

    return _divide(cov, np.sqrt(var_p * var_t))


def spearman_corr(pred: np.ndarray, target: np.ndarray, weight=None) -> float:
    '''Spearman's rank correlation coefficient

    NOTE: Implemented as weighted Pearson correlation of ordinal ranks.
    '''
    pred, target = pred.flatten(), target.flatten()

    rank_p = scipy.stats.rankdata(pred)
    rank_t = scipy.stats.rankdata(target)

    return pearson_corr(rank_p, rank_t, weight)


def dice_score(pred: np.ndarray, target: np.ndarray, weight=None) -> float:
    '''Dice coefficient: 2*|A & B| / (|A| + |B|)'''

    A, B = (pred > 0).flatten(), (target > 0).flatten()

    numer = 2 * _sum(A & B, weight)
    denom = _sum(A, weight) + _sum(B, weight)

    return _divide(numer, denom)


# ----- register metrics and profiles -----

REGISTRY.add_metric('mean', mean_norm, uses_target=False)
REGISTRY.add_metric('rms', rms_norm, uses_target=False)
REGISTRY.add_metric('std', std_norm, uses_target=False)
REGISTRY.add_metric('mae', absolute_error, uses_target=True)
REGISTRY.add_metric('mre', relative_error, uses_target=True)
REGISTRY.add_metric('rmse', absolute_rmse, uses_target=True)
REGISTRY.add_metric('nrmse', normalized_rmse, uses_target=True)
REGISTRY.add_metric('pcorr', pearson_corr, uses_target=True)
REGISTRY.add_metric('scorr', spearman_corr, uses_target=True)
REGISTRY.add_metric('dice', dice_score, uses_target=True)

REGISTRY.add_profile('scalar', ['mean', 'std', 'rms', 'mae', 'mre', 'rmse', 'nrmse', 'pcorr', 'scorr'])
REGISTRY.add_profile('vector', ['mean', 'std', 'rms', 'mae', 'mre', 'rmse', 'nrmse'])
REGISTRY.add_profile('binary', ['dice'])

