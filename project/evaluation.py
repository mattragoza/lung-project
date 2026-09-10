from typing import List, Dict, Iterable, Any, Optional

import numpy as np
import torch

from .core import utils, metrics


def _to_numpy(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _flatten_voxels(x) -> np.ndarray:
    '''(C, I, J, K) -> (N, C)'''
    x = _to_numpy(x)

    if x.ndim != 4:
        raise ValueError(f'Expected (C, I, J, K), got {x.shape}')

    return x.reshape(x.shape[0], -1).T


def _flatten_scalar(x) -> np.ndarray:
    '''(1, I, J, K) -> (N,)'''
    x = _flatten_voxels(x)

    if x.shape[1] != 1:
        raise ValueError(f'Expected (N, 1), got {x.shape}')

    return x[:,0]


def _index_batch(
    batch: Dict[str, Iterable[Any]] | None, index: int
) -> Dict[str, Any] | None:
    '''batch[key][index] -> output[key]'''
    if batch is None:
        return None

    output = {}
    for key, value in batch.items():
        if utils.is_iterable(value, string_ok=False):
            output[key] = value[index]
        else:
            output[key] = value

    return output


def _index_maybe(values: List[Any] | None, index: int) -> Any:
    '''values[index] | None'''
    return values[index] if values is not None else None


def _cell_values(field):
    return _to_numpy(field.cell_values) if field.cell_values else None


def _evaluate(
    name: str,
    pred: np.ndarray,
    true: Optional[np.ndarray] = None,
    weight: Optional[np.ndarray] = None,
    profile: Optional[str] = None
) -> Dict[str, float]:
    values = metrics.evaluate_profile(pred, true, weight, profile)
    return utils.namespace(values, name)


class Evaluator:

    def evaluate_batch(
        self,
        batch: Dict[str, Iterable[Any]],
        preds: Optional[dict] = None,
        sims:  Optional[list] = None,
        groupby: Optional[str] = None
    ) -> List[Dict[str, float]]:

        all_rows = []
        for sample_idx in range(len(batch['example'])):

            new_rows = self.evaluate_sample(
                _index_batch(batch, sample_idx),
                _index_batch(preds, sample_idx),
                _index_maybe(sims, sample_idx),
                groupby
            )
            all_rows.extend(new_rows)

        return all_rows

    def evaluate_sample(
        self,
        sample: Dict[str, Any],
        pred: Optional[dict] = None,
        sim:  Optional[dict] = None,
        groupby: Optional[str] = None
    ) -> List[Dict[str, float]]:

        # full-domain evaluation
        rows = [self.evaluate_group(sample, pred, sim)]

        if groupby is None:
            return rows

        # subdomain evaluation via groupby labels
        labels = _flatten_scalar(sample[groupby])

        for label in np.unique(labels[labels != 0]):
            rows.append(self.evaluate_group(
                sample, pred, sim, groupby, label
            ))

        return rows

    def evaluate_group(
        self,
        sample: Dict[str, Any],
        pred: Optional[dict] = None,
        sim:  Optional[dict] = None,
        group: Optional[str] = None,
        label: Optional[int] = None
    ) -> Dict[str, float]:

        values = {'group': group, 'label': label}

        if pred is not None:
            values |= self.evaluate_voxels(sample, pred, group, label)

        if sim is not None:
            values |= self.evaluate_mesh(sim, group, label)

        return values

    def evaluate_voxels(
        self,
        true_vols: Dict[str, Any],
        pred_vols: Dict[str, Any],
        group: Optional[str] = None,
        label: Optional[int] = None
    ) -> Dict[str, float]:

        selected = _flatten_scalar(true_vols['mask'])

        if group is not None:
            selected &= _flatten_scalar(true_vols[group]) == label

        values = {'num_voxels': int(selected.sum())}

        if not selected.any():
            utils.warn(f'WARNING: zero voxels selected')
            return values

        for name, pred_vox in pred_vols.items():
            pred_vox = _flatten_voxels(pred_vox)

            true_vox = None
            if name in sample:
                true_vox = _flatten_voxels(true_vols[name])

            values |= _evaluate(
                name=f'{name}_vox',
                pred=_index_maybe(pred_vox, selected),
                true=_index_maybe(true_vox, selected),
                profile=self.task.metric_profile(name)
            )

        return values

    def evaluate_mesh(
        self,
        sim: Dict[str, Any],
        group: Optional[str] = None,
        label: Optional[int] = None
    ) -> dict:

        cell_volume = _to_numpy(sim['ctx'].volume)
        true_fields = sim['ctx'].fields
        pred_fields = sim['params']

        selected = np.ones(len(cell_volume), dtype=bool)

        if group is not None:
            selected &= _cell_values(true_fields[group]) == label

        values = {'num_cells': int(selected.sum())}

        if not selected.any():
            utils.warn(f'WARNING: zero cells seleted')
            return values

        weight = cell_volume[selected]

        for name, pred_field in pred_fields.items():
            pred_cells = _cell_values(pred_field)

            true_cells = None
            if name in true_fields:
                true_cells = _cell_values(true_fields[name])

            values |= _evaluate(
                name=f'{name}_cell',
                pred=_index_maybe(pred_cells, selected),
                true=_index_maybe(true_cells, selected),
                weight=cell_volume,
                profile=self.task.metric_profile(name)
            )

        values |= _evaluate(
            name='u_cell',
            pred=_index_maybe(_cell_values(sim['u_sim']), selected),
            true=_index_maybe(_cell_values(sim['u_obs']), selected),
            weight=weight,
            profile='vector'
        )

        values |= _evaluate(
            name='res_cell',
            pred=_index_maybe(_cell_values(sim['residual']), selected),
            weight=weight,
            profile='vector'
        )

        return values


def ensure_material_preds(outputs):
    if 'mat_pred' in outputs:
        return outputs

    elif 'mat_logits' in outputs:
        mat_logits = outputs['mat_logits']                # (B,C,I,J,K)
        mat_pred = torch.argmax(mat_logits, dim=1)        # (B,I,J,K)
        outputs['mat_pred'] = mat_pred.unsqueeze(1).cpu() # (B,1,I,J,K)
        return outputs

    for key in ['mat_true', 'E_true', 'E_pred']:
        if key not in outputs:
            utils.warn('WARNING: Cannot estimate material map from provided outputs.')
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


# DEPRECATED


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

