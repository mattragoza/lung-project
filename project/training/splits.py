from collections import defaultdict
import numpy as np

from ..common import utils


def split_data(examples, method, **kwargs):
    utils.log('Splitting data examples')
    if method is None:
        return examples, [], []
    elif method == 'random':
        return random_split(examples, **kwargs)
    elif method == 'metadata':
        return metadata_split(examples, **kwargs)
    raise ValueError(f'Invalid split method: {method!r}')


def random_split(examples, test_ratio, val_ratio, seed=0):
    examples = sorted(examples, key=lambda x: x.subject)

    n_total = len(examples)
    n_test = int(round(test_ratio * n_total))
    n_val = int(round(val_ratio * n_total))
    n_train = n_total - n_test - n_val

    utils.log(f'Train examples: {n_train}')
    utils.log(f'Test examples:  {n_test}')
    utils.log(f'Val examples:   {n_val}')

    if not (0 < n_train <= n_total):
        raise ValueError(f'Invalid num train examples: {n_train}')

    if not (0 <= n_test < n_total):
        raise ValueError(f'Invalid num test examples: {n_test}')

    if not (0 <= n_val < n_total):
        raise ValueError(f'Invalid num val examples: {n_val}')

    if n_test == 0:
        utils.warn('WARNING: No test examples')

    if n_val == 0:
        utils.warn('WARNING: No val examples')

    rng = np.random.default_rng(seed)
    rng.shuffle(examples)

    train_ex = examples[:n_train]
    test_ex = examples[n_train:n_train+n_test]
    val_ex = examples[n_train+n_test:]

    return train_ex, test_ex, val_ex


def metadata_split(examples, key, test_ratio, val_ratio, seed=0):
    examples = sorted(examples, key=lambda x: x.subject)

    levels = sorted(set(ex.metadata[key] for ex in examples))
    n_total = len(levels)

    utils.log(f'All {key} levels: {n_total} {levels}')
    utils.log(f'Test ratio: {test_ratio:.2f}')
    utils.log(f'Val ratio:  {val_ratio:.2f}')

    n_test = int(round(test_ratio * n_total))
    n_val = int(round(val_ratio * n_total))
    n_train = n_total - n_test - n_val

    if not (0 < n_train <= n_total):
        raise ValueError(f'Invalid num train levels: {n_train}')

    if not (0 <= n_test < n_total):
        raise ValueError(f'Invalid num test levels: {n_test}')

    if not (0 <= n_val < n_total):
        raise ValueError(f'Invalid num val levels: {n_val}')

    if n_test == 0:
        utils.warn('WARNING: No test levels')

    if n_val == 0:
        utils.warn('WARNING: No val levels')

    utils.log(f'Split seed: {seed}')

    rng = np.random.default_rng(seed)
    rng.shuffle(levels)

    train_levels = levels[:n_train]
    test_levels = levels[n_train:n_train+n_test]
    val_levels = levels[n_train+n_test:]

    utils.log(f'Train {key} levels: {n_train} {train_levels}')
    utils.log(f'Test {key} levels:  {n_test} {test_levels}')
    utils.log(f'Val {key} levels:   {n_val} {val_levels}')

    train_ex = [ex for ex in examples if ex.metadata[key] in train_levels]
    test_ex  = [ex for ex in examples if ex.metadata[key] in test_levels]
    val_ex   = [ex for ex in examples if ex.metadata[key] in val_levels]

    utils.log(f'Train examples: {len(train_ex)} {[ex.subject for ex in train_ex]}')
    utils.log(f'Test examples:  {len(test_ex)} {[ex.subject for ex in test_ex]}')
    utils.log(f'Val examples:   {len(val_ex)} {[ex.subject for ex in val_ex]}')

    return train_ex, test_ex, val_ex

