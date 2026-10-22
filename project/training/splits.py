from collections import defaultdict
import numpy as np

from ..common import utils


def split_data(examples, method, **kwargs):
    utils.log('Splitting data examples')
    if method == 'random':
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
    examples = list(examples)

    cats_by_sid = defaultdict(set)
    sids_by_cat = defaultdict(set)

    for ex in examples:
        sid = ex.subject
        for cat in ex.metadata[key]:
            if str(cat).startswith('_'):
                continue
            cats_by_sid[sid].add(cat)
            sids_by_cat[cat].add(sid)

    subjects = list(sorted(cats_by_sid.keys()))
    categories = list(sorted(sids_by_cat.keys()))

    n_subjects = len(subjects)
    target_test = int(round(test_ratio * n_subjects))

    rng = np.random.default_rng(seed)
    rng.shuffle(categories)

    test_cats = set()
    test_sids = set()

    for cat in categories:
        if len(test_sids) >= target_test:
            break
        eligible = {
            sid for sid, c in cats_by_sid.items()
                if c.issubset(test_cats | {cat})
        }
        if eligible:
            test_cats.add(cat)
            test_sids |= eligible

    utils.log(f'Test categories: {test_cats}')
    utils.log(f'Test subjects:   {test_sids}')

    train_sids = set(subjects) - test_sids
    train_list = list(train_sids)
    rng.shuffle(train_list)

    n_val = int(round(val_ratio * n_subjects))

    val_subj = set(train_list[:n_val])
    train_subj = set(train_list[n_val:])

    utils.log(f'Train subjects: {train_subj}')
    utils.log(f'Val subjects:   {val_subj}')

    train_ex = [ex for ex in examples if ex.subject in train_subj]
    test_ex  = [ex for ex in examples if ex.subject in test_subj]
    val_ex   = [ex for ex in examples if ex.subject in val_subj]

    return train_ex, test_ex, val_ex

