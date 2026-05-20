import sys, time, random


VERBOSE = True


def set_verbose(val: bool):
    global VERBOSE
    VERBOSE = val


def log(msg, end='\n'):
    is_worker = False

    if 'torch' in sys.modules:
        import torch
        is_worker = torch.utils.data.get_worker_info()

    if VERBOSE and not is_worker:
        print(msg, end=end, file=sys.stdout, flush=True)


def warn(msg):
    print(msg, file=sys.stderr, flush=True)


def is_iterable(obj, string_ok=False):
    if isinstance(obj, str):
        return string_ok
    return hasattr(obj, '__iter__')


def as_iterable(obj, string_ok=False, length=1):
    if not is_iterable(obj, string_ok):
        return [obj] * length
    return obj


def update_defaults(overrides=None, **defaults):
    return defaults | (overrides or {})


def namespace(dct, name):
    return {f'{name}.{k}': v for k, v in dct.items()}


def missing_value(val):
    import pandas as pd
    return pd.isna(val) or str(val).strip() == ''


def check_keys(config, valid, where=None):
    invalid = set(config.keys()) - set(valid)
    if invalid:
        loc = f' for {where}' if where else ''
        raise KeyError(f'Unexpected keys{loc}: {invalid} vs. {valid}')


def make_seed(*parts):
    import hashlib
    s = ':'.join([str(part) for part in parts])
    h = hashlib.sha256(s.encode('utf-8')).digest()
    return int.from_bytes(h[:8], byteorder='little', signed=False)


def pprint(*args, **kwargs):
    from .pprint import pprint as pprint_
    return pprint_(*args, **kwargs)


class Timer:

    def __init__(self):
        import torch
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.t_last = time.perf_counter()

    def tick(self, sync=False, unit_b=2**30):
        import torch
        if sync:
            torch.cuda.synchronize()
        curr_alloc = torch.cuda.memory_allocated() / unit_b
        curr_rsvd  = torch.cuda.memory_reserved() / unit_b
        peak_alloc = torch.cuda.max_memory_allocated() / unit_b
        peak_rsvd  = torch.cuda.max_memory_reserved() / unit_b
        torch.cuda.reset_peak_memory_stats()
        t_curr = time.perf_counter()
        t_delta = t_curr - self.t_last
        self.t_last = t_curr
        return {
            't_delta': round(t_delta, 4),
            'curr_alloc': round(curr_alloc, 4),
            'curr_rsvd':  round(curr_rsvd, 4),
            'peak_alloc': round(peak_alloc, 4),
            'peak_rsvd':  round(peak_rsvd, 4)
        }

