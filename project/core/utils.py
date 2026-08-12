from typing import List, Dict, Iterable, Optional, Any
import sys, time, random


# ----- logging functions -----


VERBOSE = True

def set_verbose(val: bool) -> None:
    global VERBOSE
    VERBOSE = val


def log(msg: str, end: str = '\n') -> None:
    is_worker = False

    if 'torch' in sys.modules:
        import torch
        is_worker = torch.utils.data.get_worker_info()

    if VERBOSE and not is_worker:
        print(msg, end=end, file=sys.stdout, flush=True)


def warn(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ----- common utilities -----


def is_iterable(obj: Any, string_ok: bool = False) -> bool:
    if isinstance(obj, str):
        return string_ok
    return hasattr(obj, '__iter__')


def as_iterable(
    obj: Any, string_ok: bool = False, length: int = 1
) -> Iterable:
    if not is_iterable(obj, string_ok):
        return [obj] * length
    return obj


def check_keys(
    config: Dict[str, Any],
    valid: Iterable[str],
    where: Optional[str] = None
) -> None:
    invalid = set(config.keys()) - set(valid)
    if invalid:
        loc = f' for {where}' if where else ''
        raise KeyError(f'Unexpected keys{loc}: {invalid} vs. {valid}')


def update_defaults(overrides: Optional[Dict] = None, **defaults) -> Dict:
    return defaults | (overrides or {})


def namespace(dct: Dict[str, Any], name: str) -> Dict[str, Any]:
    return {f'{name}.{k}': v for k, v in dct.items()}


def missing_value(val: Any, strings: Iterable[str] = ('',)) -> bool:
    import pandas as pd
    return pd.isna(val) or str(val).strip() in set(strings)


def make_seed(*parts) -> int:
    import hashlib
    s = ':'.join([str(part) for part in parts])
    h = hashlib.sha256(s.encode('utf-8')).digest()
    return int.from_bytes(h[:8], byteorder='little', signed=False)


def pprint(*args, **kwargs) -> None:
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

