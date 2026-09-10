from typing import Dict, Iterable, Optional, Any

import sys, random

from .outputs import Outputs
from .pprint import pprint
from .timer import Timer

VERBOSE = True


# ----- logging functions -----


def set_verbose(val: bool) -> None:
    global VERBOSE
    VERBOSE = bool(val)


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

