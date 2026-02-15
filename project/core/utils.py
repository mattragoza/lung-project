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


def make_seed(*parts):
    import hashlib
    s = ':'.join([str(p) for p in parts])
    h = hashlib.sha256(s.encode('utf-8')).digest()
    return int.from_bytes(h[:8], byteorder='little', signed=False)


def pprint(
    obj,
    max_depth=2,
    max_items=10,
    show_hidden=False, 
    ret_string=False,
    _depth=0,
    _tab=''
):
    import numpy as np
    import pandas as pd
    import torch
    type_name = type(obj).__name__

    if isinstance(obj, torch.Tensor):
        if obj.ndim > 0:
            desc  = f'{type_name}(shape={obj.shape}, dtype={obj.dtype}, device={obj.device})'
            items = [(i, v) for i, v in enumerate(obj)]
        else:
            desc = repr(obj).replace('tensor', str(obj.dtype))
            items = None

    elif isinstance(obj, np.ndarray):
        desc  = f'{type_name}(shape={obj.shape}, dtype={obj.dtype})'
        items = [(i, v) for i, v in enumerate(obj)]

    elif isinstance(obj, pd.DataFrame):
        desc  = f'{type_name}(shape={obj.shape})'
        items = [(repr(k), v) for k, v in obj.items()]

    elif isinstance(obj, pd.Series):
        desc  = f'{type_name}(len={len(obj)}, dtype={obj.dtype})'
        items = [(repr(obj.index[i]), obj.iloc[i]) for i in range(len(obj))]

    elif isinstance(obj, dict):
        desc  = f'{type_name}(len={len(obj)})'
        items = [(repr(k), v) for k, v in obj.items()]

    elif isinstance(obj, (list, tuple)):
        desc  = f'{type_name}(len={len(obj)})'
        items = [(i, v) for i, v in enumerate(obj)]

    elif isinstance(obj, set):
        desc  = f'{type_name}(len={len(obj)})'
        items = [(':', v) for v in sorted(obj, key=repr)]

    elif hasattr(obj, '__dict__'): # generic object
        desc  = f'{type_name}()'
        items = [
            (k, v) for k, v in vars(obj).items() 
                if k[0] != '_' or show_hidden
        ]
    else: # scalar type
        desc = repr(obj)
        items = None

    out = desc

    # --- render subtree ---
    if items and _depth < max_depth:
        shown = items[:max_items]
        fmt_key = lambda x: str(x) + ':'
        max_key_len = max((len(fmt_key(k)) for k, v in shown), default=0)

        for i, (key, val) in enumerate(shown):
            last_item = (i + 1) == len(items)    
            item_tab  = _tab + ('└── ' if last_item else '├── ')
            child_tab = _tab + ('    ' if last_item else '|   ')
    
            key_str = fmt_key(key).ljust(max_key_len)
            val_str = pprint(
                val,
                max_depth=max_depth,
                max_items=max_items,
                show_hidden=show_hidden,
                ret_string=True,
                _depth=_depth+1,
                _tab=child_tab
            )
            out += f'\n{item_tab}{key_str} {val_str}'

        if len(items) > max_items:
            not_shown = len(items) - max_items
            out += f'\n{_tab}└── <{not_shown} more items>'

    return out if ret_string else print(out)


def check_keys(config, valid, where=None):
    invalid = set(config.keys()) - set(valid)
    if invalid:
        _loc = f' for {where}' if where else ''
        raise KeyError(f'Unexpected keys{_loc}: {invalid} vs. {valid}')


def update_defaults(overrides=None, **defaults):
    return defaults | (overrides or {})


def missing(val):
    import pandas as pd
    return pd.isna(val) or str(val).strip() == ''


def namespace(dct, name):
    return {f'{name}.{k}': v for k, v in dct.items()}


def is_iterable(obj, string_ok=False):
    if isinstance(obj, str):
        return string_ok
    return hasattr(obj, '__iter__')


def as_iterable(obj, string_ok=False):
    return obj if is_iterable(obj, string_ok) else [obj]


def as_bool(val):
    if isinstance(val, str):
        val = val.lower()
        if val in {'true', 't', '1'}:
            return True
        elif val in {'false', 'f', '0'}:
            return False
        raise ValueError(f'Invalid boolean string: {val:r}')
    return bool(val)


def generate_argument_parser(func):
    import inspect, argparse

    # get full argument specification
    argspec = inspect.getfullargspec(func)
    args = argspec.args or []
    defaults = argspec.defaults or ()
    undefined = object() # sentinel object
    n_undefined = len(args) - len(defaults)
    defaults = (undefined,) * n_undefined + defaults

    # auto-generate argument parser
    parser = argparse.ArgumentParser()
    for name, default in zip(argspec.args, defaults):
        type_ = argspec.annotations.get(name, None)

        if default is undefined: # positional argument
            parser.add_argument(name, type=type_)

        elif default is False or default is True and type_ in {bool, None}: # flag
            parser.add_argument(
                f'--{name}', default=False, type=as_bool, help=f'[{default}]'
            )
        else: # optional argument
            if type_ is None and default is not None:
                type_ = type(default)
            parser.add_argument(
                f'--{name}', default=default, type=type_, help=f'[{default}]'
            )

    return parser


def main(func):
    '''
    Decorator for auto parsing arguments and calling the main function
    '''
    import inspect
    parent = inspect.stack()[1].frame
    __name__ = parent.f_locals.get('__name__')
    if __name__ == '__main__':

        # parse and display command line arguments
        parser = generate_argument_parser(func)
        kwargs = vars(parser.parse_args(sys.argv[1:]))
        print(kwargs)

        # call the main function
        func(**kwargs)

    return func


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

