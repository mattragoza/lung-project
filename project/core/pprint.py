from . import utils


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

    # ----- render subtree -----

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

    return out if ret_string else utils.log(out)

