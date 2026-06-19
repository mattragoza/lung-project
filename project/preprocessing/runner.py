# preprocessing/runner.py

from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

from ..core import utils


def run_stage(func, *args, **kwargs) -> Tuple[bool, Any]:
    '''
    Call function if output path does not exist.

    Args:
        func: Function to call.
        *args, **kwargs: Passed into function call.
            kwargs['output_path']: The checked path.
    Returns:
        bool: True if the function was called.
        Any: Return value from function call.
    '''
    output_path = kwargs.get('output_path', None)
    if output_path is None:
        raise ValueError(f'{func.__name__} requires output_path')

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not output_path.exists():
        utils.log(f'INFO: {output_path} missing; Running stage {func.__name__}')
        return True, func(*args, **kwargs)

    utils.log(f'INFO: {output_path} exists; Skipping stage {func.__name__}')
    return False, None

