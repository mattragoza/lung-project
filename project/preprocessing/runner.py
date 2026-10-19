# preprocessing/runner.py

from typing import List, Dict, Tuple, Optional, Any

from pathlib import Path

from ..common import utils, fileio


def run_stage(
    func, output_path, validate_fn=None, **kwargs
) -> Tuple[bool, Any]:
    '''
    Call function if output path does not exist.

    Args:
        func: Function to call.
        output_path: Function output path.
        validate_fn: Check for valid output.
        **kwargs: Passed into function call.
    Returns:
        bool: True if the function was called.
        Any: Return value from function call.
    '''
    if output_path is None:
        raise ValueError(f'{func.__name__} requires output_path')

    output_path = Path(output_path)
    valid_output = output_path.exists()

    if valid_output and validate_fn is not None:
        try:
            validate_fn(output_path)
        except (OSError, ValueError, EOFError) as exc:
            utils.warn(
                f'WARNING: Invalid output for {func.__name__}: '
                f'{output_path}: {exc}'
            )
            valid_output = False

    if valid_output:
        utils.log(f'INFO: Skipping stage {func.__name__}')
        return False, None

    utils.log(f'INFO: Running stage {func.__name__}')
    result = func(*args, output_path=output_path, **kwargs)

    if validate_fn is not None:
        validate_fn(output_path)

    return True, result

