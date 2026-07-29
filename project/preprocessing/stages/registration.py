# preprocessing/stages/registration.py

from typing import List, Dict, Tuple, Any
from pathlib import Path

from ...core import utils, fileio


def estimate_displacement_field(
    fixed_image: Path,
    moving_image: Path,
    fixed_mask: Path,
    moving_mask: Path,
    output_path: Path,
    config: Dict[str, Any]
):
    utils.check_keys(
        config,
        valid={'method', 'kwargs'},
        where='image_registration'
    )
    from .. import registration

    method = config.get('method', 'corrfield').lower()
    kwargs = config.get('kwargs', {})

    if method == 'corrfield':
        utils.log('Running CorrField registration')

        registration.run_corrfield_registration(
            fixed_image=fixed_image,
            moving_image=moving_image,
            fixed_mask=fixed_mask,
            output_path=output_path,
            **kwargs
        )

    elif method == 'unigradicon':
        utils.log('Running uniGradICON registration')
    
        registration.run_unigradicon_registration(
            fixed_image=fixed_image,
            moving_image=moving_image,
            fixed_mask=fixed_mask,
            moving_mask=moving_mask,
            output_path=output_path,
            **kwargs
        )

    else:
        raise ValueError(f'Invalid registration method: {method!r}')

