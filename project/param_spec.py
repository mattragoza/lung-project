# param_spec.py

from typing import Optional

import torch
import torch.nn.functional as F


def get_param_spec(**kwargs):
    return ParameterSpec(**kwargs)


class ParameterSpec:

    def __init__(
        self,
        mode: str = 'linear',
        scale: float = 1.0,
        v_loc: float = 0.0,
        v_min: Optional[float] = None,
        v_max: Optional[float] = None,
        beta: Optional[float] = None
    ):
        if mode not in {'linear', 'log10'}:
            raise ValueError(f'Invalid parameter mode: {mode!r}')

        if scale <= 0:
            raise ValueError(f'Parameter scale must be positive')

        self.mode  = mode
        self.scale = scale # NOTE: this also scales the gradient!
        self.v_loc = v_loc
        self.v_min = v_min
        self.v_max = v_max
        self.beta  = beta

        if mode == 'log10' and v_min == 0:
            self.s_min = None
        else:
            self.s_min = _invert_transform(v_min, mode).item()

        self.s_max = _invert_transform(v_max, mode).item()
        self.s_loc = _invert_transform(v_loc, mode).item()

        self.shift = _invert_bounds(
            self.s_loc, self.s_min, self.s_max, beta
        ).item()

    def decode(self, z):
        '''Decode latent coordinate to physical parameter.'''
        q = _apply_affine(z, self.shift, self.scale)
        s = _apply_bounds(q, self.s_min, self.s_max, self.beta)
        return _apply_transform(s, self.mode)

    def encode(self, v):
        '''Encode physical parameter to latent coordinate.'''
        s = _invert_transform(v, self.mode)
        q = _invert_bounds(s, self.s_min, self.s_max, self.beta)
        return _invert_affine(q, self.shift, self.scale)


def _apply_transform(s, mode):
    if s is None:
        return None
    s = torch.as_tensor(s)
    if mode == 'log10':
        return torch.pow(10, s)
    return s


def _invert_transform(v, mode):
    if v is None:
        return None
    v = torch.as_tensor(v)
    if mode == 'log10':
        return torch.log10(v)
    return v


def _apply_affine(z, shift, scale):
    return shift + scale * z


def _invert_affine(q, shift, scale):
    return (q - shift) / scale


def _apply_bounds(q, s_min, s_max, beta):
    q = torch.as_tensor(q)

    if s_min is None and s_max is None:
        return q

    if beta is None or beta <= 0: # hard
        return torch.clamp(q, s_min, s_max)

    if s_min is None: # soft upper bound
        return s_max - F.softplus(s_max - q, beta)

    if s_max is None: # soft lower bound
        return s_min + F.softplus(q - s_min, beta)

    # soft two-sided bounds
    return (
        s_min
        + F.softplus(q - s_min, beta)
        - F.softplus(q - s_max, beta)
    )


def _invert_bounds(s, s_min, s_max, beta):
    s = torch.as_tensor(s)

    if s_min is None and s_max is None:
        return s

    _check_bounds(s, s_min, s_max)

    if beta is None or beta <= 0: # hard
        return s

    if s_min is None: # soft upper bound
        return s_max - _invert_softplus(s_max - s, beta)

    if s_max is None: # soft lower bound
        return s_min + _invert_softplus(s - s_min, beta)

    # soft two-sided bounds
    return s + (
        torch.log(-torch.expm1(-beta * (s - s_min)))
        - torch.log(-torch.expm1(-beta * (s_max - s)))
    ) / beta


def _invert_softplus(s, beta):
    return s + torch.log(-torch.expm1(-beta * s)) / beta


def _check_bounds(s, s_min, s_max):
    s = torch.as_tensor(s)
    if s_min is not None and torch.any(s < s_min):
        raise ValueError(f'Out of bounds: {s.min():f} < {s_min}')
    if s_max is not None and torch.any(s > s_max):
        raise ValueError(f'Out of bounds: {s.max():f} > {s_max}')

