from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import utils


DEFAULT_KERNEL_SIZE = 3
DEFAULT_RELU_LEAK = 0.01
DEFAULT_NORM_TYPE = 'batch'
DEFAULT_POOL_TYPE = 'max'
DEFAULT_POOL_SIZE = 2
DEFAULT_UPSAMPLE = 'nearest'
DEFAULT_NUM_GROUPS = 8


class ConvUnit3D(torch.nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int=DEFAULT_KERNEL_SIZE,
        relu_leak: float=DEFAULT_RELU_LEAK,
        norm_type: str=DEFAULT_NORM_TYPE,
        num_groups: int=DEFAULT_NUM_GROUPS
    ):
        super().__init__()
        self.conv = torch.nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            padding='same',
            padding_mode='replicate',
            bias=False
        )
        if norm_type == 'batch':
            self.norm = torch.nn.BatchNorm3d(out_channels, affine=True)
        elif norm_type == 'layer':
            self.norm = torch.nn.LayerNorm(out_channels, elementwise_affine=True)
        elif norm_type == 'group':
            self.norm = torch.nn.GroupNorm(num_groups, out_channels, affine=True)
        elif norm_type == 'instance':
            self.norm = torch.nn.InstanceNorm3d(out_channels, affine=True)
        else:
            raise ValueError(f'invalid norm type: {norm_type}')

        self.act = torch.nn.LeakyReLU(negative_slope=relu_leak, inplace=True)

        nn.init.kaiming_normal_(
            self.conv.weight, a=relu_leak, mode='fan_in', nonlinearity='leaky_relu'
        )
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ConvBlock3D(torch.nn.Sequential):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_conv_units: int,
        hid_channels: int=None,
        **conv_unit_kws
    ):
        super().__init__()
        
        if not hid_channels:
            hid_channels = out_channels
        elif n_conv_units < 2:
            raise ValueError('hid_channels argument requires n_conv_units >= 2')

        for i in range(n_conv_units):
            layer = ConvUnit3D(
                in_channels=(hid_channels if i > 0 else in_channels),
                out_channels=(hid_channels if i < n_conv_units - 1 else out_channels),
                **conv_unit_kws
            )
            self.add_module(f'unit{i}', layer)
            

class EncoderBlock(torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        apply_pooling: bool=True,
        pooling_type: str=DEFAULT_POOL_TYPE,
        pooling_size: int=DEFAULT_POOL_SIZE,
        **conv_block_kws
    ):
        super().__init__()

        if not apply_pooling:
            self.pooling = None
        elif pooling_type == 'max':
            self.pooling = torch.nn.MaxPool3d(kernel_size=pooling_size)
        elif pooling_type == 'avg':
            self.pooling = torch.nn.AvgPool3d(kernel_size=pooling_size)
        else:
            raise ValueError(f'invalid pooling_type: {pooling_type}')
        
        self.conv_block = ConvBlock3D(
            in_channels=in_channels,
            out_channels=out_channels,
            **conv_block_kws
        )
        
    def forward(self, x):
        if self.pooling:
            x = self.pooling(x)
        x = self.conv_block(x)
        return x


class DecoderBlock(torch.nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsample_mode: str=DEFAULT_UPSAMPLE,
        scale_factor: int=DEFAULT_POOL_SIZE,
        **conv_block_kws
    ):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=upsample_mode)

        self.conv_block = ConvBlock3D(
            in_channels=in_channels,
            out_channels=out_channels,
            **conv_block_kws
        )

    def forward(self, x, encoder_feats):
        x = self.upsample(x)
        x = torch.cat([x, encoder_feats], dim=1)
        x = self.conv_block(x)
        return x


class UNet3D(torch.nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_channels: int,
        n_enc_blocks: int,
        n_conv_units: int,
        n_init_units: int=None,
        kernel_size: int=DEFAULT_KERNEL_SIZE,
        relu_leak: float=DEFAULT_RELU_LEAK,
        norm_type: str=DEFAULT_NORM_TYPE,
        num_groups: int=DEFAULT_NUM_GROUPS,
        pooling_type: str=DEFAULT_POOL_TYPE,
        upsample_mode: str=DEFAULT_UPSAMPLE,
        output_func: str='relu'
    ):
        super().__init__()
        assert n_enc_blocks > 0
        
        self.encoder = torch.nn.Sequential()
        curr_channels = in_channels
        next_channels = conv_channels

        for i in range(n_enc_blocks):
            enc_block = EncoderBlock(
                in_channels=curr_channels,
                out_channels=next_channels,
                n_conv_units=(n_init_units if i == 0 and n_init_units else n_conv_units),
                kernel_size=kernel_size,
                relu_leak=relu_leak,
                norm_type=norm_type,
                num_groups=num_groups,
                apply_pooling=(i > 0),
                pooling_type=pooling_type
            )
            self.encoder.add_module(f'level{i}', enc_block)
            curr_channels = next_channels
            next_channels = curr_channels * 2
        
        self.decoder = torch.nn.Sequential()
        next_channels = curr_channels // 2
        
        for i in reversed(range(n_enc_blocks - 1)):
            dec_block = DecoderBlock(
                in_channels=curr_channels + next_channels,
                out_channels=next_channels,
                n_conv_units=n_conv_units,
                kernel_size=kernel_size,
                relu_leak=relu_leak,
                norm_type=norm_type,
                num_groups=num_groups,
                upsample_mode=upsample_mode
            )
            self.decoder.add_module(f'level{i}', dec_block)
            curr_channels = next_channels
            next_channels = curr_channels // 2
        
        self.output_conv = torch.nn.Conv3d(curr_channels, out_channels, kernel_size=1)
        self.output_func = _get_output_fn(output_func)

    def forward(self, x):
        features = []
        for i, enc_block in enumerate(self.encoder):
            x = enc_block(x)
            features.append(x)
        
        # reverse order to align with decoder
        features = features[::-1]
        for i, dec_block in enumerate(self.decoder):
            x = dec_block(x, features[i+1])

        theta = self.output_conv(x)
        return self.output_func(theta)


def _get_output_fn(name):
    if name == 'relu':
        return F.relu
    elif name == 'softplus':
        return F.softplus
    elif name == 'exp':
        return torch.exp
    elif name == 'pow10':
        return lambda x: torch.pow(10, x)
    else:
        raise ValueError(f'invalid name: {name}')


def count_params(model):
    total = 0
    for name, p in model.named_parameters():
        shape = tuple(p.shape)
        prod = int(np.prod(shape))
        utils.log(f'{name:50s} {prod}\t{shape}')
        total += prod
    return total


def count_activations(model, input_, ret_output=False):
    hooks, recorded = [], []
    seen = set()

    def describe(t):
        return tuple(t.shape), t.dtype, t.mean().item(), t.std().item()

    def record_output_description(name):
        def hook(m, in_, out):
            key = id(out)
            if key not in seen:
                recorded.append((name, describe(out)))
                seen.add(key)
        return hook

    for name, m in model.named_modules():
        hook = m.register_forward_hook(record_output_description(name))
        hooks.append(hook)

    model.eval()
    with torch.no_grad():
        recorded.append(('input', describe(input_)))
        output = model.forward(input_)

    for h in hooks:
        h.remove()

    total = 0
    for name, (shape, dtype, mean, std) in recorded:
        prod = int(np.prod(shape))
        utils.log(f'{name:40s} {prod}\t{shape}\t{dtype}\t{mean:.4f}\t{std:.4f}')
        total += prod

    return (total, output) if ret_output else total

