# models.py

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import utils


DEFAULT_KERNEL_SIZE = 3
DEFAULT_RELU_LEAK = 0.1
DEFAULT_NORM_TYPE = 'group'
DEFAULT_NUM_GROUPS = 4
DEFAULT_DOWNSAMPLE_FN = 'max'
DEFAULT_UPSAMPLE_MODE = 'trilinear'


def build_model(task, config):
    utils.check_keys(config, {'backbone', 'heads'})

    utils.log('Building model architecture')

    backbone_kws = config.get('backbone', {})
    backbone = build_backbone(task.in_channels, **backbone_kws)

    heads_config = config.get('heads', {})
    unused_heads = set(heads_config) - set(task.targets)
    if unused_heads:
        raise ValueError(f'Unused head config(s): {unused_heads}')

    heads = {}
    for target in task.targets:
        heads[target] = OutputHead(
            in_channels=backbone.out_channels,
            out_channels=task.out_channels(target),
            **heads_config.get(target, {})
        )

    return MultiTaskModel(backbone, heads)


def build_backbone(in_channels, type, **kwargs):
    return globals()[type](in_channels, **kwargs)


class MultiTaskModel(nn.Module):

    def __init__(self, backbone, heads):
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleDict(heads)

    def forward(self, x):
        feats = self.backbone(x)
        return {name: head(feats) for name, head in self.heads.items()}


class OutputHead(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_bias: bool = True,
        **spec_kws
    ):
        super().__init__()

        from . import param_spec

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=use_bias
        )
        self.spec = param_spec.ParameterSpec(**spec_kws)

    def forward(self, x):
        return self.spec.decode(self.conv(x))


# ----- backbone architecture -----


class UNet3Dv2(torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        conv_channels: int,
        n_sub_levels: int,
        downsample_fn: str = DEFAULT_DOWNSAMPLE_FN,
        upsample_mode: str = DEFAULT_UPSAMPLE_MODE,
        **kwargs
    ):
        super().__init__()

        if n_sub_levels < 0:
            raise ValueError('n_sub_levels must be non-negative')

        self.enc = ConvBlock3D(in_channels, conv_channels, **kwargs)
        curr_channels = conv_channels

        if n_sub_levels > 0:
            self.net = UNet3Dv2(
                in_channels=curr_channels,
                conv_channels=conv_channels * 2,
                n_sub_levels=n_sub_levels - 1,
                downsample_fn=downsample_fn,
                upsample_mode=upsample_mode,
                **kwargs
            )
            curr_channels += self.net.out_channels

        self.dec = ConvBlock3D(curr_channels, conv_channels, **kwargs)
        self.out_channels = conv_channels

        self.downsample_fn = {'max': F.max_pool3d, 'avg': F.avg_pool3d}[downsample_fn]
        self.upsample_mode = upsample_mode

    def forward(self, x):
        z = self.enc(x)
        if hasattr(self, 'net'):
            y = self.downsample_fn(z, kernel_size=2)
            y = self.net(y)
            y = F.interpolate(y, z.shape[2:], mode=self.upsample_mode)
            z = torch.cat([z, y], dim=1)
        return self.dec(z)


class ConvBlock3D(torch.nn.Sequential):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_conv_units: int,
        **kwargs
    ):
        super().__init__()

        if n_conv_units < 1:
            raise ValueError('n_conv_units must be positive')

        for i in range(n_conv_units):
            unit = ConvUnit3D(in_channels, out_channels, **kwargs)
            self.add_module(f'unit{i}', unit)
            in_channels = out_channels


class ConvUnit3D(torch.nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
        relu_leak: float = DEFAULT_RELU_LEAK,
        norm_type: str = DEFAULT_NORM_TYPE,
        num_groups: int = DEFAULT_NUM_GROUPS
    ):
        super().__init__()

        self.conv = torch.nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            padding='same',
            padding_mode='replicate',
            bias=norm_type is None
        )

        if norm_type == 'batch':
            self.norm = torch.nn.BatchNorm3d(out_channels, affine=True)

        elif norm_type == 'layer':
            self.norm = torch.nn.GroupNorm(1, out_channels, affine=True)

        elif norm_type == 'group':
            assert out_channels % num_groups == 0, out_channels
            self.norm = torch.nn.GroupNorm(num_groups, out_channels, affine=True)

        elif norm_type == 'instance':
            self.norm = torch.nn.InstanceNorm3d(out_channels, affine=True)

        elif norm_type is None: # no-op
            self.norm = torch.nn.Identity()

        else:
            raise ValueError(f'Invalid norm type: {norm_type}')

        self.act = torch.nn.LeakyReLU(negative_slope=relu_leak, inplace=True)
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(
            self.conv.weight,
            a=self.act.negative_slope,
            nonlinearity='leaky_relu',
            mode='fan_in'
        )
        if hasattr(self.norm, 'weight'):
            nn.init.ones_(self.norm.weight)
            nn.init.zeros_(self.norm.bias)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


def identity(x):
    return x


def pow10(x):
    return torch.pow(10., x)


def get_output_fn(name):
    if name == 'identity':
        return identity
    elif name == 'relu':
        return F.relu
    elif name == 'softplus':
        return F.softplus
    elif name == 'exp':
        return torch.exp
    elif name == 'pow10':
        return pow10
    raise ValueError(f'Unknown function name: {name}')


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
        def hook(mod, inp, out):
            if isinstance(out, dict):
                for k, v in out.items():
                    key = id(v)
                    if key not in seen:
                        recorded.append((name + '.' + k, describe(v)))
                        seen.add(key)
            else:
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

