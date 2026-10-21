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
        kernel_size: int = 5,
        use_bias: bool = True,
        **spec_kws
    ):
        super().__init__()

        from . import param_spec

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            padding='same',
            padding_mode='replicate',
            bias=use_bias
        )

        self.spec = param_spec.ParameterSpec(**spec_kws)

        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(
            self.conv.weight,
            nonlinearity='linear',
            mode='fan_in'
        )
        with torch.no_grad():
            self.conv.weight.mul_(0.01)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.spec.decode(self.conv(x))


# ----- backbone architecture -----


class UNet3D(torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_unet_levels: int,
        downsample_fn: str = DEFAULT_DOWNSAMPLE_FN,
        upsample_mode: str = DEFAULT_UPSAMPLE_MODE,
        do_downsample: bool = False,
        **kwargs
    ):
        super().__init__()

        if n_unet_levels < 0:
            raise ValueError('n_unet_levels must be non-negative')

        self.encoder = ConvBlock3D(in_channels, out_channels, **kwargs)
        curr_channels = out_channels

        if n_unet_levels > 0:
            self.sublevel = UNet3D(
                in_channels=curr_channels,
                out_channels=out_channels * 2,
                n_unet_levels=n_unet_levels - 1,
                downsample_fn=downsample_fn,
                upsample_mode=upsample_mode,
                do_downsample=True,
                **kwargs
            )
            curr_channels += self.sublevel.out_channels

        self.decoder = ConvBlock3D(curr_channels, out_channels, **kwargs)

        self.downsample_fn = {'max': F.max_pool3d, 'avg': F.avg_pool3d}[downsample_fn]
        self.upsample_mode = upsample_mode
        self.do_downsample = do_downsample

        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        in_shape = x.shape[2:]

        if self.do_downsample:
            x = self.downsample_fn(x, kernel_size=2)

        z = self.encoder(x)

        if hasattr(self, 'sublevel'):
            z = torch.cat([z, self.sublevel(z)], dim=1)

        y = self.decoder(z)

        if self.do_downsample:
            return F.interpolate(y, in_shape, mode=self.upsample_mode)

        return y


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

        elif norm_type is None:
            self.norm = torch.nn.Identity()

        else:
            raise ValueError(f'Invalid norm type: {norm_type}')

        self.act = torch.nn.LeakyReLU(negative_slope=relu_leak, inplace=True)
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(
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
    import weakref

    hooks, recorded = [], []
    seen = {}
    modes = [(m, m.training) for m in model.modules()]

    def record(name, value):
        if isinstance(value, dict):
            for key, item in value.items():
                record(f'{name}.{key}', item)

        elif isinstance(value, (tuple, list)):
            for idx, item in enumerate(value):
                record(f'{name}[{idx}]', item)

        elif isinstance(value, torch.Tensor):
            key = id(value)
            previous = seen.get(key)

            if previous is not None and previous() is value:
                return

            seen[key] = weakref.ref(value)
            recorded.append((
                name,
                tuple(value.shape),
                value.dtype,
                value.numel(),
                value.numel() * value.element_size(),
            ))

    def make_hook(name):
        def hook(mod, inp, out):
            record(name, out)
        return hook

    try:
        for name, module in model.named_modules():
            hooks.append(
                module.register_forward_hook(make_hook(name or 'model'))
            )

        model.eval()
        with torch.no_grad():
            record('input', input_)
            output = model(input_)

    finally:
        for hook in hooks:
            hook.remove()

        for module, training in modes:
            module.training = training

    mib = 1024 ** 2
    total = sum(row[3] for row in recorded)
    total_bytes = sum(row[4] for row in recorded)

    for name, shape, dtype, numel, nbytes in recorded:
        utils.log(
            f'{name:55s} {nbytes / mib:9.2f} MiB  '
            f'{str(shape):28s} {dtype}'
        )

    utils.log('\nLargest outputs:')
    for name, shape, dtype, numel, nbytes in sorted(
        recorded, key=lambda row: row[4], reverse=True
    )[:15]:
        utils.log(f'{name:55s} {nbytes / mib:9.2f} MiB')

    utils.log(
        f'\nTotal recorded elements: {total:,}\n'
        f'Total logical output size: {total_bytes / mib:,.2f} MiB'
    )

    return (total, output) if ret_output else total

