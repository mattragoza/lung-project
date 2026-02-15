from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import utils

DEFAULT_KERNEL_SIZE = 3
DEFAULT_RELU_LEAK = 0.01
DEFAULT_NORM_TYPE = 'group'
DEFAULT_NUM_GROUPS = 8
DEFAULT_POOL_TYPE = 'max'
DEFAULT_POOL_SIZE = 2
DEFAULT_UPSAMPLE = 'nearest'


def build_model(task, config):
    utils.check_keys(config, {'backbone', 'heads'})

    backbone_kws = config.get('backbone', {}).copy()
    backbone_cls = globals()[backbone_kws.pop('_class')]
    backbone = backbone_cls(task.in_channels, **backbone_kws)

    heads_cfg = config.get('heads', {})
    heads = {}

    for tgt in task.targets:
        out_channels = task.out_channels(tgt)

        if tgt in {'E', 'logE'}:
            head_kws = heads_cfg.get('elasticity', {})
            if head_kws.get('use_probs'):
                in_channels = task.out_channels('material')
            else:
                in_channels = backbone.out_channels

            head = ElasticityHead(in_channels, out_channels, **head_kws)
            heads['elasticity'] = head

        elif tgt == 'material':
            head_kws = heads_cfg.get(tgt, {})
            head = SegmentationHead(backbone.out_channels, out_channels, **head_kws)
            heads[tgt] = head

        else:
            head_kws = heads_cfg.get(tgt, {})
            head = RegressionHead(backbone.out_channels, out_channels, **head_kws)
            heads[tgt] = head

    return MultiTaskModel(backbone, heads)


class MultiTaskModel(nn.Module):

    def __init__(self, backbone, heads):
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleDict(heads)

    def forward(self, x):
        feats = self.backbone(x)

        outputs = {'feats': feats}
        for name, head in self.heads.items():
            outputs.update(head(outputs))

        return outputs


class ElasticityHead(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels=1,
        logE_mean=0.0,
        logE_std=1.0,
        logE_min=None,
        logE_max=None,
        use_bias=True,
        use_probs=False
    ):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=use_bias)
    
        self.logE_mean = logE_mean
        self.logE_std  = logE_std
        self.logE_min  = logE_min
        self.logE_max  = logE_max

        self.use_bias  = use_bias
        self.use_probs = use_probs

    def forward(self, inputs):

        if self.use_probs:
            z = self.conv(inputs['mat_probs'])
        else:
            z = self.conv(inputs['feats'])

        logE = torch.clamp(
            z * self.logE_std + self.logE_mean,
            min=self.logE_min,
            max=self.logE_max
        )
        E = torch.pow(10.0, logE)
        return {'E_pred': E, 'logE_pred': logE, 'z': z}


class SegmentationHead(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, inputs):
        logits = self.conv(inputs['feats'])
        probs = F.softmax(logits, dim=1)
        return {'mat_logits': logits, 'mat_probs': probs}


class RegressionHead(nn.Module):

    def __init__(self, in_channels, out_channels, out_key='img_pred'):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.out_key = out_key

    def forward(self, inputs):
        return {self.out_key: self.conv(inputs['feats'])}


# ----- architecture components -----


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
            self.norm = torch.nn.GroupNorm(1, out_channels, affine=True)
        elif norm_type == 'group':
            if out_channels >= num_groups:
                self.norm = torch.nn.GroupNorm(num_groups, out_channels, affine=True)
            else:
                self.norm = torch.nn.Identity()
        elif norm_type == 'instance':
            self.norm = torch.nn.InstanceNorm3d(out_channels, affine=True)
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

        if hid_channels is None:
            hid_channels = max(in_channels, out_channels)
        elif n_conv_units < 2:
            raise ValueError('hid_channels argument requires n_conv_units >= 2')

        for i in range(n_conv_units):
            layer = ConvUnit3D(
                in_channels=(hid_channels if i > 0 else in_channels),
                out_channels=(hid_channels if i < n_conv_units - 1 else out_channels),
                **conv_unit_kws
            )
            self.add_module(f'unit{i}', layer)


class UNet3Dv2(torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_channels: int,
        n_conv_units: int,
        n_sub_levels: int,
        pooling_type: str=DEFAULT_POOL_TYPE,
        upsample_mode: str=DEFAULT_UPSAMPLE,
        **kwargs
    ):
        assert n_sub_levels >= 0
        super().__init__()

        self.enc = ConvBlock3D(
            in_channels=in_channels,
            out_channels=conv_channels,
            n_conv_units=n_conv_units,
            **kwargs
        )
        curr_channels = conv_channels

        if n_sub_levels > 0:
            downsample = {'max': nn.MaxPool3d, 'avg': nn.AvgPool3d}[pooling_type]
            self.down = downsample(kernel_size=2)
            self.sub = type(self)(
                in_channels=conv_channels,
                out_channels=conv_channels,
                conv_channels=conv_channels * 2,
                n_conv_units=n_conv_units,
                n_sub_levels=n_sub_levels - 1,
                pooling_type=pooling_type,
                upsample_mode=upsample_mode,
                **kwargs
            )
            self.up = nn.Upsample(scale_factor=2, mode=upsample_mode)
            curr_channels += conv_channels

        self.dec = ConvBlock3D(
            in_channels=curr_channels,
            hid_channels=conv_channels if n_conv_units > 1 else None,
            out_channels=out_channels,
            n_conv_units=n_conv_units,
            **kwargs
        )
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        z = self.enc(x)
        if hasattr(self, 'sub'):
            y = self.down(z)
            y = self.sub(y)
            y = self.up(y)
            z = torch.cat([z, y], dim=1)
        return self.dec(z)


# ----- UNetv1 components -----
            

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
            raise ValueError(f'Invalid pooling_type: {pooling_type}')
        
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
        conv_channels: int,
        n_enc_blocks: int,
        n_conv_units: int,
        n_init_units: int=None,
        kernel_size: int=DEFAULT_KERNEL_SIZE,
        relu_leak: float=DEFAULT_RELU_LEAK,
        norm_type: str=DEFAULT_NORM_TYPE,
        num_groups: int=DEFAULT_NUM_GROUPS,
        pooling_type: str=DEFAULT_POOL_TYPE,
        upsample_mode: str=DEFAULT_UPSAMPLE
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
        
        self.in_channels = in_channels
        self.out_channels = curr_channels

    def forward(self, x):

        features = []
        for i, enc_block in enumerate(self.encoder):
            x = enc_block(x)
            features.append(x)
        
        # reverse order to align with decoder
        features = features[::-1]
        for i, dec_block in enumerate(self.decoder):
            x = dec_block(x, features[i+1])

        return x


# consider deprecating past this point


class ParameterMap(nn.Module):
    '''
    Maps an unconstrained log-space parameter to Young's modulus (Pa).

    Args:
        bounds_mode: 'hard' | 'soft' | 'none'
        lower_bound: float, in param_space
        upper_bound: float, in param_space
        beta: sharpness for soft clamping
    '''
    def __init__(
        self,
        param_space: str,
        bounds_mode: str,
        lower_bound: float,
        upper_bound: float,
        beta: float=10.0
    ):
        super().__init__()
        assert bounds_mode in {'soft', 'hard', 'none'}
        self.param_space = str(param_space)
        self.bounds_mode = str(bounds_mode)
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)
        self.beta = float(beta)

    def forward(self, x):
        if self.bounds_mode == 'hard':
            x = torch.clamp(x, self.lower_bound, self.upper_bound)
        elif self.bounds_mode == 'soft':
            x = soft_clamp(x, self.lower_bound, self.upper_bound, beta=self.beta)
        return torch.pow(10.0, x)



def soft_clamp(x, lo, hi, beta=10.0):
    return lo + F.softplus(x - lo, beta=beta) - F.softplus(x - hi, beta=beta)


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

