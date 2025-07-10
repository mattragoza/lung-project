import torch
import torch.nn as nn
import torch.nn.functional as F


def get_func_by_name(name):
    if name == 'id':
        return lambda x: x
    elif name == 'exp':
        return torch.exp
    elif name == 'relu':
        return F.relu
    elif name == 'softplus':
        return F.softplus



class ConvUnit(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.norm = torch.nn.BatchNorm3d(in_channels)
        self.conv = torch.nn.Conv3d(in_channels, out_channels, kernel_size, padding='same', padding_mode='replicate')
        self.relu = torch.nn.LeakyReLU(inplace=True)
        
    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        x = self.relu(x)
        return x


class ConvBlock(torch.nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size, num_conv_layers, hidden_channels=None):
        super().__init__()
        
        if not hidden_channels:
            hidden_channels = out_channels
        elif num_conv_layers < 2:
            print('Warning: hidden_channels argument only used if num_conv_layers >= 2')

        for i in range(num_conv_layers):
            layer = ConvUnit(
                in_channels=(hidden_channels if i > 0 else in_channels),
                out_channels=(hidden_channels if i < num_conv_layers - 1 else out_channels),
                kernel_size=kernel_size
            )
            self.add_module(f'conv_unit{i}', layer)
            

class Upsample(torch.nn.Module):
    
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        
    def __repr__(self):
        return f'{type(self).__name__}(mode={self.mode})'
        
    def forward(self, x, size):
        return F.interpolate(x, size=size, mode=self.mode)


class EncoderBlock(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        conv_kernel_size,
        num_conv_layers,
        hidden_channels=None,
        apply_pooling=True,
        pool_kernel_size=2,
        pool_type='max'
    ):
        super().__init__()
        assert pool_type in {'max', 'avg'}

        if apply_pooling:
            if pool_type == 'max':
                self.pooling = torch.nn.MaxPool3d(kernel_size=pool_kernel_size)
            elif pool_type == 'avg':
                self.pooling = torch.nn.AvgPool3d(kernel_size=pool_kernel_size)
        else:
            self.pooling = None
            
        self.conv_block = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=conv_kernel_size,
            num_conv_layers=num_conv_layers,
            hidden_channels=hidden_channels
        )
        
    def forward(self, x):
        if self.pooling:
            x = self.pooling(x)
        x = self.conv_block(x)
        return x


class DecoderBlock(torch.nn.Module):
    
    def __init__(
        self,
        in_channels,
        out_channels,
        conv_kernel_size,
        num_conv_layers,
        hidden_channels=None,
        upsample_mode='nearest'
    ):
        super().__init__()

        self.upsample = Upsample(mode=upsample_mode)

        self.conv_block = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=conv_kernel_size,
            num_conv_layers=num_conv_layers,
            hidden_channels=hidden_channels,
        )

    def forward(self, x, encoder_feats):
        x = self.upsample(x, size=encoder_feats.shape[2:])
        x = torch.cat([x, encoder_feats], dim=1)
        x = self.conv_block(x)
        return x


class UNet3D(torch.nn.Module):
    
    def __init__(
        self,
        in_channels,
        out_channels,
        num_levels,
        num_conv_layers,
        conv_channels,
        conv_kernel_size,
        pool_kernel_size=2,
        pool_type='max',
        upsample_mode='trilinear',
        output_func='relu'
    ):
        super().__init__()
        assert num_levels > 0
        
        curr_channels = in_channels
        next_channels = conv_channels
        
        self.encoder = torch.nn.Sequential()
        for i in range(num_levels):
        
            encoder_block = EncoderBlock(
                in_channels=curr_channels,
                out_channels=next_channels,
                conv_kernel_size=conv_kernel_size,
                num_conv_layers=num_conv_layers,
                apply_pooling=(i > 0),
                pool_kernel_size=pool_kernel_size,
                pool_type=pool_type
            )
            self.encoder.add_module(f'level{i}', encoder_block)

            curr_channels = next_channels
            next_channels = curr_channels * 2
        
        next_channels = curr_channels // 2
        
        self.decoder = torch.nn.Sequential()
        for i in reversed(range(num_levels - 1)):

            decoder_block = DecoderBlock(
                in_channels=curr_channels + next_channels,
                out_channels=next_channels,
                conv_kernel_size=conv_kernel_size,
                num_conv_layers=num_conv_layers,
                upsample_mode=upsample_mode
            )
            self.decoder.add_module(f'level{i}', decoder_block)
            
            curr_channels = next_channels
            next_channels = curr_channels // 2
        
        self.final_conv = torch.nn.Conv3d(curr_channels, out_channels, kernel_size=1)
        self.output_func = get_func_by_name(output_func)

    def forward(self, x):
        
        # encoder part
        encoder_feats = []
        for i, encoder in enumerate(self.encoder):
            x = encoder(x)
            encoder_feats.append(x)
        
        # reverse encoder features to align with decoder
        encoder_feats = encoder_feats[::-1]
    
        # decoder part
        for i, decoder in enumerate(self.decoder):
            x = decoder(x, encoder_feats[i+1])

        return self.output_func(self.final_conv(x))


class ParameterMap(torch.nn.Module):
    '''
    This class is used for directly optimizing an
    elasticity parameter map, instead of training
    a neural network to map from image to elasticity.
    '''
    def __init__(self, shape, upsample_mode, conv_kernel_size, output_func):
        super().__init__()
        assert len(shape) == 4
        self.params = torch.nn.Parameter(torch.randn(shape))
        self.upsample = Upsample(mode=upsample_mode)
        self.final_conv = torch.nn.Conv3d(
            in_channels=shape[0],
            out_channels=shape[0],
            kernel_size=conv_kernel_size,
            padding='same',
            padding_mode='replicate'
        )
        self.output_func = get_func_by_name(output_func)

    def forward(self, x):
        p = self.params.unsqueeze(0)
        p = self.upsample(p, size=x.shape[2:])
        return self.output_func(self.final_conv(p))
