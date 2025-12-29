import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import *

class FRNet(nn.Module):
    """
    FRNet: Feature Refinement Network for Medical Image Segmentation
    """
    def __init__(self, n_channels=1, n_classes=2, ls_mid_ch=None, out_k_size=11, k_size=3,
                 block_type='residual'):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        if ls_mid_ch is None:
            ls_mid_ch = [32] * 4
        self.ls_mid_ch = ls_mid_ch
        self.out_k_size = out_k_size
        self.k_size = k_size
        self.block_type = block_type

        if block_type == 'recurrent_convnext':
            cls_init_block = ResidualBlock
            cls_conv_block = RecurrentConvNeXtBlock
        else:  
            cls_init_block = ResidualBlock
            cls_conv_block = ResidualBlock
        
        self.dict_module = nn.ModuleDict()
        ch1 = n_channels
        for i in range(len(ls_mid_ch)):
            ch2 = ls_mid_ch[i]
            if ch1 != ch2:
                self.dict_module.add_module(f"conv{i}", cls_init_block(ch1, ch2, k_size=k_size, stride=1, dilation=1))
            else:
                if cls_conv_block == RecurrentConvNeXtBlock:
                    module = RecurrentConvNeXtBlock(dim=ch1, layer_scale_init_value=1)
                else:
                    module = cls_conv_block(ch1, ch2, k_size=k_size)
                self.dict_module.add_module(f"conv{i}", module)

            ch1 = ch2

        self.dict_module['final'] = nn.Sequential(
            nn.Conv2d(ch1, n_classes * 4, out_k_size, padding=out_k_size // 2, bias=False),
            nn.BatchNorm2d(n_classes * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_classes * 4, n_classes, 1, bias=False)
        )

    def forward(self, x):
        for i in range(len(self.ls_mid_ch)):
            conv = self.dict_module[f'conv{i}']
            x = conv(x)

        x = self.dict_module['final'](x)
        return x
    
    def get_model_info(self):
        return {
            'model_name': 'FRNet',
            'input_channels': self.n_channels,
            'output_channels': self.n_classes,
            'mid_channels': self.ls_mid_ch,
            'output_kernel_size': self.out_k_size,
            'kernel_size': self.k_size,
            'block_type': self.block_type,
            'description': 'FRNet: Feature Refinement Network for Medical Image Segmentation'
        }


