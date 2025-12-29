import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import *

class FRNet(nn.Module):
    """
    FRNet: Feature Refinement Network for Medical Image Segmentation
    
    Args:
        n_channels: 输入通道数
        n_classes: 输出类别数
        ls_mid_ch: 中间层通道数列表
        out_k_size: 输出卷积核大小
        k_size: 卷积核大小
        block_type: 块类型 ('residual', 'recurrent_convnext')
    """
    def __init__(self, n_channels=1, n_classes=2, ls_mid_ch=None, out_k_size=11, k_size=3,
                 block_type='residual'):
        super().__init__()
        
        # 标准化参数
        self.n_channels = n_channels
        self.n_classes = n_classes
        if ls_mid_ch is None:
            ls_mid_ch = [32] * 4
        self.ls_mid_ch = ls_mid_ch
        self.out_k_size = out_k_size
        self.k_size = k_size
        self.block_type = block_type
        
        # 选择块类型
        if block_type == 'recurrent_convnext':
            cls_init_block = ResidualBlock
            cls_conv_block = RecurrentConvNeXtBlock
        else:  # 默认使用残差块
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

        # 修改最终层以支持可配置的输出类别数
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
        """获取模型信息"""
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


class FRNetLite(FRNet):
    """FRNet轻量级版本"""
    def __init__(self, n_channels=1, n_classes=2, ls_mid_ch=None, out_k_size=7, k_size=3,
                 block_type='residual'):
        if ls_mid_ch is None:
            ls_mid_ch = [16, 16, 16]  # 更少的通道数和层数
        super().__init__(n_channels, n_classes, ls_mid_ch, out_k_size, k_size, block_type)
        
    def get_model_info(self):
        info = super().get_model_info()
        info['model_name'] = 'FRNet-Lite'
        info['description'] = 'FRNet Lite: Lightweight version for fast inference'
        return info


if __name__ == '__main__':
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FRNet(n_channels=1, n_classes=2).to(device)
    
    # 测试前向传播
    x = torch.randn(2, 1, 256, 256).to(device)
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model info: {model.get_model_info()}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")