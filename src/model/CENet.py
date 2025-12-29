"""
CENet: Context Encoder Network for 2D Medical Image Segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_components import DoubleConv, AttentionGate


class DACBlock(nn.Module):
    """Dense Atrous Convolution Block """
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=3, dilation=3)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=5, dilation=5)
        self.conv4 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=7, dilation=7)
        
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.bn4 = nn.BatchNorm2d(in_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.fusion = nn.Conv2d(in_channels * 4, in_channels, kernel_size=1)
        self.fusion_bn = nn.BatchNorm2d(in_channels)
        
    def forward(self, x):
        d1 = self.relu(self.bn1(self.conv1(x)))
        d2 = self.relu(self.bn2(self.conv2(x)))
        d3 = self.relu(self.bn3(self.conv3(x)))
        d4 = self.relu(self.bn4(self.conv4(x)))

        concat = torch.cat([d1, d2, d3, d4], dim=1)
        fused = self.relu(self.fusion_bn(self.fusion(concat)))

        return x + fused


class RMPBlock(nn.Module):
    """Residual Multi-kernel Pooling Block - 残差多核池化块"""
    def __init__(self, in_channels):
        super().__init__()
        self.pool1 = nn.AdaptiveMaxPool2d(1)
        self.pool2 = nn.AdaptiveMaxPool2d(2)
        self.pool3 = nn.AdaptiveMaxPool2d(3)
        self.pool4 = nn.AdaptiveMaxPool2d(6)

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
        self.conv4 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)

        self.fusion = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        b, c, h, w = x.size()
        
        p1 = F.interpolate(self.conv1(self.pool1(x)), size=(h, w), mode='bilinear', align_corners=False)
        p2 = F.interpolate(self.conv2(self.pool2(x)), size=(h, w), mode='bilinear', align_corners=False)
        p3 = F.interpolate(self.conv3(self.pool3(x)), size=(h, w), mode='bilinear', align_corners=False)
        p4 = F.interpolate(self.conv4(self.pool4(x)), size=(h, w), mode='bilinear', align_corners=False)

        concat = torch.cat([p1, p2, p3, p4], dim=1)
        fused = self.relu(self.bn(self.fusion(concat)))

        return x + fused


class ContextBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.dac = DACBlock(in_channels)
        self.rmp = RMPBlock(in_channels)
        
    def forward(self, x):
        x = self.dac(x)
        x = self.rmp(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_context=False):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.use_context = use_context
        
        if use_context:
            self.context = ContextBlock(out_channels)
        
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        x = self.conv(x)
        
        if self.use_context:
            x = self.context(x)
            
        features = x
        x = self.pool(x)
        
        return x, features


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_attention=True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionGate(in_channels // 2, skip_channels, skip_channels // 2)
        
        self.conv = DoubleConv((in_channels // 2) + skip_channels, out_channels)
        
    def forward(self, x, skip=None):
        x = self.up(x)
        
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            
            if self.use_attention:
                skip = self.attention(x, skip)
            
            x = torch.cat([x, skip], dim=1)
        
        x = self.conv(x)
        return x


class CENet(nn.Module):
    """
    CENet: Context Encoder Network for 2D Medical Image Segmentation
    """
    def __init__(self, n_channels=1, n_classes=2, base_channels=64, use_attention=True, deep_supervision=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.use_attention = use_attention
        self.deep_supervision = deep_supervision
        
        self.encoder1 = EncoderBlock(n_channels, base_channels, use_context=False)
        self.encoder2 = EncoderBlock(base_channels, base_channels * 2, use_context=False)
        self.encoder3 = EncoderBlock(base_channels * 2, base_channels * 4, use_context=True)
        self.encoder4 = EncoderBlock(base_channels * 4, base_channels * 8, use_context=True)
        
        self.bottleneck = nn.Sequential(
            DoubleConv(base_channels * 8, base_channels * 16),
            ContextBlock(base_channels * 16)
        )
        
        self.decoder4 = DecoderBlock(base_channels * 16, base_channels * 8, base_channels * 8, use_attention)
        self.decoder3 = DecoderBlock(base_channels * 8, base_channels * 4, base_channels * 4, use_attention)
        self.decoder2 = DecoderBlock(base_channels * 4, base_channels * 2, base_channels * 2, use_attention)
        self.decoder1 = DecoderBlock(base_channels * 2, base_channels, base_channels, use_attention)
        
        self.final_conv = nn.Conv2d(base_channels, n_classes, kernel_size=1)
        
        if deep_supervision:
            self.aux_conv4 = nn.Conv2d(base_channels * 8, n_classes, kernel_size=1)
            self.aux_conv3 = nn.Conv2d(base_channels * 4, n_classes, kernel_size=1)
            self.aux_conv2 = nn.Conv2d(base_channels * 2, n_classes, kernel_size=1)
        
    def forward(self, x):
        x1, skip1 = self.encoder1(x)     
        x2, skip2 = self.encoder2(x1)    
        x3, skip3 = self.encoder3(x2)    
        x4, skip4 = self.encoder4(x3)    
        
        bottleneck = self.bottleneck(x4)  
        

        d4 = self.decoder4(bottleneck, skip4)  
        d3 = self.decoder3(d4, skip3)          
        d2 = self.decoder2(d3, skip2)        
        d1 = self.decoder1(d2, skip1)      
        
        output = self.final_conv(d1)
        
        if self.deep_supervision and self.training:
            aux4 = F.interpolate(self.aux_conv4(d4), size=x.shape[2:], mode='bilinear', align_corners=False)
            aux3 = F.interpolate(self.aux_conv3(d3), size=x.shape[2:], mode='bilinear', align_corners=False)
            aux2 = F.interpolate(self.aux_conv2(d2), size=x.shape[2:], mode='bilinear', align_corners=False)
            
            return output, aux4, aux3, aux2
        
        return output
    
    def get_model_info(self):
        return {
            'model_name': 'CENet',
            'input_channels': self.n_channels,
            'output_channels': self.n_classes,
            'use_attention': self.use_attention,
            'deep_supervision': self.deep_supervision,
            'description': 'CENet: Context Encoder Network for 2D Medical Image Segmentation'
        }

