"""
  Reference: "Vessel-Net Retinal Vessel SegmentationUnder Multi-path Supervision"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from thop import profile
import torch.autograd.profiler as profiler


class IRBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(IRBlock, self).__init__()

    self.conv1x1_1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1),
                                nn.ReLU(),
                                nn.BatchNorm2d(out_channels))

    self.conv1x1_2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1),
                                nn.ReLU(),
                                nn.BatchNorm2d(out_channels))

    self.conv3x3_1 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=2, padding=2),
                                nn.ReLU(),
                                nn.BatchNorm2d(out_channels))
    
    self.conv3x3_2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=3, padding=3),
                                nn.ReLU(),
                                nn.BatchNorm2d(out_channels))

    self.out_conv = nn.Conv2d(out_channels*2+in_channels, out_channels, kernel_size=3, padding=1)

  def forward(self, x):
    x_path1 = self.conv1x1_1(x)

    x_path2 = self.conv1x1_2(x)
    x_path2 = x_path2 + self.conv3x3_1(x_path2)
    x_path2 = x_path2 + self.conv3x3_2(x_path2)

    out = torch.cat((x, x_path1, x_path2), dim=1)

    out = self.out_conv(out)

    return out

class ConvRL(nn.Module):
    def __init__(self, in_channel, out_channel, kernel=2, stride=2):
        super().__init__()

        self.conv_rl = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel, stride=stride),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv_rl(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, 2, stride=2)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class VesselNet(nn.Module):
    """
    VesselNet: Retinal Vessel Segmentation Under Multi-path Supervision
    """
    def __init__(self, n_channels=1, n_classes=2, base_channels=32, multi_output=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.multi_output = multi_output
        
        self.ir1 = IRBlock(n_channels, base_channels)
        self.pool1 = ConvRL(base_channels, base_channels)

        self.ir2 = IRBlock(base_channels, base_channels*2)
        self.pool2 = ConvRL(base_channels*2, base_channels*2)

        self.ir3 = IRBlock(base_channels*2, base_channels*4)

        self.ir4 = IRBlock(base_channels*6, base_channels*2)

        self.ir5 = IRBlock(base_channels*3, base_channels)

        self.out1 = ConvRL(base_channels, n_classes, 1, 1)
        self.out2 = ConvRL(base_channels*2, 1, 1, 1)
        self.out3 = ConvRL(base_channels*4, 1, 1, 1)
        self.out4 = ConvRL(base_channels*2, 1, 1, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        x1 = self.ir1(x)

        x2 = self.pool1(x1)
        x2 = self.ir2(x2)

        x3 = self.pool2(x2)
        x3 = self.ir3(x3)

        x4 = self.ir4(
            torch.cat([x2, F.interpolate(x3, size=x2.size()[-2:], mode='bilinear')], dim=1))

        x5 = self.ir5(
            torch.cat([x1, F.interpolate(x4, size=x1.size()[-2:], mode='bilinear')], dim=1))

        out1 = self.out1(x5)

        if self.multi_output and self.training:
            out2 = self.sigmoid(self.out2(torch.cat([x1, x5], dim=1)))
            out3 = self.sigmoid(self.out3(x3))
            out4 = self.sigmoid(self.out4(x4))
            
            out2 = F.interpolate(out2, size=(h, w), mode='bilinear', align_corners=False)
            out3 = F.interpolate(out3, size=(h, w), mode='bilinear', align_corners=False)
            out4 = F.interpolate(out4, size=(h, w), mode='bilinear', align_corners=False)
            
            return out1, out2, out3, out4
        else:
            return out1
    
    def get_model_info(self):
        return {
            'model_name': 'VesselNet',
            'input_channels': self.n_channels,
            'output_channels': self.n_classes,
            'multi_output': self.multi_output,
            'description': 'VesselNet: Retinal Vessel Segmentation Under Multi-path Supervision'
        }






















