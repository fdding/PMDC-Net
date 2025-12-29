import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_components import DoubleConv


class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=2, bilinear=False, **kwargs):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self._create_standard_unet()
    
    def _create_standard_unet(self):
        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        factor = 2 if self.bilinear else 1
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024 // factor))
        
        self.up1 = self._make_up_layer(1024, 512 // factor, self.bilinear)
        self.up2 = self._make_up_layer(512, 256 // factor, self.bilinear)
        self.up3 = self._make_up_layer(256, 128 // factor, self.bilinear)
        self.up4 = self._make_up_layer(128, 64, self.bilinear)
        
        self.outc = nn.Conv2d(64, self.n_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(0.1)

    def _make_up_layer(self, in_channels, out_channels, bilinear):
        if bilinear:
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                DoubleConv(in_channels, out_channels, in_channels // 2)
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2),
                DoubleConv(in_channels, out_channels)
            )

    def forward(self, x):
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            
            x = self._upsample_and_concat(self.up1, x5, x4)
            x = self._upsample_and_concat(self.up2, x, x3)
            x = self._upsample_and_concat(self.up3, x, x2)
            x = self._upsample_and_concat(self.up4, x, x1)
            
            x = self.dropout(x)
            logits = self.outc(x)
            return logits

    def _upsample_and_concat(self, up_layer, x1, x2):
        if isinstance(up_layer[0], nn.Upsample):
            x1 = up_layer[0](x1)
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            x = torch.cat([x2, x1], dim=1)
            return up_layer[1](x)
        else:
            x1 = up_layer[0](x1)
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            x = torch.cat([x2, x1], dim=1)
            return up_layer[1](x)