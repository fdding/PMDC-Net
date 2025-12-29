
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_components import DoubleConv, AttentionGate


class AttentionUNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=2, **kwargs):
        super(AttentionUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

            self._create_standard_attention_unet()
    
    def _create_standard_attention_unet(self):
        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        
        self.att1 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.att2 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.att3 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.att4 = AttentionGate(F_g=64, F_l=64, F_int=32)

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(1024, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(128, 64)
        
        self.outc = nn.Conv2d(64, self.n_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
            x1 = self.inc(x)      # [B, 64, H, W]
            x2 = self.down1(x1)   # [B, 128, H/2, W/2]
            x3 = self.down2(x2)   # [B, 256, H/4, W/4]
            x4 = self.down3(x3)   # [B, 512, H/8, W/8]
            x5 = self.down4(x4)   # [B, 1024, H/16, W/16]
            
            d5 = self.up1(x5)     # [B, 512, H/8, W/8]
            x4 = self.att1(g=d5, x=x4)  
            d5 = torch.cat((x4, d5), dim=1)
            d5 = self.conv_up1(d5)
            
            d4 = self.up2(d5)     # [B, 256, H/4, W/4]
            x3 = self.att2(g=d4, x=x3)
            d4 = torch.cat((x3, d4), dim=1)
            d4 = self.conv_up2(d4)
            
            d3 = self.up3(d4)     # [B, 128, H/2, W/2]
            x2 = self.att3(g=d3, x=x2)
            d3 = torch.cat((x2, d3), dim=1)
            d3 = self.conv_up3(d3)
            
            d2 = self.up4(d3)     # [B, 64, H, W]
            x1 = self.att4(g=d2, x=x1)
            d2 = torch.cat((x1, d2), dim=1)
            d2 = self.conv_up4(d2)
            
            d2 = self.dropout(d2)
            out = self.outc(d2)
            
            return out