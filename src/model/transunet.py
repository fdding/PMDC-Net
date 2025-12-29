"""
TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base_components import DoubleConv


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channels=1, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.position_embeddings = nn.Parameter(torch.randn(1, self.n_patches + 1, embed_dim))
        
    def forward(self, x):
        B = x.shape[0]
        x = self.projection(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, embed_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        x = x + self.position_embeddings
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        
        return x


class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class CNNEncoder(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.conv1 = DoubleConv(in_channels, 64)
        self.conv2 = DoubleConv(64, 128)
        self.conv3 = DoubleConv(128, 256)
        
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        features = []
        
        # 第一层
        x1 = self.conv1(x)
        features.append(x1)
        x = self.pool(x1)
        
        # 第二层
        x2 = self.conv2(x)
        features.append(x2)
        x = self.pool(x2)
        
        # 第三层
        x3 = self.conv3(x)
        features.append(x3)
        x = self.pool(x3)
        
        return x, features


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)
        
    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            # 确保尺寸匹配
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class TransUNet(nn.Module):
    """
    TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation
    """
    def __init__(self, n_channels=1, n_classes=2, img_size=256, patch_size=16, 
                 embed_dim=768, num_heads=12, num_layers=6, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        self.cnn_encoder = CNNEncoder(n_channels)
        
        self.patch_embed = PatchEmbedding(
            img_size=img_size // 8,  
            patch_size=patch_size,
            in_channels=256,  
            embed_dim=embed_dim
        )
        
        self.transformer = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        
        self.reconstruct = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.decoder4 = DecoderBlock(512, 256, 256)
        self.decoder3 = DecoderBlock(256, 128, 128)
        self.decoder2 = DecoderBlock(128, 64, 64)
        self.decoder1 = DecoderBlock(64, 64, 64)  
        
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)
        
    def forward(self, x):
        cnn_features, skip_features = self.cnn_encoder(x)

        transformer_input = self.patch_embed(cnn_features)

        transformer_output = self.transformer(transformer_input)

        transformer_output = transformer_output[:, 1:]  
        B, N, C = transformer_output.shape
        H = W = int(math.sqrt(N))

        features = transformer_output.transpose(1, 2).reshape(B, C, H, W)
        features = self.reconstruct(features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if features.shape[2:] != cnn_features.shape[2:]:
            features = F.interpolate(features, size=cnn_features.shape[2:], mode='bilinear', align_corners=False)

        x = self.decoder4(features, skip_features[2])  
        x = self.decoder3(x, skip_features[1])         
        x = self.decoder2(x, skip_features[0])         
        x = self.decoder1(x, skip_features[0])         

        if x.shape[2:] != (self.img_size, self.img_size):
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)

        x = self.final_conv(x)
        
        return x
    
    def get_model_info(self):
        return {
            'model_name': 'TransUNet',
            'input_channels': self.n_channels,
            'output_channels': self.n_classes,
            'image_size': self.img_size,
            'patch_size': self.patch_size,
            'embed_dim': self.embed_dim,
            'description': 'TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation'
        }
