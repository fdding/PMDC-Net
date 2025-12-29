import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(N, N, -1).permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        if min(input_resolution) <= window_size:
            self.window_size = min(input_resolution)
            self.shift_size = 0
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask, persistent=False)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = x + self.drop_path(self.norm1(x) - self.norm1(x).detach() + (self.norm1(x).detach() * 0))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else None
    def forward(self, x):
        x = self.proj(x)
        H, W = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        if self.norm:
            x = self.norm(x)
        return x, (H, W)

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x, (H // 2, W // 2)

class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, dim_scale * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // (dim_scale // 2) if dim_scale == 2 else dim)
    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = self.expand(x)
        if self.dim_scale == 2:
            x = x.view(B, H, W, 2 * C)
            x = x.permute(0, 3, 1, 2).contiguous()
            x = F.pixel_shuffle(x, 2)
            x = x.permute(0, 2, 3, 1).contiguous()
            H, W = H * 2, W * 2
            x = x.view(B, H * W, C // 2)
            x = self.norm(x)
            return x, (H, W)
        return x, (H, W)

class FinalPatchExpandX4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, (dim_scale * dim_scale) * dim, bias=False)
        self.norm = norm_layer(dim)
    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = self.expand(x)
        x = x.view(B, H, W, self.dim_scale * self.dim_scale * C)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = F.pixel_shuffle(x, self.dim_scale)
        x = x.permute(0, 2, 3, 1).contiguous()
        H, W = H * self.dim_scale, W * self.dim_scale
        x = x.view(B, H * W, C)
        x = self.norm(x)
        return x, (H, W)

class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                attn_drop=attn_drop, drop_path=dpr[i], norm_layer=norm_layer
            )
            for i in range(depth)
        ])
        self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer) if downsample else None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x, out_res = self.downsample(x)
            return x, out_res
        return x, self.input_resolution

class BasicLayerUp(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, upsample=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                attn_drop=attn_drop, drop_path=dpr[i], norm_layer=norm_layer
            )
            for i in range(depth)
        ])
        self.upsample = upsample(input_resolution, dim=dim, norm_layer=norm_layer) if upsample else None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.upsample is not None:
            x, out_res = self.upsample(x)
            return x, out_res
        return x, self.input_resolution

class SwinUNet(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=1,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.layers_down = nn.ModuleList()
        self.resolutions = []
        self.dims = []
        dim = embed_dim

        dummy_res = (img_size // patch_size, img_size // patch_size)
        for i_layer in range(self.num_layers):
            input_resolution = (dummy_res[0] // (2 ** i_layer), dummy_res[1] // (2 ** i_layer))
            self.resolutions.append(input_resolution)
            self.dims.append(dim)
            layer = BasicLayer(
                dim=dim,
                input_resolution=input_resolution,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate,
                norm_layer=norm_layer,
                downsample=PatchMerging if i_layer < self.num_layers - 1 else None
            )
            self.layers_down.append(layer)
            if i_layer < self.num_layers - 1:
                dim *= 2

        self.layers_up = nn.ModuleList()
        self.concat_linear = nn.ModuleList()
        for i_layer in range(self.num_layers - 1, 0, -1):
            dim = self.dims[i_layer]
            up_res = self.resolutions[i_layer]
            up = BasicLayerUp(
                dim=dim,
                input_resolution=up_res,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate,
                norm_layer=norm_layer,
                upsample=PatchExpand
            )
            self.layers_up.append(up)
            self.concat_linear.append(nn.Linear(dim // 2 + self.dims[i_layer - 1], dim // 2))

        self.norm = norm_layer(embed_dim)
        self.final_up = FinalPatchExpandX4(input_resolution=self.resolutions[0], dim=embed_dim, dim_scale=patch_size)
        self.output = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x, (H, W) = self.patch_embed(x)
        x = self.pos_drop(x)

        skips = []
        res = (H, W)
        for i, layer in enumerate(self.layers_down):
            x, res = layer(x)
            skips.append((x, res))

        x, res = skips[-1]
        skips = skips[:-1]

        for idx, up in enumerate(self.layers_up):
            x, res = up(x)
            skip_x, skip_res = skips[-(idx + 1)]
            if skip_res != res:
                B, L, C = skip_x.shape
                sh, sw = skip_res
                th, tw = res
                skip_x = skip_x.view(B, sh, sw, C).permute(0, 3, 1, 2)
                skip_x = F.interpolate(skip_x, size=(th, tw), mode="bilinear", align_corners=False)
                skip_x = skip_x.permute(0, 2, 3, 1).contiguous().view(B, th * tw, C)
            x = torch.cat([x, skip_x], dim=-1)
            x = self.concat_linear[idx](x)

        x = self.norm(x)
        x, (Hf, Wf) = self.final_up(x)
        x = self.output(x)
        x = x.view(x.shape[0], Hf, Wf, self.num_classes).permute(0, 3, 1, 2).contiguous()
        return x

if __name__ == "__main__":
    model = SwinUNet(img_size=224, patch_size=4, in_chans=3, num_classes=2)
    inp = torch.randn(1, 3, 224, 224)
    out = model(inp)
    print(out.shape)
