import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ===============================================================================
#                              HELPER FUNCTIONS
# ===============================================================================
def window_partition(x, window_size):
    """
    Partition feature map into non-overlapping windows.
    Used in HAT and can be reused here if needed.
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    windows = windows.view(-1, C, window_size, window_size)
    return windows, (H, W)


# ---------------------------------------
#                   HAT
# ---------------------------------------
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) for regularization."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class LayerNorm2d(nn.Module):
    """
    LayerNorm over channels for BCHW tensors.
    """
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        # x: [B, C, H, W]
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.weight + self.bias


class ChannelAttention(nn.Module):
    """
    Squeeze-and-excitation style channel attention.
    This is the "hybrid" part combined with window self-attention.
    """
    def __init__(self, dim, reduction=4):
        super().__init__()
        hidden = max(dim // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(dim, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, dim, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.act(self.fc1(y))
        y = self.fc2(y)
        return x * torch.sigmoid(y)


def window_partition_for_hat(x, window_size):
    """
    Args:
        x: tensor of shape [B, C, H, W]
        window_size: int, window size (ws)

    Returns:
        windows: [B * nH * nW, C, window_size, window_size]
        meta: (B, nH, nW) for possible debugging/use
    """
    B, C, H, W = x.shape
    assert H % window_size == 0 and W % window_size == 0, \
        f"H={H}, W={W} must be divisible by window_size={window_size}"

    nH = H // window_size
    nW = W // window_size

    # [B, C, H, W] ->
    # [B, C, nH, ws, nW, ws] ->
    # [B, nH, nW, C, ws, ws] ->
    # [B * nH * nW, C, ws, ws]
    x = x.view(B, C, nH, window_size, nW, window_size)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    windows = x.view(B * nH * nW, C, window_size, window_size)
    return windows, (B, nH, nW)


def window_reverse(windows, window_size, B, H, W):
    """
    Reverse of window_partition.

    Args:
        windows: [B * nH * nW, C, window_size, window_size]
        window_size: int
        B, H, W: original image batch size and spatial dimensions

    Returns:
        x: [B, C, H, W]
    """
    C = windows.shape[1]
    nH = H // window_size
    nW = W // window_size

    # [B * nH * nW, C, ws, ws] ->
    # [B, nH, nW, C, ws, ws] ->
    # [B, C, nH, ws, nW, ws] ->
    # [B, C, H, W]
    x = windows.view(B, nH, nW, C, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    x = x.view(B, C, H, W)
    return x


class WindowAttention(nn.Module):
    """
    Window-based multi-head self-attention with learnable relative position bias.
    Uses the same window partitioning as PFT-SR.
    """
    def __init__(self, dim, window_size=8, num_heads=6):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

        ws = window_size
        # relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * ws - 1) * (2 * ws - 1), num_heads)
        )

        # compute relative position index
        coords_h = torch.arange(ws)
        coords_w = torch.arange(ws)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # [2, ws, ws]
        coords_flatten = torch.flatten(coords, 1)  # [2, ws*ws]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, N, N]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()           # [N, N, 2]
        relative_coords[:, :, 0] += ws - 1
        relative_coords[:, :, 1] += ws - 1
        relative_coords[:, :, 0] *= 2 * ws - 1
        relative_position_index = relative_coords.sum(-1)                         # [N, N]
        self.register_buffer("relative_position_index", relative_position_index)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x):
        """
        x: [B, C, H, W] where H,W are divisible by window_size
        """
        B, C, H, W = x.shape
        ws = self.window_size

        # B*nW, C, ws, ws
        x_windows, _ = window_partition_for_hat(x, ws)
        BnW = x_windows.size(0)
        N = ws * ws

        # [BnW, C, ws, ws] -> [BnW, N, C]
        x_flat = x_windows.flatten(2).transpose(1, 2)

        qkv = self.qkv(x_flat)  # [BnW, N, 3*dim]
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(BnW, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(BnW, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(BnW, N, self.num_heads, C // self.num_heads).transpose(1, 2)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # [BnW, h, N, N]

        # add relative position bias
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(N, N, self.num_heads).permute(2, 0, 1)  # [h, N, N]
        attn = attn + bias.unsqueeze(0)

        attn = torch.softmax(attn, dim=-1)
        out = attn @ v  # [BnW, h, N, d]
        out = out.transpose(1, 2).reshape(BnW, N, C)  # [BnW, N, C]

        out = self.proj(out)
        out = out.transpose(1, 2).view(BnW, C, ws, ws)

        x_out = window_reverse(out, ws, B, H, W)
        return x_out


class HATMLP(nn.Module):
    """
    Conv-MLP used inside HAT transformer blocks.
    """
    def __init__(self, dim, mlp_ratio=2.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.conv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.conv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return x + residual


class HATBlock(nn.Module):
    """
    One HAT block:
      x -> LN2d -> WindowAttention -> DropPath -> +
         -> LN2d -> MLP -> ChannelAttention -> DropPath -> +
    """
    def __init__(self,
                 dim,
                 window_size=8,
                 num_heads=6,
                 mlp_ratio=2.0,
                 drop_path=0.0):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = LayerNorm2d(dim)
        self.mlp = HATMLP(dim, mlp_ratio)
        self.ca = ChannelAttention(dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        # window attention branch
        x_attn = self.attn(self.norm1(x))
        x = x + self.drop_path1(x_attn)
        # MLP + channel attention
        x_mlp = self.mlp(self.norm2(x))
        x_mlp = self.ca(x_mlp)
        x = x + self.drop_path2(x_mlp)
        return x


class HATGroup(nn.Module):
    """
    Residual group of multiple HAT blocks with a conv at the end.
    """
    def __init__(self,
                 dim,
                 depth,
                 window_size=8,
                 num_heads=6,
                 mlp_ratio=2.0,
                 drop_path=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            HATBlock(dim, window_size, num_heads, mlp_ratio, drop_path)
            for _ in range(depth)
        ])
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        residual = x
        for blk in self.blocks:
            x = blk(x)
        x = self.conv(x)
        return x + residual


class HATSRNet(nn.Module):
    """
    Hybrid Attention Transformer for Single Image Super-Resolution.

    - Shallow conv
    - Several HATGroups (transformer body)
    - Conv fusion
    - PixelShuffle x4 upsampling
    """
    def __init__(self,
                 scale=4,
                 num_in_ch=3,
                 num_out_ch=3,
                 dim=96,
                 num_groups=4,
                 depth_per_group=6,
                 window_size=8,
                 num_heads=6,
                 mlp_ratio=2.0,
                 drop_path=0.0):
        super().__init__()
        assert scale in [2, 4], "HATSRNet supports x2 or x4."

        self.scale = scale
        self.dim = dim
        self.window_size = window_size

        # Shallow feature extraction
        self.conv_first = nn.Conv2d(num_in_ch, dim, kernel_size=3, stride=1, padding=1)

        # Transformer body
        self.groups = nn.ModuleList([
            HATGroup(dim, depth_per_group, window_size,
                     num_heads, mlp_ratio, drop_path)
            for _ in range(num_groups)
        ])
        self.conv_between = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)

        # Upsampling head
        up_layers = []
        if scale == 4:
            for _ in range(2):
                up_layers += [
                    nn.Conv2d(dim, dim * 4, kernel_size=3, stride=1, padding=1),
                    nn.PixelShuffle(2),
                    nn.GELU(),
                ]
        else:  # scale == 2
            up_layers += [
                nn.Conv2d(dim, dim * 4, kernel_size=3, stride=1, padding=1),
                nn.PixelShuffle(2),
                nn.GELU(),
            ]
        self.upsampler = nn.Sequential(*up_layers)
        self.conv_last = nn.Conv2d(dim, num_out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        x: [B, 3, H, W] in [0,1]
        """
        B, C, H, W = x.shape
        ws = self.window_size

        # optional padding so H,W divisible by window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h != 0 or pad_w != 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        fea = self.conv_first(x)
        residual = fea

        # transformer body
        for g in self.groups:
            fea = g(fea)
        fea = self.conv_between(fea)
        fea = fea + residual

        # upsampling
        fea_up = self.upsampler(fea)
        out = self.conv_last(fea_up)

        # remove padding on HR if we padded LR
        if pad_h != 0 or pad_w != 0:
            H_out = H * self.scale
            W_out = W * self.scale
            out = out[..., :H_out, :W_out]

        return out



# ---------------------------------------
#                 ESRGAN
# ---------------------------------------
class ResidualDenseBlock(nn.Module):
    """
    ESRGAN Residual Dense Block (RDB) with 5 conv layers.
    Keeps spatial size, input/output channels = nf.
    """
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.gc = gc
        self.nf = nf

        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
        # Local residual scaling
        return x + x5 * 0.2


class RRDB(nn.Module):
    """
    Residual-in-Residual Dense Block (RRDB):
    3 RDBs with outer residual connection.
    """
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + out * 0.2


class ESRGANGenerator(nn.Module):
    """
    Original-style ESRGAN generator (RRDBNet) with internal x4 upsampling.

    Input: LR image (e.g. 64x64)
    Output: HR image (e.g. 256x256)
    """
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4):
        super().__init__()
        self.scale = scale

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)

        rrdb_blocks = [RRDB(nf, gc) for _ in range(nb)]
        self.RRDB_trunk = nn.Sequential(*rrdb_blocks)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1)

        # Upsampling for x4: two times x2
        if scale == 4:
            self.upsample_layers = nn.ModuleList([
                nn.Conv2d(nf, nf, 3, 1, 1),
                nn.Conv2d(nf, nf, 3, 1, 1),
            ])
        elif scale == 2:
            self.upsample_layers = nn.ModuleList([
                nn.Conv2d(nf, nf, 3, 1, 1),
            ])
        else:
            raise ValueError(f"Unsupported scale factor: {scale} (only 2 or 4 supported)")

        self.HR_conv = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk  # long skip

        # Upsample x2 per layer
        for upconv in self.upsample_layers:
            fea = F.interpolate(fea, scale_factor=2, mode='nearest')
            fea = self.lrelu(upconv(fea))

        fea = self.lrelu(self.HR_conv(fea))
        out = self.conv_last(fea)
        return out


class ESRGANDiscriminator(nn.Module):
    """
    PatchGAN-style discriminator for ESRGAN.
    Expects HR patches (e.g., 128x128 or 192x192).
    """
    def __init__(self, in_nc=3, base_nf=64):
        super().__init__()
        nf = base_nf

        layers = []
        # conv1: no BN
        layers += [
            nn.Conv2d(in_nc, nf, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Subsequent layers: conv + BN + LReLU
        in_out = [
            (nf, nf, 2),
            (nf, nf * 2, 1),
            (nf * 2, nf * 2, 2),
            (nf * 2, nf * 4, 1),
            (nf * 4, nf * 4, 2),
            (nf * 4, nf * 8, 1),
            (nf * 8, nf * 8, 2),
        ]

        for in_c, out_c, stride in in_out:
            layers += [
                nn.Conv2d(in_c, out_c, 3, stride, 1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(nf * 8, nf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(nf * 8, 1),
        )

    def forward(self, x):
        fea = self.features(x)
        out = self.classifier(fea)
        return out


class VGGFeatureExtractor(nn.Module):
    """
    Perceptual feature extractor using VGG19 (ImageNet pretrained).
    Default: features up to conv5_4 (layer 35).
    """
    def __init__(self, layer_index=35, use_input_norm=True):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.features = nn.Sequential(*list(vgg.children())[:layer_index + 1])
        for p in self.features.parameters():
            p.requires_grad = False

        self.use_input_norm = use_input_norm
        if use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer("mean", mean)
            self.register_buffer("std", std)

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        return self.features(x)



# ---------------------------------------
#                 SRCNN
# ---------------------------------------
class SRCNN(nn.Module):
    """
    Simple SRCNN for RGB images.
    Original paper uses Y channel only; here we use 3 channels for simplicity.
    Structure:
      Conv1: 9x9, 3 -> 64, ReLU
      Conv2: 1x1, 64 -> 32, ReLU
      Conv3: 5x5, 32 -> 3
    """

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)

        return x
