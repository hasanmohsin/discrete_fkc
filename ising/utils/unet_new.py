import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Utilities
# -------------------------
def sinusoidal_embedding(timesteps, dim):
    """
    timesteps: (B,) integer/float tensor
    returns: (B, dim) float tensor
    """
    device = timesteps.device
    half_dim = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half_dim, device=device, dtype=torch.float32) / half_dim)
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:  # odd dims
        emb = F.pad(emb, (0, 1))
    return emb  # (B, dim)


# -------------------------
# ResBlock with Time FiLM
# -------------------------
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        groups1 = min(32, in_ch)
        while in_ch % groups1 != 0 and groups1 > 1:
            groups1 -= 1
        groups2 = min(32, out_ch)
        while out_ch % groups2 != 0 and groups2 > 1:
            groups2 -= 1

        self.norm1 = nn.GroupNorm(groups1, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups2, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.time_fc = nn.Linear(time_emb_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        """
        x: (B, C, H, W)
        t_emb: (B, time_emb_dim)
        """
        h = self.conv1(F.silu(self.norm1(x)))
        # FiLM-like add (broadcast)
        t_out = self.time_fc(t_emb)[:, :, None, None]
        h = h + t_out
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


# -------------------------
# Attention Block (spatial)
# -------------------------
class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        # pick a num_heads that divides channels
        heads = min(num_heads, channels)
        while heads > 1 and (channels % heads != 0):
            heads -= 1
        if heads < 1:
            heads = 1
        self.norm = nn.GroupNorm(1, channels)  # layer norm across channels
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=heads, batch_first=False)  # expects (L,B,E)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        """
        x: (B, C, H, W)
        returns: (B, C, H, W)
        """
        B, C, H, W = x.shape
        h = self.norm(x)
        # flatten spatial dims to sequence
        seq = h.view(B, C, H * W).permute(2, 0, 1)   # (L=N, B, C)
        attn_out, _ = self.mha(seq, seq, seq, need_weights=False)  # (L, B, C)
        attn_out = attn_out.permute(1, 2, 0).view(B, C, H, W)
        return x + self.proj_out(attn_out)  # residual


# -------------------------
# Up/Down sampling helpers
# -------------------------
def downsample_conv(in_ch, out_ch):
    return nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)  # halves spatial dims

def upsample_conv(in_ch, out_ch):
    return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)  # doubles spatial dims


# -------------------------
# Upgraded Spin U-Net
# -------------------------
class SpinUNetUpgraded(nn.Module):
    def __init__(self, base_channels=64, time_emb_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )

        # channel schedule: 16x16 -> 8x8 -> 4x4 (bottleneck)
        c1 = base_channels               # 16x16
        c2 = base_channels * 2           # 8x8
        c3 = base_channels * 4           # 4x4 (bottleneck)

        # Encoder stage 1 (two ResBlocks)
        self.enc1_rb1 = ResBlock(2, c1, time_emb_dim)
        self.enc1_rb2 = ResBlock(c1, c1, time_emb_dim)
        self.down1 = downsample_conv(c1, c2)  # -> 8x8

        # Encoder stage 2 (two ResBlocks)
        self.enc2_rb1 = ResBlock(c2, c2, time_emb_dim)
        self.enc2_rb2 = ResBlock(c2, c2, time_emb_dim)
        self.down2 = downsample_conv(c2, c3)  # -> 4x4

        # Bottleneck: 2 ResBlocks + Attention
        self.bot_rb1 = ResBlock(c3, c3, time_emb_dim)
        self.attn = AttentionBlock(c3, num_heads=8)
        self.bot_rb2 = ResBlock(c3, c3, time_emb_dim)

        # Decoder stage 1 (upsample 4x4 -> 8x8)
        self.up1 = upsample_conv(c3, c2)
        # after cat with encoder features: channels = c2 + c2 = 2*c2
        self.dec1_rb1 = ResBlock(c2 * 2, c2, time_emb_dim)
        self.dec1_rb2 = ResBlock(c2, c2, time_emb_dim)

        # Decoder stage 2 (8x8 -> 16x16)
        self.up2 = upsample_conv(c2, c1)
        # after cat with encoder features: channels = c1 + c1 = 2*c1
        self.dec2_rb1 = ResBlock(c1 * 2, c1, time_emb_dim)
        self.dec2_rb2 = ResBlock(c1, c1, time_emb_dim)

        # final conv to logits (2 classes)
        self.out_conv = nn.Conv2d(c1, 2, kernel_size=1)

    def forward(self, x_idx, t):
        """
        x_idx: (B, 256) integers {0,1}
        t: (B,) integer/float timesteps
        returns logits: (B, 256, 2) -- ready for cross_entropy after flattening
        """
        B = x_idx.size(0)
        # indices -> one-hot -> (B,2,16,16)
        x_onehot = F.one_hot(x_idx.long(), num_classes=2).float()  # (B,256,2)
        x = x_onehot.view(B, 16, 16, 2).permute(0, 3, 1, 2).contiguous()

        # time embedding
        t_emb = sinusoidal_embedding(1000*t.squeeze(), self.time_mlp[0].in_features)
        t_emb = self.time_mlp(t_emb)

        # encoder 1
        e1 = self.enc1_rb1(x, t_emb)
        e1 = self.enc1_rb2(e1, t_emb)
        d1 = self.down1(e1)   # (B, c2, 8, 8)

        # encoder 2
        e2 = self.enc2_rb1(d1, t_emb)
        e2 = self.enc2_rb2(e2, t_emb)
        d2 = self.down2(e2)   # (B, c3, 4, 4)

        # bottleneck
        b = self.bot_rb1(d2, t_emb)
        b = self.attn(b)
        b = self.bot_rb2(b, t_emb)

        # decoder 1
        u1 = self.up1(b)  # (B, c2, 8, 8)
        u1 = torch.cat([u1, e2], dim=1)  # (B, 2*c2, 8,8)
        u1 = self.dec1_rb1(u1, t_emb)
        u1 = self.dec1_rb2(u1, t_emb)

        # decoder 2
        u2 = self.up2(u1)  # (B, c1, 16,16)
        u2 = torch.cat([u2, e1], dim=1)  # (B, 2*c1, 16,16)
        u2 = self.dec2_rb1(u2, t_emb)
        u2 = self.dec2_rb2(u2, t_emb)

        out = self.out_conv(F.silu(u2))  # (B, 2, 16,16)
        out = out.permute(0, 2, 3, 1).reshape(B, 256, 2)  # (B, 256, 2)
        return out
