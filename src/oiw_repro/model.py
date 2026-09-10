from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    in_channels: int = 3
    cnn_channels: Sequence[int] = (128, 256, 512)
    embedding_dim: int = 96
    num_transformer_blocks: int = 4
    num_attention_heads: int = 4
    sra_reduction_ratios: Sequence[int] = (8, 4, 2, 1)
    dropout: float = 0.10


class ConvBNGELU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SpatialReductionAttention(nn.Module):
    """Multi-head self-attention with spatial reduction for keys and values."""

    def __init__(self, dim: int, num_heads: int, sr_ratio: int = 1, dropout: float = 0.1):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("embedding dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sr_ratio = int(max(1, sr_ratio))
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.sr_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        if self.sr_ratio > 1:
            fmap = x.transpose(1, 2).reshape(b, c, h, w)
            kh = max(1, min(self.sr_ratio, h))
            kw = max(1, min(self.sr_ratio, w))
            fmap = F.avg_pool2d(fmap, kernel_size=(kh, kw), stride=(kh, kw), ceil_mode=True)
            kv_tokens = fmap.flatten(2).transpose(1, 2)
            kv_tokens = self.sr_norm(kv_tokens)
        else:
            kv_tokens = x

        kv = self.kv(kv_tokens).reshape(b, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class TransformerBlockSRA(nn.Module):
    def __init__(self, dim: int, num_heads: int, sr_ratio: int, dropout: float = 0.1, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SpatialReductionAttention(dim, num_heads, sr_ratio, dropout)
        self.drop_path = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), h, w))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


class HybridOIWDetector(nn.Module):
    """CNN stem + SRA Transformer + pooled binary classifier."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        channels = list(cfg.cnn_channels)
        cnn_layers = []
        in_ch = cfg.in_channels
        for out_ch in channels:
            cnn_layers.append(ConvBNGELU(in_ch, out_ch, stride=2))
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)
        self.proj = nn.Conv2d(channels[-1], cfg.embedding_dim, kernel_size=1)
        ratios = list(cfg.sra_reduction_ratios)
        if len(ratios) < cfg.num_transformer_blocks:
            ratios = ratios + [ratios[-1] if ratios else 1] * (cfg.num_transformer_blocks - len(ratios))
        self.blocks = nn.ModuleList([
            TransformerBlockSRA(cfg.embedding_dim, cfg.num_attention_heads, ratios[i], cfg.dropout)
            for i in range(cfg.num_transformer_blocks)
        ])
        self.norm = nn.LayerNorm(cfg.embedding_dim)
        self.head = nn.Linear(cfg.embedding_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = self.proj(x)
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        for block in self.blocks:
            tokens = block(tokens, h, w)
        pooled = self.norm(tokens).mean(dim=1)
        return self.head(pooled).squeeze(1)


def build_model_from_config(config: dict) -> HybridOIWDetector:
    model_cfg = config.get("model", {})
    cfg = ModelConfig(
        in_channels=3,
        cnn_channels=tuple(model_cfg.get("cnn_channels", [128, 256, 512])),
        embedding_dim=int(model_cfg.get("embedding_dim", 96)),
        num_transformer_blocks=int(model_cfg.get("num_transformer_blocks", 4)),
        num_attention_heads=int(model_cfg.get("num_attention_heads", 4)),
        sra_reduction_ratios=tuple(model_cfg.get("sra_reduction_ratios", [8, 4, 2, 1])),
        dropout=float(model_cfg.get("dropout", 0.10)),
    )
    return HybridOIWDetector(cfg)
