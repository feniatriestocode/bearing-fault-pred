"""Custom Standard Vision Transformer Baseline."""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):

  def __init__(
      self,
      img_size: int = 64,
      patch_size: int = 8,
      in_channels: int = 2,
      embed_dim: int = 128,
  ):
    super().__init__()
    self.img_size = img_size
    self.patch_size = patch_size
    self.num_patches = (img_size // patch_size) ** 2  # (64/8)^2 = 64
    self.proj = nn.Conv2d(
        in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: [B, 2, 64, 64] -> proj: [B, D, 8, 8] -> flatten: [B, D, 64] -> transpose: [B, 64, D]
    return self.proj(x).flatten(2).transpose(1, 2)


class BaselineViT(nn.Module):

  def __init__(
      self,
      img_size: int = 64,
      patch_size: int = 8,
      in_channels: int = 2,
      num_classes: int = 4,
      embed_dim: int = 128,
      depth: int = 4,
      heads: int = 4,
      mlp_dim: int = 256,
      dropout: float = 0.1,
  ):
    super().__init__()
    self.patch_embed = PatchEmbedding(
        img_size, patch_size, in_channels, embed_dim
    )
    num_patches = self.patch_embed.num_patches

    self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
    self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
    self.pos_drop = nn.Dropout(p=dropout)

    encoder_layer = nn.TransformerEncoderLayer(
        d_model=embed_dim,
        nhead=heads,
        dim_feedforward=mlp_dim,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
    )
    self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
    self.norm = nn.LayerNorm(embed_dim)
    self.head = nn.Linear(embed_dim, num_classes)

    nn.init.trunc_normal_(self.pos_embed, std=0.02)
    nn.init.trunc_normal_(self.cls_token, std=0.02)

  def forward(
      self, x: torch.Tensor, condition: torch.Tensor = None
  ) -> torch.Tensor:
    b = x.shape[0]
    tokens = self.patch_embed(x)  # [B, 64, D]

    cls_tokens = self.cls_token.expand(b, -1, -1)  # [B, 1, D]
    tokens = torch.cat((cls_tokens, tokens), dim=1)  # [B, 65, D]
    tokens = self.pos_drop(tokens + self.pos_embed)

    encoded = self.encoder(tokens)
    cls_out = self.norm(encoded[:, 0])
    return self.head(cls_out)