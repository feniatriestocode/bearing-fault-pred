"""Dual-Branch Cross-Attention ViT Baseline (standard positional encoding).

Two separate token streams (vibration, current), each refined with a few
intra-modal self-attention layers first, then exchanged via bidirectional
cross-attention (vibration<->current) for a few more layers, before fusion
and classification.

This is intentionally the "standard PE" cross-attention baseline — same
architecture skeleton as the eventual physics-warped model, but with plain
learned positional embeddings, so the two can be compared head-to-head in
the ablation (standard PE vs physics-warped PE, holding fusion strategy
fixed).
"""
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
  """Single-modality patch embedding: [B, 1, H, W] -> [B, num_patches, D]."""

  def __init__(
      self,
      img_size: int = 64,
      patch_size: int = 8,
      in_channels: int = 1,
      embed_dim: int = 128,
  ):
    super().__init__()
    self.num_patches = (img_size // patch_size) ** 2
    self.proj = nn.Conv2d(
        in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: [B, 1, H, W] -> [B, D, H/p, W/p] -> [B, num_patches, D]
    return self.proj(x).flatten(2).transpose(1, 2)


class CrossAttentionBlock(nn.Module):
  """One directional cross-attention step: query_tokens attend to
  context_tokens (Key/Value), with residual + FFN, pre-norm style.
  """

  def __init__(self, embed_dim: int, heads: int, mlp_dim: int, dropout: float):
    super().__init__()
    self.norm_q = nn.LayerNorm(embed_dim)
    self.norm_kv = nn.LayerNorm(embed_dim)
    self.attn = nn.MultiheadAttention(
        embed_dim, heads, dropout=dropout, batch_first=True
    )
    self.norm_ffn = nn.LayerNorm(embed_dim)
    self.ffn = nn.Sequential(
        nn.Linear(embed_dim, mlp_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(mlp_dim, embed_dim),
        nn.Dropout(dropout),
    )

  def forward(
      self, query_tokens: torch.Tensor, context_tokens: torch.Tensor
  ) -> torch.Tensor:
    q = self.norm_q(query_tokens)
    kv = self.norm_kv(context_tokens)
    attn_out, _ = self.attn(q, kv, kv, need_weights=False)
    query_tokens = query_tokens + attn_out
    query_tokens = query_tokens + self.ffn(self.norm_ffn(query_tokens))
    return query_tokens


class BidirectionalCrossAttentionLayer(nn.Module):
  """Runs cross-attention in both directions for one layer:
  vibration attends to current, AND current attends to vibration,
  using separate (not shared) attention weights per direction.
  """

  def __init__(self, embed_dim: int, heads: int, mlp_dim: int, dropout: float):
    super().__init__()
    self.vib_attends_cur = CrossAttentionBlock(embed_dim, heads, mlp_dim, dropout)
    self.cur_attends_vib = CrossAttentionBlock(embed_dim, heads, mlp_dim, dropout)

  def forward(
      self, vib_tokens: torch.Tensor, cur_tokens: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    new_vib = self.vib_attends_cur(vib_tokens, cur_tokens)
    new_cur = self.cur_attends_vib(cur_tokens, vib_tokens)
    return new_vib, new_cur


class DualBranchCrossAttentionViT(nn.Module):
  """Baseline #2: dual-branch, standard positional encoding, bidirectional
  cross-attention fusion. Compare against BaselineCNN and BaselineViT
  (early/channel fusion) as the "cross-attention fusion" reference point,
  BEFORE physics-warped PE is introduced.
  """

  def __init__(
      self,
      img_size: int = 64,
      patch_size: int = 8,
      num_classes: int = 4,
      embed_dim: int = 128,
      intra_depth: int = 2,
      cross_depth: int = 2,
      heads: int = 4,
      mlp_dim: int = 256,
      dropout: float = 0.1,
  ):
    super().__init__()

    # --- Separate patch embedding per modality ---
    self.patch_embed_vib = PatchEmbedding(img_size, patch_size, 1, embed_dim)
    self.patch_embed_cur = PatchEmbedding(img_size, patch_size, 1, embed_dim)
    num_patches = self.patch_embed_vib.num_patches

    # --- Separate cls token + standard learned positional embedding per branch ---
    self.cls_vib = nn.Parameter(torch.zeros(1, 1, embed_dim))
    self.cls_cur = nn.Parameter(torch.zeros(1, 1, embed_dim))
    self.pos_embed_vib = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
    self.pos_embed_cur = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
    self.pos_drop = nn.Dropout(p=dropout)

    # --- Intra-modal self-attention refinement, BEFORE the two streams meet ---
    intra_layer = nn.TransformerEncoderLayer(
        d_model=embed_dim,
        nhead=heads,
        dim_feedforward=mlp_dim,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
    )
    self.intra_vib = nn.TransformerEncoder(intra_layer, num_layers=intra_depth)
    intra_layer_cur = nn.TransformerEncoderLayer(
        d_model=embed_dim,
        nhead=heads,
        dim_feedforward=mlp_dim,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
    )
    self.intra_cur = nn.TransformerEncoder(intra_layer_cur, num_layers=intra_depth)

    # --- Bidirectional cross-attention layers ---
    self.cross_layers = nn.ModuleList([
        BidirectionalCrossAttentionLayer(embed_dim, heads, mlp_dim, dropout)
        for _ in range(cross_depth)
    ])

    # --- Fusion + classification head ---
    self.norm_vib = nn.LayerNorm(embed_dim)
    self.norm_cur = nn.LayerNorm(embed_dim)
    self.head = nn.Linear(embed_dim * 2, num_classes)

    for p in [self.pos_embed_vib, self.pos_embed_cur, self.cls_vib, self.cls_cur]:
      nn.init.trunc_normal_(p, std=0.02)

  def forward(
      self, x: torch.Tensor, condition: torch.Tensor = None
  ) -> torch.Tensor:
    """Args:
        x: [B, 2, H, W] — channel 0 = vibration, channel 1 = current
           (same convention as PhysicsPreservingDownsampler's output).
        condition: unused here, kept for interface parity with the other
            baselines and the future physics-conditioned model.
    """
    b = x.shape[0]
    vib_in = x[:, 0:1]  # [B, 1, H, W]
    cur_in = x[:, 1:2]

    vib_tokens = self.patch_embed_vib(vib_in)  # [B, num_patches, D]
    cur_tokens = self.patch_embed_cur(cur_in)

    cls_vib = self.cls_vib.expand(b, -1, -1)
    cls_cur = self.cls_cur.expand(b, -1, -1)
    vib_tokens = torch.cat((cls_vib, vib_tokens), dim=1)
    cur_tokens = torch.cat((cls_cur, cur_tokens), dim=1)

    vib_tokens = self.pos_drop(vib_tokens + self.pos_embed_vib)
    cur_tokens = self.pos_drop(cur_tokens + self.pos_embed_cur)

    # Intra-modal refinement, independently
    vib_tokens = self.intra_vib(vib_tokens)
    cur_tokens = self.intra_cur(cur_tokens)

    # Bidirectional cross-attention, layer by layer
    for layer in self.cross_layers:
      vib_tokens, cur_tokens = layer(vib_tokens, cur_tokens)

    vib_cls_out = self.norm_vib(vib_tokens[:, 0])
    cur_cls_out = self.norm_cur(cur_tokens[:, 0])
    fused = torch.cat([vib_cls_out, cur_cls_out], dim=-1)
    return self.head(fused)