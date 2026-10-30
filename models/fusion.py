"""
Fusion modules — combine image and gene embeddings into a single vector.

  CrossModalAttention : learned per-patient softmax weights (default)
  ConcatFusion        : project both → concatenate → linear (ablation)

Both output a 256-dim fused vector so downstream heads are identical.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

FUSED_DIM = 256  # output dimension for all fusion modules


class CrossModalAttention(nn.Module):
    """
    Projects image and gene features to FUSED_DIM each, then computes
    a learned softmax attention weight per modality and returns the
    weighted sum.

    This lets the model assign α_image and α_gene per patient —
    useful when modality informativeness varies across samples.

    Input:  img_feat  [B, image_dim]
            gene_feat [B, gene_dim]
    Output: fused     [B, FUSED_DIM]
            weights   [B, 2]  (image weight, gene weight)
    """

    def __init__(self, image_dim: int, gene_dim: int):
        super().__init__()
        d = FUSED_DIM
        self.img_proj  = nn.Linear(image_dim, d)
        self.gene_proj = nn.Linear(gene_dim,  d)
        self.attn = nn.Sequential(
            nn.Linear(d, 64), nn.Tanh(), nn.Linear(64, 1)
        )

    def forward(self, img_feat: torch.Tensor,
                gene_feat: torch.Tensor):
        img_p  = self.img_proj(img_feat)             # [B, 256]
        gene_p = self.gene_proj(gene_feat)           # [B, 256]

        stack   = torch.stack([img_p, gene_p], dim=1)  # [B, 2, 256]
        scores  = self.attn(stack)                      # [B, 2, 1]
        weights = F.softmax(scores, dim=1)              # [B, 2, 1]
        fused   = (stack * weights).sum(dim=1)          # [B, 256]
        return fused, weights.squeeze(-1)               # [B,256], [B,2]


class ConcatFusion(nn.Module):
    """
    Ablation alternative to CrossModalAttention.

    Projects both modalities to FUSED_DIM, concatenates them (→ 2×FUSED_DIM),
    then projects back to FUSED_DIM with a learned linear layer.

    Unlike attention, the combination weights are fixed (equal contribution
    before the final linear), making this a strong but simpler baseline.

    Input:  img_feat  [B, image_dim]
            gene_feat [B, gene_dim]
    Output: fused     [B, FUSED_DIM]
            weights   [B, 2]  (always 0.5, 0.5 — placeholder for API compat)
    """

    def __init__(self, image_dim: int, gene_dim: int):
        super().__init__()
        d = FUSED_DIM
        self.img_proj  = nn.Linear(image_dim, d)
        self.gene_proj = nn.Linear(gene_dim,  d)
        self.combine   = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.ReLU(inplace=True),
        )

    def forward(self, img_feat: torch.Tensor,
                gene_feat: torch.Tensor):
        img_p  = self.img_proj(img_feat)             # [B, 256]
        gene_p = self.gene_proj(gene_feat)           # [B, 256]

        cat    = torch.cat([img_p, gene_p], dim=1)   # [B, 512]
        fused  = self.combine(cat)                   # [B, 256]

        # Return uniform weights for API compatibility with attention
        weights = torch.full((img_feat.size(0), 2), 0.5,
                             device=img_feat.device)
        return fused, weights


def build_fusion(fusion_type: str, image_dim: int, gene_dim: int) -> nn.Module:
    """
    Factory function.
    fusion_type: 'attention' (default) | 'concat'
    """
    if fusion_type == "attention":
        return CrossModalAttention(image_dim, gene_dim)
    elif fusion_type == "concat":
        return ConcatFusion(image_dim, gene_dim)
    else:
        raise ValueError(f"Unknown fusion type '{fusion_type}'. "
                         f"Choose 'attention' or 'concat'.")


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, IMG_DIM, GENE_DIM = 4, 512, 128
    img  = torch.randn(B, IMG_DIM)
    gene = torch.randn(B, GENE_DIM)

    for name in ("attention", "concat"):
        fusion = build_fusion(name, IMG_DIM, GENE_DIM)
        fused, weights = fusion(img, gene)
        assert fused.shape   == (B, FUSED_DIM), f"{name}: wrong fused shape"
        assert weights.shape == (B, 2),         f"{name}: wrong weights shape"
        params = sum(p.numel() for p in fusion.parameters())
        print(f"{name:12s}: fused={fused.shape}  weights={weights.shape}  "
              f"params={params:,}  ✓")
        print(f"              sample weights: "
              f"img={weights[0,0]:.3f}  gene={weights[0,1]:.3f}")
