"""
Encoder modules for the multimodal cancer model.

  SlideEmbeddingEncoder : projects pre-extracted UNI / UNI2-h patch embeddings
  GeneEncoder           : MLP that compresses bulk RNA-seq to a latent vector
"""

import torch
import torch.nn as nn


class SlideEmbeddingEncoder(nn.Module):
    """
    Projects mean-pooled patch embeddings (UNI 1024-dim or UNI2-h 1536-dim)
    to a fixed 512-dim slide representation.

    Input:  [B, embedding_dim]
    Output: [B, 512]
    """
    FEATURE_DIM = 512

    def __init__(self, embedding_dim: int = 1024, dropout: float = 0.3):
        super().__init__()
        self.feature_dim = self.FEATURE_DIM
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class GeneEncoder(nn.Module):
    """
    Three-layer MLP that compresses high-dimensional RNA-seq profiles
    to a 256-dim latent vector.

    Input:  [B, n_genes]
    Output: [B, 128]
    """
    FEATURE_DIM = 128

    def __init__(self, n_genes: int, dropout: float = 0.3):
        super().__init__()
        self.feature_dim = self.FEATURE_DIM
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, 512),  nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),    nn.Dropout(dropout),
            nn.Linear(512, 256),      nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),    nn.Dropout(dropout),
            nn.Linear(256, 128),      nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B = 4

    slide_enc = SlideEmbeddingEncoder(embedding_dim=1536)
    gene_enc  = GeneEncoder(n_genes=20000)

    x_slide = torch.randn(B, 1536)
    x_gene  = torch.randn(B, 20000)

    out_slide = slide_enc(x_slide)
    out_gene  = gene_enc(x_gene)

    assert out_slide.shape == (B, 512), f"Expected (4,512), got {out_slide.shape}"
    assert out_gene.shape  == (B, 128), f"Expected (4,128), got {out_gene.shape}"

    print(f"SlideEmbeddingEncoder: {x_slide.shape} → {out_slide.shape}  ✓")
    print(f"GeneEncoder:           {x_gene.shape}  → {out_gene.shape}   ✓")
    print(f"Params (slide): {sum(p.numel() for p in slide_enc.parameters()):,}")
    print(f"Params (gene):  {sum(p.numel() for p in gene_enc.parameters()):,}")
