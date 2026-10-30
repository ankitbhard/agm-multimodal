"""
Unified Joint Model for pan-cancer multimodal survival learning.

Handles all 6 cancer types (BRCA, LUAD, KIRC, STAD, LUSC, GBM) via config:
  - Variable image embedding dim (1024 UNI / 1536 UNI2-h)
  - Pluggable fusion: 'attention' (default) or 'concat' (ablation)
  - Single survival head only (binary: died ≤3yr vs alive >3yr)

Usage:
    from models.joint_model import build_model

    model = build_model(
        n_genes      = 20000,
        image_dim    = 1536,   # UNI2-h
        fusion_type  = "attention",
    )
    out = model({"image": img_tensor, "gene": gene_tensor})
    # out["survival"]  → [B, 2]
"""

import torch
import torch.nn as nn

try:
    from models.encoders import SlideEmbeddingEncoder, GeneEncoder
    from models.fusion   import build_fusion, FUSED_DIM
except ModuleNotFoundError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from models.encoders import SlideEmbeddingEncoder, GeneEncoder
    from models.fusion   import build_fusion, FUSED_DIM


class JointModel(nn.Module):
    """
    image embedding + gene expression → survival (single-task).

    Forward expects:
        inputs = {"image": Tensor[B, image_dim], "gene": Tensor[B, n_genes]}
    Returns:
        {"survival": Tensor[B, 2]}

    Attention weights (image vs gene per patient) are cached after each
    forward pass and readable via get_attention_weights().
    """

    def __init__(self,
                 n_genes:     int,
                 image_dim:   int   = 1024,
                 fusion_type: str   = "attention",
                 dropout:     float = 0.3):
        super().__init__()

        self.image_encoder = SlideEmbeddingEncoder(image_dim, dropout)
        self.gene_encoder  = GeneEncoder(n_genes, dropout)
        self.fusion        = build_fusion(
            fusion_type,
            self.image_encoder.FEATURE_DIM,
            self.gene_encoder.FEATURE_DIM,
        )
        self.fusion_type   = fusion_type

        # Shared representation layer
        d = FUSED_DIM
        self.shared = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(inplace=True), nn.Dropout(dropout)
        )

        # Survival head (single task)
        self.head_survival = nn.Linear(d, 2)

        self._attn_weights: torch.Tensor | None = None

    def forward(self, inputs: dict) -> dict:
        img_feat  = self.image_encoder(inputs["image"])
        gene_feat = self.gene_encoder(inputs["gene"])

        fused, attn_w = self.fusion(img_feat, gene_feat)
        self._attn_weights = attn_w.detach()

        shared = self.shared(fused)

        return {"survival": self.head_survival(shared)}

    def get_attention_weights(self) -> torch.Tensor:
        """Returns [B, 2]: (image_weight, gene_weight) per sample."""
        if self._attn_weights is None:
            raise RuntimeError("Call forward() before get_attention_weights().")
        return self._attn_weights


# ── Factory ───────────────────────────────────────────────────────────────────

def build_model(n_genes:     int,
                image_dim:   int = 1024,
                fusion_type: str = "attention",
                dropout:     float = 0.3,
                verbose:     bool = True) -> JointModel:
    """
    Build and optionally print a model summary.

    Args:
        n_genes      : number of RNA-seq features after preprocessing
        image_dim    : slide embedding dim (1024=UNI, 1536=UNI2-h)
        fusion_type  : 'attention' | 'concat'
        dropout      : dropout rate throughout
        verbose      : print parameter counts
    """
    model = JointModel(n_genes, image_dim, fusion_type, dropout)

    if verbose:
        total     = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nJointModel  [fusion={fusion_type} | image_dim={image_dim} | "
              f"n_genes={n_genes}]")
        print(f"  SlideEmbeddingEncoder : "
              f"{sum(p.numel() for p in model.image_encoder.parameters()):>10,}")
        print(f"  GeneEncoder           : "
              f"{sum(p.numel() for p in model.gene_encoder.parameters()):>10,}")
        print(f"  Fusion ({fusion_type:9s})  : "
              f"{sum(p.numel() for p in model.fusion.parameters()):>10,}")
        print(f"  Shared + head         : "
              f"{sum(p.numel() for p in model.shared.parameters()) + sum(p.numel() for p in model.head_survival.parameters()):>10,}")
        print(f"  ─────────────────────────────────")
        print(f"  Total                 : {total:>10,}")
        print(f"  Trainable             : {trainable:>10,}")

    return model


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/Users/ankitbhardwaj/Documents/chaotic-multimodal-v2")

    B = 4

    configs = [
        dict(n_genes=20000, image_dim=1024, fusion_type="attention"),  # BRCA
        dict(n_genes=20000, image_dim=1536, fusion_type="attention"),  # LUAD/KIRC/STAD/LUSC/GBM
        dict(n_genes=20000, image_dim=1536, fusion_type="concat"),     # ablation
    ]

    for cfg in configs:
        model = build_model(**cfg)
        dummy = {
            "image": torch.randn(B, cfg["image_dim"]),
            "gene":  torch.randn(B, cfg["n_genes"]),
        }
        model.eval()
        with torch.no_grad():
            out = model(dummy)

        assert out["survival"].shape == (B, 2), "survival shape error"
        assert "subtype" not in out,            "subtype head should not exist"

        attn = model.get_attention_weights()
        assert attn.shape == (B, 2), "attention weights shape error"
        print(f"  survival={out['survival'].shape}  "
              f"attn=[{attn[0,0]:.2f}, {attn[0,1]:.2f}]  ✓\n")

    print("All JointModel self-tests PASSED.")