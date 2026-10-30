# Adaptive Gradient Modulation (AGM)

Official code for **"Adaptive Gradient Modulation for Multimodal Cancer Prognosis Prediction from Histopathology and Genomics"** (ML4BMI Workshop @ IEEE BIBM 2026).

## Overview

Multimodal cancer prognosis models that jointly train on histopathology images and genomic data suffer from modality imbalance: image-derived gradients dominate early training and suppress slower-evolving genomic representations. AGM addresses this with a **dual-gated** gradient intervention mechanism:

- **Gate 1 (imbalance):** Fires only when the contribution gap Δ exceeds a threshold θ
- **Gate 2 (trend):** Fires only when the weaker modality is *not* already self-correcting

When both gates pass, AGM applies **proportional suppression** scaled to the excess imbalance, with Gaussian noise injection for generalisation enhancement. This makes gradient modulation *conditional* rather than always-on, avoiding the regressions that static OGM-GE causes on naturally balanced cohorts.

## Key Results

Across 5 TCGA cohorts and 760+ training runs:
- **Adam+AGM** achieves the highest multi-seed average C-index (0.630) with the lowest cross-seed variance
- AGM never substantially degrades performance on any cohort — unlike static OGM-GE
- Optimizer choice alone varies C-index by up to 0.143 within a single cohort

## Repository Structure

```
agm-multimodal/
├── train.py                # Training script (Cox loss, C-index, all 14 configs)
├── models/
│   ├── joint_model.py      # Multimodal model (image + gene → survival)
│   ├── encoders.py         # Slide embedding encoder, gene expression encoder
│   └── fusion.py           # Cross-modal attention fusion
├── optimizers/
│   └── agm.py              # AGM wrapper (dual-gated adaptive gradient modulation)
└── configs/
    ├── brca.yaml           # Breast Invasive Carcinoma (UNI 1024-dim)
    ├── luad.yaml           # Lung Adenocarcinoma (UNI2-h 1536-dim)
    ├── kirc.yaml           # Kidney Renal Clear Cell Carcinoma
    ├── stad.yaml           # Stomach Adenocarcinoma
    └── lusc.yaml           # Lung Squamous Cell Carcinoma
```

## 14 Optimizer Configurations

| Group | Optimizers |
|-------|-----------|
| Baseline (no modulation) | SGD, SGD+Mom, Adam, Adadelta, Cosine |
| Always-on OGM-GE | SGD-GE, SGD+Mom-GE, Adam-GE, Adadelta-GE, Cosine-GE |
| **AGM (ours)** | Adam+AGM, SGD+Mom+AGM, Adadelta+AGM, Cosine+AGM |

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare data

Each cancer cohort requires:
- **Slide embeddings:** Pre-extracted UNI or UNI2-h features (`.pt` or `.h5` files named by TCGA barcode)
- **Gene expression CSV:** Columns `sample_id`, `label` (binary survival), `OS_months`, `vital_status`, plus ~60k gene expression columns

Update paths in `configs/*.yaml` to point to your data.

### 3. Train

```bash
# Single optimizer (AGM with Adam base)
python train.py --config configs/brca.yaml --optimizers adam_agm

# All 14 paper configurations
python train.py --config configs/brca.yaml --optimizers all

# Multi-seed evaluation
for seed in 42 123 456 789 999; do
    python train.py --config configs/brca.yaml --optimizers all \
        --seed $seed --save_dir results/brca/seed_$seed
done

# 5-fold cross-validation
for fold in 0 1 2 3 4; do
    python train.py --config configs/brca.yaml --optimizers all \
        --fold $fold --save_dir results/brca/fold_$fold
done
```

## AGM Usage (Standalone)

```python
from optimizers.agm import AGMWrapper

base_opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# AGM: adaptive dual-gated modulation (paper default)
agm = AGMWrapper(base_opt, model, modality_names=["image", "gene"],
                 imbalance_threshold=0.10, trend_window=10)

# Always-on OGM-GE baseline (no gating)
ogm = AGMWrapper(base_opt, model, modality_names=["image", "gene"],
                 imbalance_threshold=0.0)

# Training step
for inputs, labels in loader:
    out  = model(inputs)
    loss = criterion(out, labels)
    agm.step(loss, inputs, labels)
```

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `imbalance_threshold` (θ) | 0.10 | Minimum contribution gap to trigger suppression |
| `trend_window` (τ) | 10 | Steps for weaker-modality trend estimation |
| `alpha` (α) | 0.5 | Suppression sensitivity |

## Citation

```bibtex
@inproceedings{bhardwaj2026agm,
  title={Adaptive Gradient Modulation for Multimodal Cancer Prognosis Prediction from Histopathology and Genomics},
  author={Bhardwaj, Ankit and Nabavi, Sheida},
  booktitle={Proc. IEEE BIBM Workshop on Machine Learning for Biomedical Informatics (ML4BMI)},
  year={2026}
}
```

## License

MIT
