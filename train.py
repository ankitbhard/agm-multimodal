"""
train.py — Cox PH survival training for the AGM paper.

Self-contained training script supporting all 14 optimizer configurations
from the paper (5 baselines + 5 OGM-GE + 4 AGM).

Loss:   Cox partial negative log-likelihood (handles censoring)
Metric: Harrell's C-index

Usage:
    # Single optimizer
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
"""

import os, sys, json, random, argparse
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from models.joint_model import build_model
from optimizers.agm import AGMWrapper

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


# ═════════════════════════════════════════════════════════════════════════════
# Optimizer configurations (14 total, matching Table 2 of the paper)
# ═════════════════════════════════════════════════════════════════════════════

OPTIMIZER_NAMES = [
    # Baseline (no gradient modulation)
    "sgd", "sgd_mom", "adam", "adadelta", "cosine",
    # Always-on OGM-GE (Pan et al. CVPR 2022)
    "sgd_ge", "sgd_mom_ge", "adam_ge", "adadelta_ge", "cosine_ge",
    # AGM (ours): adaptive dual-gated modulation
    "adam_agm", "sgd_mom_agm", "adadelta_agm", "cosine_agm",
]


def build_optimizer(name: str, model, cfg: dict):
    """
    Build optimizer, optional scheduler, and chaotic flag.

    Returns:
        (optimizer_or_wrapper, scheduler_or_None, is_wrapped)

    is_wrapped=True means the optimizer handles backward() internally
    via wrapper.step(loss, inputs, labels).
    """
    lr = cfg["lr"]
    wd = cfg["weight_decay"]
    epochs = cfg["epochs"]

    # ── Baselines ────────────────────────────────────────────────────────────

    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd), None, False

    if name == "sgd_mom":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd), None, False

    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd), None, False

    if name == "adadelta":
        return torch.optim.Adadelta(model.parameters(), weight_decay=wd), None, False

    if name == "cosine":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        return opt, sch, False

    # ── Always-on OGM-GE (imbalance_threshold=0.0 → no gating) ──────────────

    if name == "sgd_ge":
        base = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
        return AGMWrapper(base, model, ["image", "gene"], alpha=0.5, imbalance_threshold=0.0), None, True

    if name == "sgd_mom_ge":
        base = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        return AGMWrapper(base, model, ["image", "gene"], alpha=0.5, imbalance_threshold=0.0), None, True

    if name == "adam_ge":
        base = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        return AGMWrapper(base, model, ["image", "gene"], alpha=0.5, imbalance_threshold=0.0), None, True

    if name == "adadelta_ge":
        base = torch.optim.Adadelta(model.parameters(), weight_decay=wd)
        return AGMWrapper(base, model, ["image", "gene"], alpha=0.5, imbalance_threshold=0.0), None, True

    if name == "cosine_ge":
        base = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(base, T_max=epochs)
        return AGMWrapper(base, model, ["image", "gene"], alpha=0.5, imbalance_threshold=0.0), sch, True

    # ── AGM (ours): adaptive dual-gated modulation ───────────────────────────

    if name == "adam_agm":
        base = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        return AGMWrapper(base, model, ["image", "gene"], alpha=0.5,
                          imbalance_threshold=0.10, trend_window=10), None, True

    if name == "sgd_mom_agm":
        base = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        return AGMWrapper(base, model, ["image", "gene"], alpha=0.5,
                          imbalance_threshold=0.10, trend_window=10), None, True

    if name == "adadelta_agm":
        base = torch.optim.Adadelta(model.parameters(), weight_decay=wd)
        return AGMWrapper(base, model, ["image", "gene"], alpha=0.5,
                          imbalance_threshold=0.10, trend_window=10), None, True

    if name == "cosine_agm":
        base = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(base, T_max=epochs)
        return AGMWrapper(base, model, ["image", "gene"], alpha=0.5,
                          imbalance_threshold=0.10, trend_window=10), sch, True

    raise ValueError(f"Unknown optimizer: '{name}'. Choose from {OPTIMIZER_NAMES}")


# ═════════════════════════════════════════════════════════════════════════════
# C-index and Cox loss
# ═════════════════════════════════════════════════════════════════════════════

def concordance_index(times, risk_scores, events):
    """Harrell's C-index. Higher risk_score → predicted to die sooner."""
    times       = np.asarray(times,       dtype=float)
    risk_scores = np.asarray(risk_scores, dtype=float)
    events      = np.asarray(events,      dtype=float)

    concordant, permissible = 0.0, 0.0
    for i in range(len(times)):
        if events[i] != 1:
            continue
        for j in range(len(times)):
            if times[j] > times[i]:
                permissible += 1
                if risk_scores[i] > risk_scores[j]:
                    concordant += 1
                elif risk_scores[i] == risk_scores[j]:
                    concordant += 0.5
    return concordant / permissible if permissible > 0 else 0.5


def cox_loss(risk_scores, os_months, vital_status):
    """Negative Cox partial log-likelihood."""
    order       = torch.argsort(os_months, descending=True)
    risk        = risk_scores[order]
    events      = vital_status[order].float()
    log_cum_sum = torch.logcumsumexp(risk, dim=0)
    return -torch.mean((risk - log_cum_sum) * events)


# ═════════════════════════════════════════════════════════════════════════════
# Dataset
# ═════════════════════════════════════════════════════════════════════════════

def _load_embedding(path, image_dim):
    if not path or not os.path.exists(path):
        return torch.zeros(image_dim)
    if path.endswith(".pt"):
        t = torch.load(path, map_location="cpu", weights_only=True)
        if t.dim() > 1:
            t = t.mean(0)
        return t.float()
    import h5py
    try:
        with h5py.File(path, "r") as f:
            key = "features" if "features" in f else list(f.keys())[0]
            arr = f[key][()]
        t = torch.tensor(arr, dtype=torch.float32)
        t = t.reshape(-1, t.shape[-1]).mean(0)
        return t
    except Exception as e:
        print(f"  [WARN] skipping bad h5 file {os.path.basename(path)}: {e}")
        return torch.zeros(image_dim)


class SurvivalDataset(Dataset):
    def __init__(self, records, gene_matrix, image_dim, emb_cache=None):
        self.records     = records
        self.gene_matrix = gene_matrix
        self.image_dim   = image_dim
        self._cache      = emb_cache if emb_cache is not None else {}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec   = self.records[idx]
        genes = torch.tensor(self.gene_matrix[idx], dtype=torch.float32)
        path  = rec["uni_path"]
        if path not in self._cache:
            self._cache[path] = _load_embedding(path, self.image_dim)
        image = self._cache[path]
        return (
            {"image": image, "gene": genes},
            {
                "survival":     rec["surv_label"],
                "os_months":    rec["os_months"],
                "vital_status": rec["vital_status"],
            },
        )


def _discover_uni_dir(uni_dir):
    for subdir in ["uni_features/tcga_features", "uni_features", ""]:
        candidate = os.path.join(uni_dir, subdir) if subdir else uni_dir
        if os.path.isdir(candidate):
            files = [f for f in os.listdir(candidate) if f.endswith((".pt", ".h5"))]
            if files:
                return candidate
    return uni_dir


def build_dataloaders(uni_dir, gene_csv_paths, image_dim=1024,
                      n_top_genes=20000, val_fraction=0.15, test_fraction=0.10,
                      batch_size=32, seed=42, fold=None, n_folds=5):
    frames = [pd.read_csv(p) for p in gene_csv_paths if os.path.exists(p)]
    if not frames:
        raise FileNotFoundError(f"No gene CSVs: {gene_csv_paths}")
    df = pd.concat(frames, ignore_index=True)
    df["patient_barcode"] = df["sample_id"].str[:12]
    df = df.drop_duplicates(subset="patient_barcode").reset_index(drop=True)
    df = df.dropna(subset=["OS_months", "vital_status"]).reset_index(drop=True)

    emb_files = {}
    try:
        emb_dir = _discover_uni_dir(uni_dir)
        for fname in os.listdir(emb_dir):
            if fname.endswith((".pt", ".h5")):
                emb_files[fname[:12]] = os.path.join(emb_dir, fname)
    except (FileNotFoundError, OSError):
        pass

    NON_GENE = {
        "sample_id", "patient_barcode", "label", "survival_label",
        "subtype_label", "uni_path", "image_path", "OS_months",
        "vital_status", "subtype_raw", "patient_id", "magnification",
    }
    gene_cols = [c for c in df.columns
                 if c not in NON_GENE and pd.api.types.is_numeric_dtype(df[c])]
    surv_col = "survival_label" if "survival_label" in df.columns else "label"

    records, gene_rows = [], []
    patient_iter = emb_files.items() if emb_files else (
        (bc, "") for bc in df["patient_barcode"])
    for barcode, fpath in patient_iter:
        row = df[df["patient_barcode"] == barcode]
        if row.empty:
            continue
        row = row.iloc[0]
        records.append({
            "uni_path": fpath, "patient_barcode": barcode,
            "surv_label": int(row.get(surv_col, 0)),
            "os_months": float(row["OS_months"]),
            "vital_status": int(row["vital_status"]),
        })
        gene_rows.append(row[gene_cols].infer_objects(copy=False).fillna(0).values.astype(np.float32))

    print(f"  Matched patients: {len(records)}  "
          f"(events={sum(r['vital_status'] for r in records)})")

    if fold is not None:
        from sklearn.model_selection import StratifiedKFold
        labels = [r["vital_status"] for r in records]
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = list(skf.split(range(len(records)), labels))
        trainval_idx, test_idx = splits[fold]
        trainval_idx, test_idx = list(trainval_idx), list(test_idx)
        rng = random.Random(seed)
        rng.shuffle(trainval_idx)
        n_val = int(len(trainval_idx) * 0.15)
        val_idx, train_idx = trainval_idx[:n_val], trainval_idx[n_val:]
        print(f"  Fold {fold}/{n_folds}: {len(train_idx)} train / "
              f"{len(val_idx)} val / {len(test_idx)} test")
    else:
        rng = random.Random(seed)
        indices = list(range(len(records)))
        rng.shuffle(indices)
        n_test = int(len(indices) * test_fraction)
        n_val  = int(len(indices) * val_fraction)
        test_idx  = indices[:n_test]
        val_idx   = indices[n_test:n_test + n_val]
        train_idx = indices[n_test + n_val:]

    gene_matrix = np.stack(gene_rows, axis=0)
    var_idx     = np.argsort(gene_matrix[train_idx].var(axis=0))[::-1][:n_top_genes]
    scaler      = StandardScaler()
    train_genes = scaler.fit_transform(gene_matrix[train_idx][:, var_idx])
    val_genes   = scaler.transform(gene_matrix[val_idx][:, var_idx])
    test_genes  = scaler.transform(gene_matrix[test_idx][:, var_idx])

    def _subset(idx_list, gene_data):
        return [records[i] for i in idx_list], gene_data

    train_recs, train_genes = _subset(train_idx, train_genes)
    val_recs,   val_genes   = _subset(val_idx,   val_genes)
    test_recs,  test_genes  = _subset(test_idx,  test_genes)
    n_genes = train_genes.shape[1]

    print(f"  Split: {len(train_recs)} / {len(val_recs)} / {len(test_recs)}  "
          f"n_genes={n_genes}")

    emb_cache: Dict[str, torch.Tensor] = {}

    def _loader(recs, genes, shuffle):
        ds = SurvivalDataset(recs, genes, image_dim, emb_cache)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=False)

    return {
        "train": _loader(train_recs, train_genes, shuffle=True),
        "val":   _loader(val_recs,   val_genes,   shuffle=False),
        "test":  _loader(test_recs,  test_genes,  shuffle=False),
        "n_genes": n_genes, "image_dim": image_dim, "test_recs": test_recs,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Training loop
# ═════════════════════════════════════════════════════════════════════════════

def _to_tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(x, dtype=torch.long, device=device)

def _to_float(x, device):
    if isinstance(x, torch.Tensor):
        return x.float().to(device)
    return torch.tensor(x, dtype=torch.float32, device=device)


def train_one_epoch(model, loader, optimizer, device, is_wrapped):
    model.train()
    total_loss = 0.0
    for inputs, labels in loader:
        img  = inputs["image"].to(device)
        gene = inputs["gene"].to(device)
        os_m = _to_float(labels["os_months"], device)
        vs   = _to_float(labels["vital_status"], device)
        inp  = {"image": img, "gene": gene}

        if is_wrapped:
            outputs = model(inp)
            risk = outputs["survival"][:, 1]
            loss = cox_loss(risk, os_m, vs)
            labels_dev = {"survival": _to_tensor(labels["survival"], device),
                          "os_months": os_m, "vital_status": vs}
            optimizer.step(loss, inp, labels_dev)
        else:
            optimizer.zero_grad()
            outputs = model(inp)
            risk = outputs["survival"][:, 1]
            loss = cox_loss(risk, os_m, vs)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_risk, all_time, all_event = [], [], []
    total_loss = 0.0
    for inputs, labels in loader:
        img  = inputs["image"].to(device)
        gene = inputs["gene"].to(device)
        os_m = _to_float(labels["os_months"], device)
        vs   = _to_float(labels["vital_status"], device)
        outputs = model({"image": img, "gene": gene})
        risk = outputs["survival"][:, 1]
        loss = cox_loss(risk, os_m, vs)
        total_loss += loss.item()
        all_risk.extend(risk.cpu().numpy().tolist())
        all_time.extend(os_m.cpu().numpy().tolist())
        all_event.extend(vs.cpu().numpy().tolist())
    cindex = concordance_index(all_time, all_risk, all_event)
    return total_loss / max(len(loader), 1), cindex, all_risk, all_time, all_event


def train_one_optimizer(opt_name, model_factory, loaders, cfg, device):
    model = model_factory().to(device)
    opt, scheduler, is_wrapped = build_optimizer(opt_name, model, cfg)

    epochs   = cfg["epochs"]
    save_dir = cfg["save_dir"]
    history  = {"train_loss": [], "val_loss": [], "val_cindex": []}
    best_val_cindex, best_epoch = -1.0, 0

    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, loaders["train"], opt, device, is_wrapped)
        val_loss, val_ci, _, _, _ = evaluate(model, loaders["val"], device)
        if scheduler:
            scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_cindex"].append(val_ci)

        if val_ci > best_val_cindex:
            best_val_cindex = val_ci
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(save_dir, f"{opt_name}_best.pt"))

        if epoch % 20 == 0 or epoch == epochs:
            print(f"    ep {epoch:3d}/{epochs}  tr_loss={tr_loss:.4f}  "
                  f"val_ci={val_ci:.4f}  (best={best_val_cindex:.4f} @ ep{best_epoch})")

    # Test with best checkpoint
    ckpt = os.path.join(save_dir, f"{opt_name}_best.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    test_loss, test_ci, risk_scores, _, _ = evaluate(model, loaders["test"], device)
    print(f"  {opt_name}: test C-index = {test_ci:.4f}")

    history["test_cindex"]     = test_ci
    history["test_loss"]       = test_loss
    history["best_val_cindex"] = best_val_cindex
    history["best_epoch"]      = best_epoch
    history["test_predictions"] = [
        {"patient_barcode": rec["patient_barcode"],
         "os_months": rec["os_months"], "vital_status": rec["vital_status"],
         "risk_score": float(risk_scores[i])}
        for i, rec in enumerate(loaders["test_recs"])
    ]

    with open(os.path.join(save_dir, f"{opt_name}_survival.json"), "w") as f:
        json.dump(history, f, indent=2)

    if os.path.exists(ckpt):
        os.remove(ckpt)
    return history


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

def main(cfg, opt_names):
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    loaders = build_dataloaders(
        uni_dir=cfg["uni_dir"], gene_csv_paths=cfg["gene_csv_paths"],
        image_dim=cfg["image_dim"], n_top_genes=cfg.get("n_top_genes", 20000),
        val_fraction=cfg.get("val_fraction", 0.15),
        test_fraction=cfg.get("test_fraction", 0.10),
        batch_size=cfg.get("batch_size", 32), seed=seed,
        fold=cfg.get("fold", None), n_folds=cfg.get("n_folds", 5),
    )

    def model_factory():
        return build_model(
            n_genes=loaders["n_genes"], image_dim=loaders["image_dim"],
            fusion_type=cfg.get("fusion_type", "attention"),
            dropout=cfg.get("dropout", 0.3),
        )

    save_dir = cfg.get("save_dir", "results")
    os.makedirs(save_dir, exist_ok=True)
    cfg["save_dir"] = save_dir

    for opt_name in opt_names:
        out_path = os.path.join(save_dir, f"{opt_name}_survival.json")
        if os.path.exists(out_path):
            with open(out_path) as f:
                h = json.load(f)
            if "test_cindex" in h:
                print(f"  [SKIP] {opt_name} — test_cindex={h['test_cindex']:.4f}")
                continue
        print(f"\n{'='*60}\n  {opt_name}\n{'='*60}")
        train_one_optimizer(opt_name, model_factory, loaders, cfg, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AGM: Adaptive Gradient Modulation")
    parser.add_argument("--config",     required=True, help="Path to cancer config YAML")
    parser.add_argument("--optimizers", nargs="+", default=["adam_agm"],
                        help="Optimizer names or 'all' for all 14")
    parser.add_argument("--save_dir",   default=None)
    parser.add_argument("--epochs",     type=int, default=None)
    parser.add_argument("--seed",       type=int, default=None)
    parser.add_argument("--fold",       type=int, default=None)
    parser.add_argument("--n_folds",    type=int, default=5)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.save_dir is not None: cfg["save_dir"] = args.save_dir
    if args.epochs   is not None: cfg["epochs"]   = args.epochs
    if args.seed     is not None: cfg["seed"]     = args.seed
    if args.fold     is not None: cfg["fold"]     = args.fold
    cfg["n_folds"] = args.n_folds

    opt_names = OPTIMIZER_NAMES if "all" in args.optimizers else args.optimizers
    main(cfg, opt_names)
