from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Subset

from .dataset import NPZPatchDataset, create_synthetic_npz
from .model import build_model_from_config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_indices(n: int, train_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(round(n * train_fraction))
    return idx[:cut], idx[cut:]


def compute_metrics(y_true: np.ndarray, logits: np.ndarray) -> Dict[str, float]:
    probs = 1.0 / (1.0 + np.exp(-logits))
    pred = (probs >= 0.5).astype(np.int64)
    acc = accuracy_score(y_true, pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_true, probs)
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
    }


def run_epoch(model, loader, device, criterion, optimizer=None, grad_clip: float = 1.0):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    ys, logits_all = [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        if is_train:
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        total_loss += float(loss.detach().cpu()) * int(x.size(0))
        ys.append(y.detach().cpu().numpy())
        logits_all.append(logits.detach().cpu().numpy())
    y_true = np.concatenate(ys)
    logits_np = np.concatenate(logits_all)
    metrics = compute_metrics(y_true, logits_np)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def train_main(config_path: str | Path, synthetic_demo: bool = False) -> dict:
    cfg = load_config(config_path)
    seed = int(cfg.get("training", {}).get("seed", 42))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if synthetic_demo:
        dataset_path = Path(cfg["data_paths"]["demo_npz"])
        create_synthetic_npz(dataset_path, n_samples=160, size=int(cfg["input"].get("patch_size", 64)), seed=seed)
    else:
        dataset_path = Path(cfg["data_paths"].get("real_npz_train", ""))
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Real dataset not found at {dataset_path}. Use --synthetic-demo for a code-path check, "
                "or update data_paths.real_npz_train in the config."
            )

    ds = NPZPatchDataset(dataset_path)
    train_idx, val_idx = split_indices(len(ds), float(cfg["training"].get("train_fraction", 0.8)), seed)
    batch_size = int(cfg["training"].get("batch_size", 64))
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model_from_config(cfg).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"].get("learning_rate", 1e-3)),
        weight_decay=float(cfg["training"].get("weight_decay", 1e-4)),
    )
    epochs = int(cfg["training"].get("epochs", 100))
    grad_clip = float(cfg["training"].get("gradient_clipping", 1.0))

    out_csv = Path(cfg["output_paths"].get("metrics_csv", "outputs/train_metrics.csv"))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(cfg["output_paths"].get("checkpoint", "outputs/best_model.pt"))
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    final_json = Path(cfg["output_paths"].get("final_metrics_json", "outputs/final_metrics.json"))

    print("=== OIW hybrid CNN--Transformer training ===")
    print(f"config={config_path}")
    print(f"dataset={dataset_path} samples={len(ds)} train={len(train_idx)} val={len(val_idx)}")
    print(f"device={device} torch={torch.__version__}")
    print(f"model_parameters={sum(p.numel() for p in model.parameters())}")
    print(f"epochs={epochs} batch_size={batch_size} seed={seed}")

    best_f1 = -1.0
    rows = []
    t0 = time.time()
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "split", "loss", "accuracy", "precision", "recall", "f1", "auc"])
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            train_metrics = run_epoch(model, train_loader, device, criterion, optimizer, grad_clip)
            with torch.no_grad():
                val_metrics = run_epoch(model, val_loader, device, criterion, optimizer=None, grad_clip=0)
            for split, m in [("train", train_metrics), ("validation", val_metrics)]:
                row = {"epoch": epoch, "split": split, **m}
                writer.writerow(row)
                rows.append(row)
            print(
                f"epoch={epoch:03d} "
                f"train_loss={train_metrics['loss']:.4f} train_f1={train_metrics['f1']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
                f"val_precision={val_metrics['precision']:.4f} val_recall={val_metrics['recall']:.4f} "
                f"val_f1={val_metrics['f1']:.4f} val_auc={val_metrics['auc']:.4f}"
            )
            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                torch.save({"model_state": model.state_dict(), "config": cfg, "epoch": epoch, "val_metrics": val_metrics}, checkpoint_path)
                print(f"saved_checkpoint={checkpoint_path} best_val_f1={best_f1:.4f}")

    elapsed = time.time() - t0
    final = {
        "config": str(config_path),
        "dataset": str(dataset_path),
        "synthetic_demo": bool(synthetic_demo),
        "device": str(device),
        "torch_version": torch.__version__,
        "model_parameters": int(sum(p.numel() for p in model.parameters())),
        "epochs": epochs,
        "best_validation_f1": float(best_f1),
        "elapsed_seconds": round(float(elapsed), 3),
        "metrics_csv": str(out_csv),
        "checkpoint": str(checkpoint_path),
        "note": "Synthetic demo logs verify the code path only; use original SWOT/SAR data for editorial validation."
    }
    final_json.write_text(json.dumps(final, indent=2), encoding="utf-8")
    print("final_metrics_json=" + str(final_json))
    print(json.dumps(final, indent=2))
    return final


def main():
    parser = argparse.ArgumentParser(description="Train the OIW hybrid CNN--Transformer detector.")
    parser.add_argument("--config", default="configs/method_config_article.json")
    parser.add_argument("--synthetic-demo", action="store_true", help="Run a deterministic synthetic smoke test.")
    args = parser.parse_args()
    train_main(args.config, synthetic_demo=args.synthetic_demo)


if __name__ == "__main__":
    main()
