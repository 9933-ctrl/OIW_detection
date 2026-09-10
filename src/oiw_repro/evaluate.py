from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import NPZPatchDataset
from .model import build_model_from_config
from .train import compute_metrics


def evaluate(config_path: str, checkpoint: str, dataset_path: str | None = None, out_json: str | None = None):
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    ckpt = torch.load(checkpoint, map_location="cpu")
    model = build_model_from_config(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    if dataset_path is None:
        dataset_path = cfg["data_paths"].get("real_npz_test") or cfg["data_paths"].get("demo_npz")
    ds = NPZPatchDataset(dataset_path)
    loader = DataLoader(ds, batch_size=int(cfg["training"].get("batch_size", 64)), shuffle=False, num_workers=0)
    ys, logits_all = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            ys.append(y.numpy())
            logits_all.append(logits.numpy())
    metrics = compute_metrics(np.concatenate(ys), np.concatenate(logits_all))
    result = {
        "config": config_path,
        "checkpoint": checkpoint,
        "dataset": dataset_path,
        "num_samples": len(ds),
        "metrics": metrics,
    }
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate the OIW detector from a checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint, args.dataset, args.out_json)


if __name__ == "__main__":
    main()
