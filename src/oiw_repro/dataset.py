from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocess import build_three_channel_patch


class NPZPatchDataset(Dataset):
    """Dataset reader for preprocessed OIW patch .npz files."""

    def __init__(self, npz_path: str | Path):
        p = Path(npz_path)
        if not p.exists():
            raise FileNotFoundError(f"Missing dataset: {p}")
        data = np.load(p, allow_pickle=True)
        if "x" not in data or "y" not in data:
            raise KeyError("The npz file must contain arrays named 'x' and 'y'.")
        self.x = data["x"].astype(np.float32)
        self.y = data["y"].astype(np.float32)
        if self.x.ndim != 4:
            raise ValueError("x must have shape (N, C, H, W).")
        if self.y.ndim != 1 or self.y.shape[0] != self.x.shape[0]:
            raise ValueError("y must have shape (N,) and match x.shape[0].")

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.x[idx]), torch.tensor(self.y[idx], dtype=torch.float32)


def _diagonal_stripe_field(size: int, rng: np.random.Generator, positive: bool) -> np.ndarray:
    """Generate deterministic OIW-like or hard-negative SSH fields for smoke testing."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    theta = float(rng.uniform(-0.9, 0.9))
    freq = float(rng.uniform(0.22, 0.38))
    phase = float(rng.uniform(0, 2*np.pi))
    u = np.cos(theta) * xx + np.sin(theta) * yy
    v = -np.sin(theta) * xx + np.cos(theta) * yy
    envelope = np.exp(-((v - size * rng.uniform(0.35, 0.65)) ** 2) / (2 * (size * rng.uniform(0.12, 0.22)) ** 2))
    noise = rng.normal(0, 0.18, size=(size, size)).astype(np.float32)
    background = 0.25 * np.sin(xx / size * np.pi * rng.uniform(1.0, 3.0)) + 0.15 * np.cos(yy / size * np.pi * rng.uniform(1.0, 2.0))
    if positive:
        stripe = np.sin(freq * u + phase) * envelope
        field = background + 1.35 * stripe + noise
    else:
        # Hard negatives: wind-streak-like broad bands or ship-wake-like diagonal artefacts.
        wind = 0.65 * np.sin(freq * xx + phase) if rng.random() < 0.5 else 0.65 * np.sin(freq * yy + phase)
        wake = 0.9 * np.exp(-((v - size * rng.uniform(0.15, 0.85)) ** 2) / (2 * (size * 0.025) ** 2))
        field = background + (wind if rng.random() < 0.65 else wake) + noise
    return field.astype(np.float32)


def create_synthetic_npz(out_path: str | Path, n_samples: int = 160, size: int = 64, seed: int = 42) -> Path:
    """Create a deterministic synthetic OIW-like dataset for code-path verification.

    This is not a substitute for the original SWOT/SAR dataset. It only supports a runnable
    smoke test and log generation when the real data are unavailable.
    """
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for i in range(n_samples):
        positive = i < n_samples // 2
        field = _diagonal_stripe_field(size=size, rng=rng, positive=positive)
        xs.append(build_three_channel_patch(field, sigma=max(1.0, size / 16.0)))
        ys.append(1 if positive else 0)
    idx = rng.permutation(n_samples)
    x = np.stack(xs, axis=0)[idx].astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)[idx]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, x=x, y=y, seed=np.array(seed), generator="synthetic_smoke_test")
    return out_path
