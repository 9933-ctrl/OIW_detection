from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, rotate as scipy_rotate


def normalize_patch(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Z-score normalize a single 2-D patch."""
    arr = arr.astype(np.float32)
    return (arr - float(arr.mean())) / (float(arr.std()) + eps)


def high_pass_filter(ssh: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    """Remove broad background variability by subtracting a Gaussian-smoothed field."""
    low = gaussian_filter(ssh.astype(np.float32), sigma=sigma)
    return ssh.astype(np.float32) - low.astype(np.float32)


def gradient_magnitude(ssh: np.ndarray) -> np.ndarray:
    """Compute local gradient magnitude used as the third input channel."""
    gy, gx = np.gradient(ssh.astype(np.float32))
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def build_three_channel_patch(ssh: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    """Return a 3xHxW tensor: SSH anomaly, high-pass SSH, local gradient magnitude."""
    ssh0 = normalize_patch(ssh)
    hp = normalize_patch(high_pass_filter(ssh0, sigma=sigma))
    grad = normalize_patch(gradient_magnitude(ssh0))
    return np.stack([ssh0, hp, grad], axis=0).astype(np.float32)


def add_noise_at_snr(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Inject additive Gaussian noise at a target SNR level."""
    signal_power = float(np.mean(x ** 2)) + 1e-12
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, math.sqrt(noise_power), size=x.shape).astype(np.float32)
    return (x + noise).astype(np.float32)


def add_salt_and_pepper(x: np.ndarray, amount: float, rng: np.random.Generator) -> np.ndarray:
    """Apply sparse impulse noise for robustness augmentation."""
    out = x.copy()
    mask = rng.random(x.shape) < amount
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=x.shape)
    out[mask] = signs[mask] * max(1.0, float(np.max(np.abs(x))))
    return out.astype(np.float32)


def add_speckle_noise(x: np.ndarray, scale: float, rng: np.random.Generator) -> np.ndarray:
    """Apply multiplicative speckle-like noise."""
    return (x + x * rng.normal(0.0, scale, size=x.shape).astype(np.float32)).astype(np.float32)


def random_small_rotation(x: np.ndarray, degrees: Tuple[float, float], rng: np.random.Generator) -> np.ndarray:
    """Rotate each channel by a small angle while preserving patch size."""
    angle = float(rng.uniform(degrees[0], degrees[1]))
    rotated = [scipy_rotate(c, angle=angle, reshape=False, order=1, mode="nearest") for c in x]
    return np.stack(rotated, axis=0).astype(np.float32)
