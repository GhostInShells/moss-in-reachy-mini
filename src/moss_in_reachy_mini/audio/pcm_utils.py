from __future__ import annotations

from math import gcd

import numpy as np
from scipy import signal


def safe_int(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except Exception:
        return None


def ensure_2d_f32(data: np.ndarray, *, allow_transpose: bool = True) -> np.ndarray:
    """Normalize PCM ndarray to float32 2D (frames, channels).

    - Accepts 1D (frames,) and returns (frames, 1)
    - Accepts 2D (frames, channels)
    - Optionally auto-transposes a channels-first layout (channels, frames)
      when it looks unambiguous (small first dim, bigger second dim).
    """

    arr = np.asarray(data)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim == 2:
        if allow_transpose and arr.shape[0] <= 8 and arr.shape[0] < arr.shape[1]:
            return arr.T
        return arr
    raise ValueError(f"Unsupported audio ndarray shape: {arr.shape}")


def ensure_channels(
    data: np.ndarray,
    channels: int,
    *,
    default_channels: int = 2,
    allow_transpose: bool = True,
) -> np.ndarray:
    """Ensure ndarray is (frames, channels) with desired channel count."""

    arr = ensure_2d_f32(data, allow_transpose=allow_transpose)
    if channels <= 0:
        channels = int(default_channels)
    channels = max(1, int(channels))
    if arr.shape[1] == channels:
        return np.ascontiguousarray(arr)
    return np.ascontiguousarray(np.tile(arr[:, [0]], (1, channels)).astype(np.float32, copy=False))


def resample_f32(
    data: np.ndarray,
    *,
    origin_rate: int,
    target_rate: int,
    allow_transpose: bool = True,
) -> np.ndarray:
    """Resample float32 PCM to a new sample rate.

    Returns float32 PCM shaped (frames, channels).
    """

    arr = ensure_2d_f32(data, allow_transpose=allow_transpose)
    if origin_rate == target_rate:
        return arr

    if origin_rate <= 0 or target_rate <= 0:
        raise ValueError(f"Invalid sample rate: origin_rate={origin_rate} target_rate={target_rate}")

    g = gcd(int(origin_rate), int(target_rate))
    up = int(target_rate // g)
    down = int(origin_rate // g)
    if arr.shape[0] == 0:
        return np.zeros((0, arr.shape[1]), dtype=np.float32)

    out = signal.resample_poly(arr, up=up, down=down, axis=0)
    return out.astype(np.float32, copy=False)

