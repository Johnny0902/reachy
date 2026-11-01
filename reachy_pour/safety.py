"""Joint-command guards used whenever this process is allowed to move the arm."""

from __future__ import annotations

import numpy as np


def clip_limits(values: np.ndarray, limits: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(values, limits[:, 0]), limits[:, 1])


def clip_relative(goal: np.ndarray, present: np.ndarray, max_delta: float | None) -> np.ndarray:
    """Limit how far a command may move from the measured position in one cycle."""
    if max_delta is None:
        return goal
    delta = np.clip(goal - present, -max_delta, max_delta)
    return present + delta


def in_workspace(point: np.ndarray, low: np.ndarray, high: np.ndarray) -> bool:
    point = np.asarray(point, dtype=np.float64)
    return bool(np.all(point >= low) and np.all(point <= high))
