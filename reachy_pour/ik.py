"""Damped least-squares inverse kinematics for the local Reachy arm model."""

from __future__ import annotations

import numpy as np

from reachy_pour.joints import limits_for
from reachy_pour.kinematics import tip_position, tip_rotation

_EPS_DEG = 0.25


def position_ik(
    arm: str,
    target_m: np.ndarray,
    q0_deg: np.ndarray,
    *,
    lock: dict[int, float] | None = None,
    up_axis: int | None = None,
    up_target: np.ndarray | None = None,
    up_weight: float = 0.25,
    max_iter: int = 40,
    tolerance_m: float = 0.008,
    damping: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Solve joint degrees that put the gripper tip on ``target_m``.

    ``lock`` forces joints (by index into the 7-DoF chain) to a fixed angle.
    ``up_axis`` asks a column of the tip rotation to point along ``up_target``
    (a soft constraint, so position still wins). The returned error is the tip
    position error in meters, ignoring orientation.
    """
    q = np.asarray(q0_deg, dtype=np.float64).reshape(7).copy()
    lock = lock or {}
    for index, value in lock.items():
        q[index] = value
    limits = limits_for(arm)
    best_q = q.copy()
    best_err = float("inf")
    target = np.asarray(target_m, dtype=np.float64).reshape(3)
    desired = None if up_target is None else np.asarray(up_target, dtype=np.float64)
    if desired is not None:
        desired = desired / np.linalg.norm(desired)
    use_up = up_axis is not None and desired is not None

    for _ in range(max_iter):
        pos = tip_position(arm, q)
        pos_err = target - pos
        norm = float(np.linalg.norm(pos_err))
        if norm < best_err:
            best_err = norm
            best_q = q.copy()
        aligned = True
        if use_up:
            column = tip_rotation(arm, q)[:, up_axis]
            aligned = float(np.dot(column, desired)) > 0.92
        if norm < tolerance_m and aligned:
            break
        rows = 6 if use_up else 3
        jac = np.zeros((rows, 7), dtype=np.float64)
        if use_up:
            column = tip_rotation(arm, q)[:, up_axis]
            ori_err = up_weight * (desired - column)
            err = np.concatenate([pos_err, ori_err])
        else:
            err = pos_err
        for index in range(7):
            if index in lock:
                continue
            step = q.copy()
            step[index] += _EPS_DEG
            jac[:3, index] = (tip_position(arm, step) - pos) / _EPS_DEG
            if use_up:
                stepped = tip_rotation(arm, step)[:, up_axis]
                jac[3:, index] = up_weight * (stepped - column) / _EPS_DEG
        lhs = jac @ jac.T + (damping * 1e-4) * np.eye(rows)
        dq = jac.T @ np.linalg.solve(lhs, err)
        dq_norm = float(np.linalg.norm(dq))
        if dq_norm > 12.0:
            dq *= 12.0 / dq_norm
        q = q + dq
        for index, value in lock.items():
            q[index] = value
        q = np.minimum(np.maximum(q, limits[:, 0]), limits[:, 1])

    return best_q, best_err
