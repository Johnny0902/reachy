"""Map a Meta Quest 2 controller pose into the Reachy frame.

Unity / OpenXR space is X right, Y up, Z forward, meters, origin at the tracking
space. Reachy space is X forward, Y to the robot's left, Z up, origin between
the shoulders. Calibration stores the headset pose at the moment A is pressed
and removes that yaw, which is the same idea as the official teleoperation app:
do not move your feet after pressing A.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from reachy_pour.bench import Bench
from reachy_pour.safety import in_workspace


def quat_to_matrix(xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = [float(v) for v in xyzw]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def yaw_about_y(rotation: np.ndarray) -> float:
    forward = rotation @ np.array([0.0, 0.0, 1.0])
    return float(np.arctan2(forward[0], forward[2]))


def _yaw_matrix(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


# Unity (x right, y up, z forward) -> Reachy (x forward, y left, z up).
_UNITY_TO_REACHY = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)


@dataclass
class Calibration:
    head_pos: np.ndarray
    yaw: float


@dataclass
class HandTarget:
    position: np.ndarray
    rotation: np.ndarray
    gripper_closed: float
    inside_workspace: bool


def parse_packet(payload: bytes | str | dict) -> dict:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    data = json.loads(payload)
    if data.get("type") not in (None, "quest2"):
        raise ValueError(f"unexpected packet type {data.get('type')!r}")
    return data


def calibrate_from_packet(data: dict) -> Calibration:
    rotation = quat_to_matrix(np.array([data["hqx"], data["hqy"], data["hqz"], data["hqw"]], dtype=np.float64))
    return Calibration(
        head_pos=np.array([data["hx"], data["hy"], data["hz"]], dtype=np.float64),
        yaw=yaw_about_y(rotation),
    )


def controller_to_hand(data: dict, calibration: Calibration, bench: Bench, *, side: str = "right") -> HandTarget:
    prefix = "r" if side == "right" else "l"
    position = np.array([data[f"{prefix}x"], data[f"{prefix}y"], data[f"{prefix}z"]], dtype=np.float64)
    quat = np.array(
        [data[f"{prefix}qx"], data[f"{prefix}qy"], data[f"{prefix}qz"], data[f"{prefix}qw"]],
        dtype=np.float64,
    )
    relative = _yaw_matrix(-calibration.yaw) @ (position - calibration.head_pos)
    cfg = bench.retarget
    scale = float(cfg["scale"])
    reachy_point = np.array(
        [
            relative[2] * scale + float(cfg["forward_bias_m"]),
            -relative[0] * scale,
            relative[1] * scale - float(cfg["head_above_shoulder_m"]),
        ],
        dtype=np.float64,
    )
    aligned = _yaw_matrix(-calibration.yaw) @ quat_to_matrix(quat)
    rotation = _UNITY_TO_REACHY @ aligned @ _UNITY_TO_REACHY.T
    trigger = float(data[f"{prefix}trig"])
    low = np.array(cfg["workspace_min"], dtype=np.float64)
    high = np.array(cfg["workspace_max"], dtype=np.float64)
    return HandTarget(
        position=reachy_point,
        rotation=rotation,
        gripper_closed=float(np.clip(trigger, 0.0, 1.0)),
        inside_workspace=in_workspace(reachy_point, low, high),
    )


def reachy_point_to_unity_relative(point: np.ndarray, bench: Bench) -> np.ndarray:
    """Inverse of the position map, returned as Unity (right, up, forward)."""
    cfg = bench.retarget
    scale = float(cfg["scale"])
    forward = (point[0] - float(cfg["forward_bias_m"])) / scale
    right = -point[1] / scale
    up = (point[2] + float(cfg["head_above_shoulder_m"])) / scale
    return np.array([right, up, forward], dtype=np.float64)
