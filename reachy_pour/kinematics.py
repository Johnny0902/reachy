"""Local forward kinematics for the Reachy 2023 arm.

The translation/axis table is the one published in the Reachy 2023 SDK
("Arms kinematics"). On the real robot, prefer ``reachy.r_arm.inverse_kinematics``:
that solver runs inside the robot's kinematics service. This module exists so the
Quest bridge and the offline practice loop have a model when that service is not
connected.

Joint angles are degrees, matching ``present_position`` / ``goal_position``.
The frame is Reachy's: +X forward, +Y to the robot's left, +Z up, origin at the
midpoint between the shoulders.
"""

from __future__ import annotations

import numpy as np

from reachy_pour.joints import LEFT_CHAIN, RIGHT_CHAIN

# (translation meters in the previous frame, rotation axis of this joint)
RIGHT_MODEL = (
    (np.array([0.0, -0.019, 0.0]), np.array([0.0, 1.0, 0.0])),
    (np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
    (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
    (np.array([0.0, 0.0, -0.28]), np.array([0.0, 1.0, 0.0])),
    (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
    (np.array([0.0, 0.0, -0.25]), np.array([0.0, 1.0, 0.0])),
    (np.array([0.0, 0.0, -0.0325]), np.array([1.0, 0.0, 0.0])),
)
# The published table is for the right arm. The left arm is the same chain with
# the shoulder and gripper Y offsets mirrored.
LEFT_MODEL = (
    (np.array([0.0, 0.019, 0.0]), np.array([0.0, 1.0, 0.0])),
    (np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
    (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
    (np.array([0.0, 0.0, -0.28]), np.array([0.0, 1.0, 0.0])),
    (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
    (np.array([0.0, 0.0, -0.25]), np.array([0.0, 1.0, 0.0])),
    (np.array([0.0, 0.0, -0.0325]), np.array([1.0, 0.0, 0.0])),
)
RIGHT_TIP_OFFSET = np.array([0.0, -0.01, -0.075])
LEFT_TIP_OFFSET = np.array([0.0, 0.01, -0.075])


def _rot(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    k = axis / np.linalg.norm(axis)
    skew = np.array(
        [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3) + np.sin(angle_rad) * skew + (1.0 - np.cos(angle_rad)) * (skew @ skew)


def model_for(arm: str) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    if arm == "right":
        return RIGHT_MODEL
    if arm == "left":
        return LEFT_MODEL
    raise ValueError(arm)


def tip_offset_for(arm: str) -> np.ndarray:
    return RIGHT_TIP_OFFSET if arm == "right" else LEFT_TIP_OFFSET


def chain_names(arm: str) -> tuple[str, ...]:
    return RIGHT_CHAIN if arm == "right" else LEFT_CHAIN


def forward(arm: str, q_deg: np.ndarray) -> np.ndarray:
    """4x4 pose of the gripper tip. ``q_deg`` is the 7 arm joints, gripper excluded."""
    q = np.asarray(q_deg, dtype=np.float64).reshape(7)
    pose = np.eye(4)
    for (translation, axis), angle_deg in zip(model_for(arm), q, strict=True):
        local = np.eye(4)
        local[:3, :3] = _rot(axis, np.deg2rad(angle_deg))
        local[:3, 3] = translation
        pose = pose @ local
    tip = np.eye(4)
    tip[:3, 3] = tip_offset_for(arm)
    return pose @ tip


def tip_position(arm: str, q_deg: np.ndarray) -> np.ndarray:
    return forward(arm, q_deg)[:3, 3].copy()


def tip_rotation(arm: str, q_deg: np.ndarray) -> np.ndarray:
    return forward(arm, q_deg)[:3, :3].copy()
