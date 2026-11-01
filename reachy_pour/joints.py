"""Joint names for a Reachy 2023 arm, in the order used by reachy-sdk kinematics."""

from __future__ import annotations

import numpy as np

# 7-DoF chain. The gripper is a separate jaw and is not part of inverse kinematics.
RIGHT_CHAIN = (
    "r_shoulder_pitch",
    "r_shoulder_roll",
    "r_arm_yaw",
    "r_elbow_pitch",
    "r_forearm_yaw",
    "r_wrist_pitch",
    "r_wrist_roll",
)
LEFT_CHAIN = (
    "l_shoulder_pitch",
    "l_shoulder_roll",
    "l_arm_yaw",
    "l_elbow_pitch",
    "l_forearm_yaw",
    "l_wrist_pitch",
    "l_wrist_roll",
)

RIGHT_JOINTS = RIGHT_CHAIN + ("r_gripper",)
LEFT_JOINTS = LEFT_CHAIN + ("l_gripper",)

# Resting posture close to the one shown in the Reachy 2023 SDK docs, degrees.
RIGHT_HOME_DEG = np.array([25.0, -8.0, -6.0, -72.0, 0.0, -40.0, -12.0], dtype=np.float64)
LEFT_HOME_DEG = np.array([-25.0, 8.0, 6.0, -72.0, 0.0, -40.0, 12.0], dtype=np.float64)

# Software guards used when this process is the one commanding the motors.
# They are wider than the pour task and are not a substitute for the hardware limits.
RIGHT_LIMITS_DEG = np.array(
    [
        [-140.0, 140.0],
        [-120.0, 40.0],
        [-140.0, 140.0],
        [-140.0, 5.0],
        [-110.0, 110.0],
        [-90.0, 40.0],
        [-110.0, 110.0],
    ],
    dtype=np.float64,
)
LEFT_LIMITS_DEG = np.array(
    [
        [-140.0, 140.0],
        [-40.0, 120.0],
        [-140.0, 140.0],
        [-140.0, 5.0],
        [-110.0, 110.0],
        [-90.0, 40.0],
        [-110.0, 110.0],
    ],
    dtype=np.float64,
)

# Jaw angles are robot-specific. `reachy-pour check` prints the live value so these
# two numbers can be copied from a real open/closed grasp before the first recording.
GRIPPER_OPEN_DEG = -10.0
GRIPPER_CLOSED_DEG = 35.0


def chain_for(arm: str) -> tuple[str, ...]:
    if arm == "right":
        return RIGHT_CHAIN
    if arm == "left":
        return LEFT_CHAIN
    raise ValueError(f"arm must be 'right' or 'left', got {arm!r}")


def joints_for(arm: str) -> tuple[str, ...]:
    if arm == "right":
        return RIGHT_JOINTS
    if arm == "left":
        return LEFT_JOINTS
    raise ValueError(f"arm must be 'right' or 'left', got {arm!r}")


def home_for(arm: str) -> np.ndarray:
    return RIGHT_HOME_DEG.copy() if arm == "right" else LEFT_HOME_DEG.copy()


def limits_for(arm: str) -> np.ndarray:
    return RIGHT_LIMITS_DEG if arm == "right" else LEFT_LIMITS_DEG


def feature_keys(arm: str) -> list[str]:
    return [f"{name}.pos" for name in joints_for(arm)]
