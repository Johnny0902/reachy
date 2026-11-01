"""Read and command a Reachy 2023 arm, or a local stand-in with the same joints.

The real session talks to ``reachy-sdk`` (the 2021/2023 SDK, not ``reachy2_sdk``).
Motor torque is off unless ``set_stiff(True)`` is called. External-command mode
never writes goals: the official Quest application is already doing that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from reachy_pour.bench import Bench, load_bench
from reachy_pour.joints import feature_keys, home_for, joints_for, limits_for
from reachy_pour.kinematics import tip_position
from reachy_pour.render import render_view
from reachy_pour.safety import clip_limits, clip_relative


@dataclass
class SessionConfig:
    ip_address: str = "192.168.0.42"
    mock: bool = False
    arm: str = "right"
    use_external_commands: bool = False
    use_head_cameras: bool = True
    head_camera_width: int = 640
    head_camera_height: int = 480
    head_cameras_bgr: bool = True
    max_relative_target: float | None = 5.0
    max_temperature_c: float = 55.0
    table_camera_index: int | None = None
    table_camera_width: int = 640
    table_camera_height: int = 480


def resize_nn(image: np.ndarray, height: int, width: int) -> np.ndarray:
    if image.shape[0] == height and image.shape[1] == width:
        return image
    ys = np.linspace(0, image.shape[0] - 1, height).astype(np.int32)
    xs = np.linspace(0, image.shape[1] - 1, width).astype(np.int32)
    return np.ascontiguousarray(image[ys][:, xs])


class ReachySession:
    def __init__(self, config: SessionConfig, bench: Bench | None = None):
        self.config = config
        self.bench = bench if bench is not None else load_bench()
        self.names = joints_for(config.arm)
        self._reachy = None
        self._connected = False
        self._stiff = False
        self._present = np.concatenate([home_for(config.arm), [self.bench.gripper_open_deg]])
        self._goal = self._present.copy()
        self._table_cap = None

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        if self.config.mock:
            self._connected = True
            return
        try:
            from reachy_sdk import ReachySDK
        except ImportError as exc:
            raise ImportError(
                "没有安装 reachy-sdk。真机请执行: pip install -e \".[robot]\""
            ) from exc
        self._reachy = ReachySDK(host=self.config.ip_address)
        self._connected = True
        if self.config.table_camera_index is not None:
            import cv2

            self._table_cap = cv2.VideoCapture(self.config.table_camera_index)
            self._table_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.table_camera_width)
            self._table_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.table_camera_height)

    def close(self) -> None:
        if self._stiff:
            self.set_stiff(False)
        if self._table_cap is not None:
            self._table_cap.release()
            self._table_cap = None
        self._reachy = None
        self._connected = False

    def set_stiff(self, on: bool) -> None:
        if self.config.use_external_commands:
            return
        if self.config.mock:
            self._stiff = on
            return
        part = "r_arm" if self.config.arm == "right" else "l_arm"
        if on:
            self._reachy.turn_on(part)
        else:
            self._reachy.turn_off_smoothly(part)
        self._stiff = on

    def emergency_stop(self) -> None:
        if self.config.mock or self._reachy is None:
            self._stiff = False
            return
        part = "r_arm" if self.config.arm == "right" else "l_arm"
        self._reachy.turn_off_smoothly(part)
        self._stiff = False

    def _joint(self, name: str):
        joints = self._reachy.joints
        if hasattr(joints, name):
            return getattr(joints, name)
        return joints[name]

    def _temperatures(self) -> list[float]:
        if self.config.mock or self._reachy is None:
            return [30.0] * len(self.names)
        values = []
        for name in self.names:
            try:
                values.append(float(self._joint(name).temperature))
            except Exception:
                values.append(0.0)
        return values

    def read_present(self) -> np.ndarray:
        self._ensure()
        if self.config.mock:
            return self._present.copy()
        return np.array([float(self._joint(name).present_position) for name in self.names], dtype=np.float64)

    def read_goal(self) -> np.ndarray:
        self._ensure()
        if self.config.mock:
            return self._goal.copy()
        return np.array([float(self._joint(name).goal_position) for name in self.names], dtype=np.float64)

    def write_goal(self, goal: np.ndarray) -> np.ndarray:
        """Clip to the software limits and to ``max_relative_target``, then send.

        Returns the command that was actually sent. In external mode this is a
        no-op and the original vector is returned unchanged.
        """
        self._ensure()
        goal = np.asarray(goal, dtype=np.float64).reshape(8).copy()
        if self.config.use_external_commands:
            return goal
        present = self.read_present()
        hottest = max(self._temperatures())
        if hottest > self.config.max_temperature_c:
            self.emergency_stop()
            raise RuntimeError(f"关节温度 {hottest:.0f}°C 超过 {self.config.max_temperature_c:.0f}°C，已卸力。")
        limits = limits_for(self.config.arm)
        goal[:7] = clip_limits(goal[:7], limits)
        goal[7] = float(np.clip(goal[7], -45.0, 70.0))
        goal = clip_relative(goal, present, self.config.max_relative_target)
        if self.config.mock:
            self._goal = goal.copy()
            self._present = self._present + 0.35 * (goal - self._present)
            return goal.copy()
        for name, value in zip(self.names, goal, strict=True):
            self._joint(name).goal_position = float(value)
        return goal.copy()

    def read_images(self) -> dict[str, np.ndarray]:
        self._ensure()
        images: dict[str, np.ndarray] = {}
        if self.config.use_head_cameras:
            if self.config.mock:
                images.update(self._mock_images())
            else:
                images["head_left"] = self._sdk_frame("left")
                images["head_right"] = self._sdk_frame("right")
        if self.config.table_camera_index is not None and not self.config.mock:
            images["table"] = self._opencv_frame()
        if height == 0:
            return images
        return images

    def observation(self) -> dict:
        present = self.read_present()
        obs = {key: float(value) for key, value in zip(feature_keys(self.config.arm), present, strict=True)}
        obs.update(self.read_images())
        return obs

    def tip(self) -> np.ndarray:
        return tip_position(self.config.arm, self.read_present()[:7])

    def _ensure(self) -> None:
        if not self._connected:
            raise RuntimeError("Reachy session is not connected.")

    def _sdk_frame(self, side: str) -> np.ndarray:
        camera = self._reachy.left_camera if side == "left" else self._reachy.right_camera
        frame = None if camera is None else getattr(camera, "last_frame", None)
        blank = np.zeros((self.config.head_camera_height, self.config.head_camera_width, 3), dtype=np.uint8)
        if frame is None:
            return blank
        image = np.asarray(frame)
        if image.ndim == 3 and image.shape[2] == 3 and self.config.head_cameras_bgr:
            image = image[:, :, ::-1]
        return resize_nn(image, self.config.head_camera_height, self.config.head_camera_width)

    def _opencv_frame(self) -> np.ndarray:
        blank = np.zeros((self.config.table_camera_height, self.config.table_camera_width, 3), dtype=np.uint8)
        if self._table_cap is None:
            return blank
        ok, frame = self._table_cap.read()
        if not ok or frame is None:
            return blank
        rgb = frame[:, :, ::-1]
        return resize_nn(rgb, self.config.table_camera_height, self.config.table_camera_width)

    def _mock_images(self) -> dict[str, np.ndarray]:
        present = self._present
        hand = tip_position(self.config.arm, present[:7])
        cup = self.bench.cup_center()
        beads = cup + np.zeros((self.bench.bead_count, 3))
        beads[:, 2] -= 0.02
        width = min(160, self.config.head_camera_width)
        height = min(120, self.config.head_camera_height)
        closed = 0.0 if present[7] < 10 else 1.0
        kwargs = dict(hand=hand, cup_center=cup, cup_tilt=0.0, beads=beads, gripper_closed=closed, width=width, height=height)
        table = render_view(self.bench, "table", **kwargs)
        head = render_view(self.bench, "head", **kwargs)
        return {
            "head_left": resize_nn(head, self.config.head_camera_height, self.config.head_camera_width),
            "head_right": resize_nn(head, self.config.head_camera_height, self.config.head_camera_width),
        }
