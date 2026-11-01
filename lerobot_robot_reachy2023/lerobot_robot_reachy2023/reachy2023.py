"""Reachy 2023 as a LeRobot ``Robot``.

``get_observation`` / ``send_action`` are the only methods the record, train, and
eval tools need. Joint values are degrees, the same unit as ``reachy-sdk``.
"""

from __future__ import annotations

from typing import Any

from lerobot.robots.robot import Robot

from reachy_pour.driver import ReachySession, SessionConfig
from reachy_pour.joints import feature_keys

from .config_reachy2023 import Reachy2023Config

try:
    from lerobot.cameras import make_cameras_from_configs
except ImportError:
    from lerobot.common.cameras.camera import make_cameras_from_configs


class Reachy2023(Robot):
    config_class = Reachy2023Config
    name = "reachy2023"

    def __init__(self, config: Reachy2023Config):
        if config.id is None:
            config.id = "reachy2023"
        super().__init__(config)
        self.config = config
        self.session = ReachySession(
            SessionConfig(
                ip_address=config.ip_address,
                mock=config.mock,
                arm=config.arm,
                use_external_commands=config.use_external_commands,
                use_head_cameras=config.use_head_cameras,
                head_camera_width=config.head_camera_width,
                head_camera_height=config.head_camera_height,
                head_cameras_bgr=config.head_cameras_bgr,
                max_relative_target=config.max_relative_target,
                max_temperature_c=config.max_temperature_c,
                table_camera_index=config.table_camera_index,
            )
        )
        self.cameras = make_cameras_from_configs(config.cameras) if config.cameras else {}
        # External mode does not send targets, so a rollout must not pretend it can hold one.
        self.supports_position_hold = not config.use_external_commands and not config.mock

    @property
    def _joint_features(self) -> dict[str, type]:
        return {key: float for key in feature_keys(self.config.arm)}

    @property
    def observation_features(self) -> dict:
        features: dict[str, Any] = dict(self._joint_features)
        if self.config.use_head_cameras:
            shape = (self.config.head_camera_height, self.config.head_camera_width, 3)
            features["head_left"] = shape
            features["head_right"] = shape
        for name, camera in self.cameras.items():
            features[name] = (camera.height, camera.width, 3)
        return features

    @property
    def action_features(self) -> dict:
        return dict(self._joint_features)

    @property
    def is_connected(self) -> bool:
        cameras_ok = all(camera.is_connected for camera in self.cameras.values())
        return self.session.connected and cameras_ok

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def connect(self, calibrate: bool = True) -> None:
        self.session.connect()
        for camera in self.cameras.values():
            camera.connect()
        if self.config.torque_on_connect and not self.config.use_external_commands:
            self.session.set_stiff(True)

    def disconnect(self) -> None:
        self.session.close()
        for camera in self.cameras.values():
            camera.disconnect()

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")
        observation = self.session.observation()
        for name, camera in self.cameras.items():
            observation[name] = camera.async_read()
        return observation

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")
        keys = feature_keys(self.config.arm)
        goal = [float(action[key]) for key in keys]
        applied = self.session.write_goal(goal)
        return {key: float(value) for key, value in zip(keys, applied, strict=True)}
