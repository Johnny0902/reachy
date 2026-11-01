"""LeRobot config for Reachy 2023.

This is the 2023 torso, reached through ``reachy-sdk``. It is not the built-in
``reachy2`` robot, which talks to ``reachy2_sdk``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

try:
    from lerobot.cameras import CameraConfig
    from lerobot.robots import RobotConfig
except ImportError:  # lerobot < 0.4 kept cameras under lerobot.common
    from lerobot.common.cameras.camera import CameraConfig
    from lerobot.common.robot_devices.robots.configs import RobotConfig

try:
    from lerobot.cameras.opencv import OpenCVCameraConfig as _OpenCVCameraConfig
except ImportError:  # pragma: no cover - registration side effect for the CLI parser
    try:
        from lerobot.common.cameras.opencv import OpenCVCameraConfig as _OpenCVCameraConfig
    except ImportError:
        _OpenCVCameraConfig = None

_ = _OpenCVCameraConfig


@RobotConfig.register_subclass("reachy2023")
@dataclass(kw_only=True)
class Reachy2023Config(RobotConfig):
    ip_address: str = "192.168.0.42"
    mock: bool = False
    arm: str = "right"
    # True while the official Quest application is already sending motor goals.
    use_external_commands: bool = False
    torque_on_connect: bool = False
    use_head_cameras: bool = True
    head_camera_width: int = 640
    head_camera_height: int = 480
    # reachy-sdk frames are shown with cv2.imshow in the docs, so they are BGR.
    head_cameras_bgr: bool = True
    max_relative_target: float | None = 5.0
    max_temperature_c: float = 55.0
    table_camera_index: int | None = None
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
