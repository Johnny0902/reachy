"""Teleoperator config for Reachy 2023 VR.

``mode="external"`` reads the joint target the official Quest application already
sent to the robot. ``mode="quest"`` listens for the UDP stream from the custom
Unity scene in ``unity/ReachyPourTeleop``.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    from lerobot.teleoperators.config import TeleoperatorConfig
except ImportError:
    from lerobot.common.robot_devices.teleoperations.configs import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("reachy2023_vr")
@dataclass(kw_only=True)
class Reachy2023VrConfig(TeleoperatorConfig):
    ip_address: str = "192.168.0.42"
    mode: str = "external"
    use_present_position: bool = False
    mock: bool = False
    udp_bind: str = "0.0.0.0"
    udp_port: int = 29023
    arm: str = "right"
