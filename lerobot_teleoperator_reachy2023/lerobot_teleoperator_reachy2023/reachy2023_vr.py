"""Quest 2 / official-app teleoperator for Reachy 2023."""

from __future__ import annotations

from typing import Any

import numpy as np
from lerobot.teleoperators.teleoperator import Teleoperator

from reachy_pour.bench import load_bench
from reachy_pour.driver import ReachySession, SessionConfig
from reachy_pour.joints import feature_keys, home_for
from reachy_pour.teleop import QuestMapper, QuestReceiver

from .config_reachy2023_vr import Reachy2023VrConfig


class Reachy2023Vr(Teleoperator):
    config_class = Reachy2023VrConfig
    name = "reachy2023_vr"

    def __init__(self, config: Reachy2023VrConfig):
        if config.id is None:
            config.id = "reachy2023_vr"
        super().__init__(config)
        self.config = config
        if config.mode not in ("external", "quest"):
            raise ValueError("mode must be 'external' or 'quest'")
        self._bench = load_bench()
        self._session: ReachySession | None = None
        self._receiver: QuestReceiver | None = None
        self._mapper: QuestMapper | None = None
        self._connected = False
        if config.mode == "external":
            self._session = ReachySession(
                SessionConfig(
                    ip_address=config.ip_address,
                    mock=config.mock,
                    arm=config.arm,
                    use_external_commands=True,
                    use_head_cameras=False,
                ),
                self._bench,
            )
        else:
            self._receiver = QuestReceiver(config.udp_bind, config.udp_port)
            self._mapper = QuestMapper(self._bench)

    @property
    def action_features(self) -> dict:
        return {key: float for key in feature_keys(self.config.arm)}

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        if self._mapper is None:
            return True
        return self._mapper.calibration is not None

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def connect(self, calibrate: bool = True) -> None:
        if self._session is not None:
            self._session.connect()
        if self._receiver is not None:
            self._receiver.open()
        self._connected = True

    def disconnect(self) -> None:
        if self._session is not None:
            self._session.close()
        if self._receiver is not None:
            self._receiver.close()
        self._connected = False

    def get_action(self) -> dict[str, float]:
        if not self._connected:
            raise ConnectionError(f"{self} is not connected.")
        keys = feature_keys(self.config.arm)
        if self._session is not None:
            values = self._session.read_present() if self.config.use_present_position else self._session.read_goal()
        else:
            assert self._receiver is not None and self._mapper is not None
            packet = self._receiver.poll()
            if packet is None:
                values = np.concatenate([home_for(self.config.arm), [self._bench.gripper_open_deg]])
                if self._mapper.q is not None:
                    values = np.concatenate([self._mapper.q, [self._bench.gripper_open_deg]])
            else:
                values = self._mapper.step(packet).action
        return {key: float(value) for key, value in zip(keys, values, strict=True)}

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        return None
