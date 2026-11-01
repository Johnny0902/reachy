"""Quest 2 packets and the external-command reader used while the official app drives Reachy."""

from __future__ import annotations

import socket
from dataclasses import dataclass, field

import numpy as np

from reachy_pour.bench import Bench
from reachy_pour.ik import position_ik
from reachy_pour.joints import home_for
from reachy_pour.retarget import Calibration, calibrate_from_packet, controller_to_hand, parse_packet


@dataclass
class TeleopStep:
    action: np.ndarray
    tip_error_m: float
    calibrated: bool
    estop: bool
    episode_edge: bool
    workspace_ok: bool
    events: list[str] = field(default_factory=list)


class QuestMapper:
    """Pure mapper from a Quest packet to an 8-DoF joint command. No sockets, no torque."""

    def __init__(self, bench: Bench):
        self.bench = bench
        self.calibration: Calibration | None = None
        self.q = home_for(bench.arm).copy()
        self._prev_a = 0
        self._prev_y = 0
        self.estop = False

    def step(self, packet: dict) -> TeleopStep:
        events: list[str] = []
        a = int(packet.get("a", 0))
        b = int(packet.get("b", 0))
        y = int(packet.get("y", 0))
        episode_edge = y == 1 and self._prev_y == 0
        if a == 1 and self._prev_a == 0:
            self.calibration = calibrate_from_packet(packet)
            events.append("calibrated")
        self._prev_a = a
        self._prev_y = y
        if b == 1:
            self.estop = True
            events.append("estop")
        grip = self.bench.gripper_open_deg
        error = 0.0
        workspace_ok = False
        if self.calibration is not None and not self.estop:
            hand = controller_to_hand(packet, self.calibration, self.bench, side="right")
            grip = self.bench.gripper_open_deg + hand.gripper_closed * (
                self.bench.gripper_closed_deg - self.bench.gripper_open_deg
            )
            workspace_ok = hand.inside_workspace
            if workspace_ok:
                up = hand.rotation[:, 2]
                self.q, error = position_ik(
                    self.bench.arm,
                    hand.position,
                    self.q,
                    up_axis=1,
                    up_target=up,
                    up_weight=0.05,
                    max_iter=30,
                )
                if error > 0.02:
                    self.q, error = position_ik(self.bench.arm, hand.position, self.q, max_iter=30)
        action = np.concatenate([self.q, [grip]])
        return TeleopStep(
            action=action,
            tip_error_m=float(error),
            calibrated=self.calibration is not None,
            estop=self.estop,
            episode_edge=episode_edge,
            workspace_ok=workspace_ok,
            events=events,
        )


class QuestReceiver:
    def __init__(self, bind: str = "0.0.0.0", port: int = 29023):
        self.bind = bind
        self.port = port
        self._sock: socket.socket | None = None
        self.latest: dict | None = None

    def open(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.bind, self.port))
        sock.setblocking(False)
        self._sock = sock

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def poll(self) -> dict | None:
        if self._sock is None:
            raise RuntimeError("Quest receiver is not open.")
        while True:
            try:
                payload, _addr = self._sock.recvfrom(65535)
            except BlockingIOError:
                break
            self.latest = parse_packet(payload)
        return self.latest


def neutral_packet(**overrides) -> dict:
    """A Quest packet with the headset at the tracking origin and the right controller in front."""
    packet = {
        "type": "quest2",
        "a": 0,
        "b": 0,
        "x": 0,
        "y": 0,
        "hx": 0.0,
        "hy": 0.0,
        "hz": 0.0,
        "hqx": 0.0,
        "hqy": 0.0,
        "hqz": 0.0,
        "hqw": 1.0,
        "rx": 0.2,
        "ry": -0.35,
        "rz": 0.30,
        "rqx": 0.0,
        "rqy": 0.0,
        "rqz": 0.0,
        "rqw": 1.0,
        "rtrig": 0.0,
        "lx": -0.2,
        "ly": -0.35,
        "lz": 0.30,
        "lqx": 0.0,
        "lqy": 0.0,
        "lqz": 0.0,
        "lqw": 1.0,
        "ltrig": 0.0,
    }
    packet.update(overrides)
    return packet
