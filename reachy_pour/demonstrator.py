"""Scripted right-arm demonstration of pick, pour, and place.

Joint targets come from the local inverse kinematics. The beaker is parented to
the hand while the gripper is closed, and it tilts with the same clock as the
wrist-roll offset used during the pour. This is a rehearsal of the data layout,
not a dynamics model of Reachy or of the liquid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from reachy_pour.bench import Bench
from reachy_pour.ik import position_ik
from reachy_pour.joints import home_for
from reachy_pour.kinematics import tip_position
from reachy_pour.render import _rot_x, render_view

PHASES: tuple[tuple[str, float], ...] = (
    ("approach", 1.3),
    ("descend", 1.0),
    ("grasp", 0.55),
    ("lift", 1.05),
    ("carry", 1.25),
    ("pour", 1.7),
    ("untilt", 0.75),
    ("return", 1.2),
    ("place", 1.0),
    ("release", 0.5),
    ("retreat", 1.05),
)


def episode_duration() -> float:
    return float(sum(duration for _, duration in PHASES))


def _smoother(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3.0 - 2.0 * u)


def _lerp(a: np.ndarray, b: np.ndarray, u: float) -> np.ndarray:
    return (1.0 - u) * a + u * b


def phase_at(t: float) -> tuple[str, float]:
    cursor = 0.0
    for name, duration in PHASES:
        if t < cursor + duration:
            return name, (t - cursor) / duration
        cursor += duration
    return PHASES[-1][0], 1.0


def _bead_locals(count: int, z: float) -> np.ndarray:
    locals_ = np.zeros((count, 3), dtype=np.float64)
    for index in range(count):
        angle = 2.0 * np.pi * index / count
        locals_[index] = [0.012 * np.cos(angle), 0.012 * np.sin(angle), z]
    return locals_


@dataclass
class Episode:
    state: np.ndarray
    action: np.ndarray
    phase_id: np.ndarray
    phase_name: list[str]
    timestamp: np.ndarray
    image_table: np.ndarray
    image_head: np.ndarray
    cup_xy: np.ndarray
    max_ik_error_m: float
    success: bool
    task: str


def _waypoints(bench: Bench, cup_xy: np.ndarray) -> dict[str, np.ndarray]:
    cup = bench.cup_center(cup_xy)
    lift = np.array([0.0, 0.0, bench.lift_m])
    tray = bench.tray_center()
    carry = np.array([tray[0], tray[1], cup[2] + bench.lift_m])
    return {
        "hover": cup + np.array([0.0, 0.035, bench.lift_m + 0.01]),
        "grasp": cup,
        "lift": cup + lift,
        "carry": carry,
        "place": cup,
    }


def _target_for(phase: str, u: float, points: dict[str, np.ndarray], pour_deg: float) -> tuple[np.ndarray, float, float]:
    """Return Cartesian target, gripper closure in [0, 1], and extra wrist-roll degrees."""
    s = _smoother(u)
    if phase == "approach":
        return _lerp(points["hover"] + np.array([-0.02, 0.03, 0.0]), points["hover"], s), 0.0, 0.0
    if phase == "descend":
        return _lerp(points["hover"], points["grasp"], s), 0.0, 0.0
    if phase == "grasp":
        return points["grasp"], s, 0.0
    if phase == "lift":
        return _lerp(points["grasp"], points["lift"], s), 1.0, 0.0
    if phase == "carry":
        return _lerp(points["lift"], points["carry"], s), 1.0, 0.0
    if phase == "pour":
        return points["carry"], 1.0, pour_deg * s
    if phase == "untilt":
        return points["carry"], 1.0, pour_deg * (1.0 - s)
    if phase == "return":
        return _lerp(points["carry"], points["lift"], s), 1.0, 0.0
    if phase == "place":
        return _lerp(points["lift"], points["place"], s), 1.0, 0.0
    if phase == "release":
        return points["place"], 1.0 - s, 0.0
    return _lerp(points["place"], points["hover"], s), 0.0, 0.0


def simulate_episode(
    bench: Bench,
    *,
    seed: int = 0,
    cup_noise_m: float = 0.0,
    image_scale: float = 1.0,
) -> Episode:
    rng = np.random.default_rng(seed)
    cup_xy = bench.beaker_xy + rng.uniform(-cup_noise_m, cup_noise_m, size=2)
    points = _waypoints(bench, cup_xy)
    names = [name for name, _ in PHASES]
    dt = 1.0 / bench.fps
    times = np.arange(0.0, episode_duration() - 1e-9, dt)
    width = max(16, int(bench.image_width * image_scale))
    height = max(12, int(bench.image_height * image_scale))

    q = home_for(bench.arm)
    present = np.concatenate([q, [bench.gripper_open_deg]])
    roll_base: float | None = None
    attached = False
    hold_offset = np.zeros(3)
    bead_local = _bead_locals(bench.bead_count, -bench.beaker_height * 0.28)
    rest_center = bench.cup_center(cup_xy)
    cup_center = rest_center.copy()
    tilt = 0.0
    released = np.zeros(bench.bead_count, dtype=bool)
    release_from = np.zeros_like(bead_local)
    release_age = np.zeros(bench.bead_count)
    beads = rest_center + bead_local

    state = np.zeros((len(times), 8), dtype=np.float32)
    action = np.zeros((len(times), 8), dtype=np.float32)
    phase_id = np.zeros(len(times), dtype=np.int16)
    phase_name: list[str] = []
    image_table = np.zeros((len(times), height, width, 3), dtype=np.uint8)
    image_head = np.zeros_like(image_table)
    max_err = 0.0
    release_cup = None
    tray = bench.tray_center()
    slots = np.array(
        [
            [
                tray[0] + 0.028 * np.cos(2 * np.pi * i / bench.bead_count),
                tray[1] + 0.02 * np.sin(2 * np.pi * i / bench.bead_count),
                bench.table_top_z + bench.tray_size[2] + bench.bead_radius,
            ]
            for i in range(bench.bead_count)
        ]
    )

    for index, t in enumerate(times):
        phase, u = phase_at(float(t))
        target, grip01, roll_extra = _target_for(phase, u, points, bench.pour_roll_deg)
        lock = None
        if phase in ("pour", "untilt"):
            if roll_base is None:
                roll_base = float(q[6])
            lock = {6: roll_base + roll_extra}
        else:
            roll_base = None
        q, err = position_ik(
            bench.arm,
            target,
            q,
            lock=lock,
            max_iter=24 if index else 60,
            tolerance_m=0.008,
        )
        max_err = max(max_err, err)
        grip = bench.gripper_open_deg + grip01 * (bench.gripper_closed_deg - bench.gripper_open_deg)
        command = np.concatenate([q, [grip]])
        alpha = 1.0 - np.exp(-dt / 0.10)
        present = present + alpha * (command - present)

        hand = tip_position(bench.arm, q)
        tilt = float(np.deg2rad(roll_extra))
        if phase == "lift" and not attached:
            attached = True
            hold_offset = rest_center - hand
        if phase == "retreat":
            attached = False
        if attached:
            cup_center = hand + _rot_x(tilt) @ hold_offset
        else:
            cup_center = rest_center.copy()
            tilt = 0.0
        if phase == "release":
            release_cup = cup_center.copy()

        rotation = _rot_x(tilt if attached else 0.0)
        for bead_index in range(bench.bead_count):
            if not released[bead_index]:
                beads[bead_index] = cup_center + rotation @ bead_local[bead_index]
                if attached and tilt > np.deg2rad(42.0):
                    released[bead_index] = True
                    release_from[bead_index] = beads[bead_index]
                    release_age[bead_index] = 0.0
            else:
                release_age[bead_index] += dt
                mix = _smoother(release_age[bead_index] / 0.55)
                beads[bead_index] = _lerp(release_from[bead_index], slots[bead_index], mix)

        view_kwargs = dict(
            hand=hand,
            cup_center=cup_center,
            cup_tilt=tilt if attached else 0.0,
            beads=beads,
            gripper_closed=grip01,
            width=width,
            height=height,
        )
        image_table[index] = render_view(bench, "table", **view_kwargs)
        image_head[index] = render_view(bench, "head", **view_kwargs)
        state[index] = present.astype(np.float32)
        action[index] = command.astype(np.float32)
        phase_id[index] = names.index(phase)
        phase_name.append(phase)

    beads_in_tray = True
    half = bench.tray_size / 2.0
    for center in beads:
        if abs(center[0] - tray[0]) > half[0] or abs(center[1] - tray[1]) > half[1]:
            beads_in_tray = False
        if center[2] > bench.table_top_z + 0.05:
            beads_in_tray = False
    cup_home = (
        release_cup is not None
        and float(np.linalg.norm(release_cup[:2] - rest_center[:2])) < 0.025
        and abs(float(release_cup[2] - rest_center[2])) < 0.025
    )
    gripper_open = float(action[-1, 7]) < bench.gripper_open_deg + 5.0
    hand_up = float(tip_position(bench.arm, action[-1, :7])[2]) > rest_center[2] + 0.05
    poured = bool(np.any(phase_id == names.index("pour")))
    success = beads_in_tray and cup_home and gripper_open and hand_up and poured and max_err < 0.02

    return Episode(
        state=state,
        action=action,
        phase_id=phase_id,
        phase_name=phase_name,
        timestamp=times.astype(np.float32),
        image_table=image_table,
        image_head=image_head,
        cup_xy=np.asarray(cup_xy, dtype=np.float32),
        max_ik_error_m=float(max_err),
        success=bool(success),
        task=bench.task_en,
    )
