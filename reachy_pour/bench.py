"""The pour-bench specification shared by the demonstrator, Blender, and glTF export."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from reachy_pour.paths import repo_root


@dataclass
class Bench:
    task_en: str
    task_zh: str
    arm: str
    fps: int
    table_size: np.ndarray
    table_top_center: np.ndarray
    beaker_outer_r: float
    beaker_inner_r: float
    beaker_height: float
    beaker_xy: np.ndarray
    tray_size: np.ndarray
    tray_xy: np.ndarray
    bead_count: int
    bead_radius: float
    cameras: dict
    retarget: dict
    pour_roll_deg: float
    lift_m: float
    gripper_open_deg: float
    gripper_closed_deg: float
    image_width: int
    image_height: int

    @property
    def table_top_z(self) -> float:
        return float(self.table_top_center[2])

    def cup_center(self, xy: np.ndarray | None = None) -> np.ndarray:
        point = self.beaker_xy if xy is None else np.asarray(xy, dtype=np.float64)
        return np.array([point[0], point[1], self.table_top_z + self.beaker_height / 2.0])

    def tray_center(self) -> np.ndarray:
        return np.array(
            [self.tray_xy[0], self.tray_xy[1], self.table_top_z + self.tray_size[2] / 2.0]
        )

    def table_box(self) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned table as (center, half_extents). The top face is ``table_top_z``."""
        half = self.table_size / 2.0
        center = self.table_top_center.copy()
        center[2] = self.table_top_z - half[2]
        return center, half

    def describe(self) -> str:
        top = self.table_top_z
        cup = self.cup_center()
        tray = self.tray_center()
        lines = [
            "实验台尺寸（Reachy 坐标系，肩部中点为原点，+X 向前，+Y 向机器人左侧，+Z 向上）",
            f"  桌面: {self.table_size[0]*100:.0f} × {self.table_size[1]*100:.0f} cm，"
            f"台面在原点下方 {-top*100:.1f} cm",
            f"  烧杯: 外径 {self.beaker_outer_r*2*100:.1f} cm，高 {self.beaker_height*100:.1f} cm，"
            f"放在 ({cup[0]*100:.1f}, {cup[1]*100:.1f}) cm",
            f"  托盘: {self.tray_size[0]*100:.0f} × {self.tray_size[1]*100:.0f} cm，"
            f"放在 ({tray[0]*100:.1f}, {tray[1]*100:.1f}) cm",
            f"  物体: {self.bead_count} 颗直径 {self.bead_radius*2*100:.1f} cm 的小球",
            f"  任务: {self.task_zh}",
            f"        {self.task_en}",
        ]
        return "\n".join(lines)


def default_bench_path() -> Path:
    return repo_root() / "configs" / "pour_bench.json"


def load_bench(path: Path | None = None) -> Bench:
    path = default_bench_path() if path is None else Path(path)
    raw = json.loads(path.read_text())
    return Bench(
        task_en=raw["task_en"],
        task_zh=raw["task_zh"],
        arm=raw["arm"],
        fps=int(raw["fps"]),
        table_size=np.array(raw["table"]["size_xyz"], dtype=np.float64),
        table_top_center=np.array(raw["table"]["top_center"], dtype=np.float64),
        beaker_outer_r=float(raw["beaker"]["outer_radius"]),
        beaker_inner_r=float(raw["beaker"]["inner_radius"]),
        beaker_height=float(raw["beaker"]["height"]),
        beaker_xy=np.array(raw["beaker"]["xy"], dtype=np.float64),
        tray_size=np.array(raw["tray"]["size_xyz"], dtype=np.float64),
        tray_xy=np.array(raw["tray"]["xy"], dtype=np.float64),
        bead_count=int(raw["beads"]["count"]),
        bead_radius=float(raw["beads"]["radius"]),
        cameras=raw["cameras"],
        retarget=raw["retarget"],
        pour_roll_deg=float(raw["pour_roll_deg"]),
        lift_m=float(raw["lift_m"]),
        gripper_open_deg=float(raw["gripper_open_deg"]),
        gripper_closed_deg=float(raw["gripper_closed_deg"]),
        image_width=int(raw["image_width"]),
        image_height=int(raw["image_height"]),
    )
