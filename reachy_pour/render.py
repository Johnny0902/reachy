"""Small ray-traced views of the pour bench.

The pictures are a kinematic cartoon for the record/train/eval rehearsal. They are
not a physics render and are not a substitute for the real camera streams.
"""

from __future__ import annotations

import numpy as np

from reachy_pour.bench import Bench

_BACKGROUND = np.array([232, 228, 220], dtype=np.float64)
_TABLE = np.array([176, 132, 86], dtype=np.float64)
_TRAY = np.array([64, 74, 84], dtype=np.float64)
_BEAD = np.array([36, 96, 210], dtype=np.float64)
_GRIPPER = np.array([28, 28, 32], dtype=np.float64)
_GLASS = np.array([186, 214, 220], dtype=np.float64)


def _normalize(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def camera_rays(eye: np.ndarray, target: np.ndarray, fov_y_deg: float, height: int, width: int):
    forward = _normalize(target - eye)
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(forward, world_up))) > 0.95:
        world_up = np.array([0.0, 1.0, 0.0])
    right = _normalize(np.cross(forward, world_up))
    up = np.cross(right, forward)
    aspect = width / height
    scale = np.tan(np.deg2rad(fov_y_deg) / 2.0)
    xs = ((np.arange(width) + 0.5) / width * 2.0 - 1.0) * scale * aspect
    ys = -((np.arange(height) + 0.5) / height * 2.0 - 1.0) * scale
    dirs = forward + xs[None, :, None] * right + ys[:, None, None] * up
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    return dirs


def _paint(color: np.ndarray, depth: np.ndarray, t_hit: np.ndarray, rgb: np.ndarray) -> None:
    valid = np.isfinite(t_hit) & (t_hit > 1e-4) & (t_hit < depth)
    color[valid] = rgb
    depth[valid] = t_hit[valid]


def _aabb(eye: np.ndarray, dirs: np.ndarray, center: np.ndarray, half: np.ndarray) -> np.ndarray:
    low = center - half
    high = center + half
    safe = np.where(np.abs(dirs) < 1e-8, 1e-8, dirs)
    t0 = (low - eye) / safe
    t1 = (high - eye) / safe
    t_near = np.minimum(t0, t1).max(axis=-1)
    t_far = np.maximum(t0, t1).min(axis=-1)
    hit = (t_far >= np.maximum(t_near, 0.0)) & (t_far > 0.0)
    t_hit = np.where(t_near > 0.0, t_near, t_far)
    return np.where(hit, t_hit, np.inf)


def _sphere(eye: np.ndarray, dirs: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    oc = eye - center
    b = 2.0 * (dirs @ oc)
    c = float(oc @ oc) - radius * radius
    disc = b * b - 4.0 * c
    sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
    t0 = (-b - sqrt_disc) / 2.0
    t1 = (-b + sqrt_disc) / 2.0
    t_hit = np.where(t0 > 1e-4, t0, t1)
    return np.where((disc >= 0.0) & (t_hit > 1e-4), t_hit, np.inf)


def _z_cylinder(eye_l: np.ndarray, dirs_l: np.ndarray, radius: float, half_h: float) -> np.ndarray:
    ox, oy, oz = eye_l
    dx = dirs_l[..., 0]
    dy = dirs_l[..., 1]
    dz = dirs_l[..., 2]
    a = dx * dx + dy * dy
    b = 2.0 * (ox * dx + oy * dy)
    c = ox * ox + oy * oy - radius * radius
    disc = b * b - 4.0 * a * c
    sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
    safe_a = np.where(np.abs(a) < 1e-8, 1.0, a)
    t0 = (-b - sqrt_disc) / (2.0 * safe_a)
    t1 = (-b + sqrt_disc) / (2.0 * safe_a)
    z0 = oz + t0 * dz
    z1 = oz + t1 * dz
    ok0 = (disc >= 0.0) & (np.abs(a) > 1e-8) & (t0 > 1e-4) & (np.abs(z0) <= half_h)
    ok1 = (disc >= 0.0) & (np.abs(a) > 1e-8) & (t1 > 1e-4) & (np.abs(z1) <= half_h)
    t_hit = np.where(ok0, t0, np.where(ok1, t1, np.inf))
    return t_hit


def _rot_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def render_view(
    bench: Bench,
    camera: str,
    *,
    hand: np.ndarray,
    cup_center: np.ndarray,
    cup_tilt: float,
    beads: np.ndarray,
    gripper_closed: float,
    width: int | None = None,
    height: int | None = None,
) -> np.ndarray:
    spec = bench.cameras[camera]
    width = bench.image_width if width is None else width
    height = bench.image_height if height is None else height
    eye = np.array(spec["eye"], dtype=np.float64)
    target = np.array(spec["target"], dtype=np.float64)
    dirs = camera_rays(eye, target, float(spec["fov_y_deg"]), height, width)
    color = np.broadcast_to(_BACKGROUND, (height, width, 3)).copy()
    depth = np.full((height, width), np.inf)

    table_center, table_half = bench.table_box()
    _paint(color, depth, _aabb(eye, dirs, table_center, table_half), _TABLE)
    _paint(color, depth, _aabb(eye, dirs, bench.tray_center(), bench.tray_size / 2.0), _TRAY)
    for center in beads:
        _paint(color, depth, _sphere(eye, dirs, center, bench.bead_radius), _BEAD)

    grip_radius = 0.014 + 0.004 * gripper_closed
    _paint(color, depth, _sphere(eye, dirs, hand + np.array([0.0, 0.0, 0.02]), grip_radius), _GRIPPER)

    rotation = _rot_x(cup_tilt)
    eye_l = rotation.T @ (eye - cup_center)
    dirs_l = dirs @ rotation
    outer = _z_cylinder(eye_l, dirs_l, bench.beaker_outer_r, bench.beaker_height / 2.0)
    inner = _z_cylinder(eye_l, dirs_l, bench.beaker_inner_r, bench.beaker_height / 2.0 - 0.004)
    shell = np.isfinite(outer) & (outer < depth) & (outer + 1e-4 < inner)
    color[shell] = 0.42 * _GLASS + 0.58 * color[shell]
    return np.clip(color, 0, 255).astype(np.uint8)
