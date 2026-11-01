"""Write a glTF of the bench without Blender.

Blender is Z-up. This file uses the same axis change Blender's glTF exporter
applies when the blend file is authored as blender_x = -reachy_y, blender_y =
reachy_x, blender_z = reachy_z (see ``blender/build_pour_scene.py``).
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from reachy_pour.bench import Bench


def _to_gltf(points: np.ndarray) -> np.ndarray:
    """Reachy (x forward, y left, z up) -> glTF (x right, y up, z backward)."""
    out = np.empty_like(points, dtype=np.float32)
    out[:, 0] = -points[:, 1]
    out[:, 1] = points[:, 2]
    out[:, 2] = -points[:, 0]
    return out


def _box(center: np.ndarray, size: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    half = size / 2.0
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)
    vertices = center + corners * half
    # Corner index bits: x, y, z. Two triangles per face.
    quads = [
        (0, 1, 3, 2),  # z-
        (4, 6, 7, 5),  # z+
        (0, 2, 6, 4),  # x-
        (1, 5, 7, 3),  # x+
        (0, 4, 5, 1),  # y-
        (2, 3, 7, 6),  # y+
    ]
    triangles = []
    for a, b, c, d in quads:
        triangles.extend((a, b, c, a, c, d))
    return vertices, np.array(triangles, dtype=np.uint32)


def _cylinder(center: np.ndarray, radius: float, height: float, sides: int = 20) -> tuple[np.ndarray, np.ndarray]:
    angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    ring = np.stack([np.cos(angles) * radius, np.sin(angles) * radius], axis=1)
    bottom = np.column_stack([ring[:, 0], ring[:, 1], np.full(sides, -height / 2.0)]) + center
    top = np.column_stack([ring[:, 0], ring[:, 1], np.full(sides, height / 2.0)]) + center
    vertices = np.vstack([bottom, top, center + np.array([0, 0, -height / 2]), center + np.array([0, 0, height / 2])])
    bottom_center = sides * 2
    top_center = bottom_center + 1
    triangles = []
    for i in range(sides):
        j = (i + 1) % sides
        triangles.extend((i, j, sides + j, i, sides + j, sides + i))
        triangles.extend((bottom_center, j, i))
        triangles.extend((top_center, sides + i, sides + j))
    return vertices, np.array(triangles, dtype=np.uint32)


def _mesh(positions: np.ndarray, indices: np.ndarray) -> dict:
    positions = _to_gltf(positions.astype(np.float32))
    indices = indices.astype(np.uint32)
    pos_bytes = positions.tobytes()
    idx_bytes = indices.tobytes()
    blob = pos_bytes + idx_bytes
    return {
        "positions": positions,
        "indices": indices,
        "blob": blob,
        "pos_len": len(pos_bytes),
        "min": positions.min(axis=0).tolist(),
        "max": positions.max(axis=0).tolist(),
    }


def build_gltf(bench: Bench) -> dict:
    table_center, table_half = bench.table_box()
    meshes_raw = {
        "Table": _mesh(*_box(table_center, table_half * 2.0)),
        "Tray": _mesh(*_box(bench.tray_center(), bench.tray_size)),
        "Beaker": _mesh(*_cylinder(bench.cup_center(), bench.beaker_outer_r, bench.beaker_height)),
    }
    for index in range(bench.bead_count):
        angle = 2.0 * np.pi * index / bench.bead_count
        center = bench.cup_center() + np.array([0.012 * np.cos(angle), 0.012 * np.sin(angle), -0.02])
        meshes_raw[f"Bead_{index}"] = _mesh(*_box(center, np.full(3, bench.bead_radius * 2.0)))

    blob = b"".join(item["blob"] for item in meshes_raw.values())
    buffer_uri = "data:application/octet-stream;base64," + base64.b64encode(blob).decode("ascii")
    views = []
    accessors = []
    meshes = []
    nodes = []
    offset = 0
    colors = {
        "Table": [0.62, 0.45, 0.28, 1.0],
        "Tray": [0.25, 0.28, 0.32, 1.0],
        "Beaker": [0.75, 0.86, 0.9, 0.45],
    }
    materials = [
        {"name": "wood", "pbrMetallicRoughness": {"baseColorFactor": colors["Table"], "roughnessFactor": 0.8}},
        {"name": "tray", "pbrMetallicRoughness": {"baseColorFactor": colors["Tray"], "roughnessFactor": 0.5}},
        {
            "name": "glass",
            "pbrMetallicRoughness": {"baseColorFactor": colors["Beaker"], "roughnessFactor": 0.1},
            "alphaMode": "BLEND",
        },
        {"name": "bead", "pbrMetallicRoughness": {"baseColorFactor": [0.12, 0.35, 0.85, 1.0], "roughnessFactor": 0.3}},
    ]
    material_index = {"Table": 0, "Tray": 1, "Beaker": 2}

    for name, item in meshes_raw.items():
        view_pos = len(views)
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": item["pos_len"], "target": 34962})
        offset += item["pos_len"]
        view_idx = len(views)
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(item["indices"].tobytes()), "target": 34963})
        offset += len(item["indices"].tobytes())
        acc_pos = len(accessors)
        accessors.append(
            {
                "bufferView": view_pos,
                "componentType": 5126,
                "count": int(item["positions"].shape[0]),
                "type": "VEC3",
                "min": item["min"],
                "max": item["max"],
            }
        )
        acc_idx = len(accessors)
        accessors.append(
            {
                "bufferView": view_idx,
                "componentType": 5125,
                "count": int(item["indices"].shape[0]),
                "type": "SCALAR",
            }
        )
        mesh_index = len(meshes)
        meshes.append(
            {
                "name": name,
                "primitives": [
                    {
                        "attributes": {"POSITION": acc_pos},
                        "indices": acc_idx,
                        "material": material_index.get(name, 3),
                    }
                ],
            }
        )
        nodes.append({"name": name, "mesh": mesh_index})

    nodes.append({"name": "ReachyOrigin"})
    return {
        "asset": {"version": "2.0", "generator": "reachy-pour"},
        "scene": 0,
        "scenes": [{"name": "PourBench", "nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(blob), "uri": buffer_uri}],
    }


def write_gltf(path: Path, bench: Bench) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_gltf(bench)))
    return path
