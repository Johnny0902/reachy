"""Build the pour bench in Blender from configs/pour_bench.json.

Run from the repository root:

    blender --background --python blender/build_pour_scene.py -- --out assets/pour_bench.blend

Axes inside the .blend file are Blender's (X right, Y forward, Z up), mapped from
Reachy (X forward, Y left, Z up) by:

    blender_x = -reachy_y
    blender_y = reachy_x
    blender_z = reachy_z

The glTF written by ``reachy-pour export-scene`` uses the same conversion.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import bpy


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def reachy_to_blender(point):
    return (-point[1], point[0], point[2])


def args_after_double_dash():
    argv = sys.argv
    if "--" not in argv:
        return []
    return argv[argv.index("--") + 1 :]


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.cameras):
        for item in list(block):
            if item.users == 0:
                block.remove(item)


def material(name, color, transmission=0.0, alpha=1.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = color
        alpha_input = bsdf.inputs.get("Alpha")
        if alpha_input is not None:
            alpha_input.default_value = alpha
        for key in ("Transmission Weight", "Transmission"):
            socket = bsdf.inputs.get(key)
            if socket is not None:
                socket.default_value = transmission
                break
    if alpha < 1.0:
        mat.blend_method = "BLEND"
    return mat


def add_cube(name, center_reachy, size_reachy, mat):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=reachy_to_blender(center_reachy))
    obj = bpy.context.active_object
    obj.name = name
    # Cube of size 1 is scaled in Blender axes. Reachy size (sx, sy, sz)
    # becomes Blender scale (sy, sx, sz) because X/Y are swapped.
    obj.scale = (size_reachy[1], size_reachy[0], size_reachy[2])
    obj.data.materials.append(mat)
    return obj


def add_beaker(bench):
    center = [
        bench["beaker"]["xy"][0],
        bench["beaker"]["xy"][1],
        bench["table"]["top_center"][2] + bench["beaker"]["height"] / 2.0,
    ]
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=48,
        radius=bench["beaker"]["outer_radius"],
        depth=bench["beaker"]["height"],
        location=reachy_to_blender(center),
    )
    beaker = bpy.context.active_object
    beaker.name = "Beaker"
    glass = material("Glass", (0.75, 0.86, 0.90, 1.0), transmission=0.85, alpha=0.35)
    beaker.data.materials.append(glass)
    solidify = beaker.modifiers.new("Wall", "SOLIDIFY")
    solidify.thickness = bench["beaker"]["outer_radius"] - bench["beaker"]["inner_radius"]
    bead_mat = material("Bead", (0.12, 0.35, 0.85, 1.0))
    count = bench["beads"]["count"]
    for index in range(count):
        angle = 2.0 * math.pi * index / count
        local_reachy = (0.012 * math.cos(angle), 0.012 * math.sin(angle), -0.02)
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=bench["beads"]["radius"],
            location=reachy_to_blender(
                (center[0] + local_reachy[0], center[1] + local_reachy[1], center[2] + local_reachy[2])
            ),
        )
        bead = bpy.context.active_object
        bead.name = f"Bead_{index}"
        bead.data.materials.append(bead_mat)
        world = bead.matrix_world.copy()
        bead.parent = beaker
        bead.matrix_world = world
    return beaker


def enable_gltf_addon():
    try:
        bpy.ops.preferences.addon_enable(module="io_scene_gltf2")
    except Exception:
        pass


def main():
    options = args_after_double_dash()
    out = repo_root() / "assets" / "pour_bench.blend"
    if "--out" in options:
        out = Path(options[options.index("--out") + 1])
    bench = json.loads((repo_root() / "configs" / "pour_bench.json").read_text())
    enable_gltf_addon()
    clear_scene()

    table = bench["table"]
    top = table["top_center"]
    size = table["size_xyz"]
    table_center = (top[0], top[1], top[2] - size[2] / 2.0)
    wood = material("Wood", (0.62, 0.45, 0.28, 1.0))
    add_cube("Table", table_center, size, wood)

    tray = bench["tray"]
    tray_center = (tray["xy"][0], tray["xy"][1], top[2] + tray["size_xyz"][2] / 2.0)
    add_cube("Tray", tray_center, tray["size_xyz"], material("Tray", (0.25, 0.28, 0.32, 1.0)))
    add_beaker(bench)

    bpy.ops.object.empty_add(type="ARROWS", location=(0.0, 0.0, 0.0))
    bpy.context.active_object.name = "ReachyOrigin"

    eye = bench["cameras"]["table"]["eye"]
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=reachy_to_blender(eye))
    bpy.context.active_object.name = "TableCamera"

    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out))
    gltf = out.with_suffix(".gltf")
    bpy.ops.export_scene.gltf(filepath=str(gltf), export_format="GLTF_SEPARATE")
    print(f"wrote {out}")
    print(f"wrote {gltf}")


if __name__ == "__main__":
    main()
