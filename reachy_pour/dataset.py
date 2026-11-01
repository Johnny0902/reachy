"""Studio dataset layout used before a LeRobot install is available.

One compressed ``.npz`` per episode holds state, action, phase, and both cameras.
``meta/info.json`` uses the same feature names LeRobot expects, so
``reachy-pour export-lerobot`` can turn the folder into a ``LeRobotDataset``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from reachy_pour.bench import Bench
from reachy_pour.demonstrator import PHASES, Episode
from reachy_pour.joints import feature_keys
from reachy_pour.png import hstack, vstack, write_png

STUDIO_VERSION = "studio-v1"


def feature_spec(bench: Bench, image_shape: tuple[int, int, int]) -> dict:
    names = [key.removesuffix(".pos") for key in feature_keys(bench.arm)]
    vector = {"dtype": "float32", "shape": [8], "names": names}
    image = {"dtype": "image", "shape": list(image_shape), "names": ["height", "width", "channels"]}
    return {
        "observation.state": vector,
        "action": vector,
        "observation.images.table": image,
        "observation.images.head": image,
    }


def write_dataset(root: Path, bench: Bench, episodes: list[Episode]) -> Path:
    root = Path(root)
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "meta").mkdir(parents=True, exist_ok=True)
    if not episodes:
        raise ValueError("no episodes to write")
    image_shape = episodes[0].image_table.shape[1:]
    for index, episode in enumerate(episodes):
        np.savez_compressed(
            root / "data" / f"episode_{index:06d}.npz",
            state=episode.state,
            action=episode.action,
            phase_id=episode.phase_id,
            timestamp=episode.timestamp,
            image_table=episode.image_table,
            image_head=episode.image_head,
            cup_xy=episode.cup_xy,
        )
    info = {
        "codebase_version": STUDIO_VERSION,
        "kind": "practice",
        "robot_type": "reachy2023",
        "fps": bench.fps,
        "task": bench.task_en,
        "task_zh": bench.task_zh,
        "features": feature_spec(bench, image_shape),
        "phases": [name for name, _ in PHASES],
        "note": "Practice cartoon. Train a deployable policy on real Quest demonstrations.",
    }
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    episodes_path = root / "meta" / "episodes.jsonl"
    tasks_path = root / "meta" / "tasks.jsonl"
    with episodes_path.open("w") as handle:
        for index, episode in enumerate(episodes):
            handle.write(
                json.dumps(
                    {
                        "episode_index": index,
                        "length": int(episode.state.shape[0]),
                        "success": episode.success,
                        "max_ik_error_m": episode.max_ik_error_m,
                        "cup_xy": episode.cup_xy.tolist(),
                        "task": episode.task,
                    }
                )
                + "\n"
            )
    tasks_path.write_text(json.dumps({"task_index": 0, "task": bench.task_en, "task_zh": bench.task_zh}) + "\n")
    write_preview(root / "preview.png", episodes[0])
    return root


def write_preview(path: Path, episode: Episode) -> None:
    count = episode.image_table.shape[0]
    picks = [int(count * fraction) for fraction in (0.08, 0.22, 0.40, 0.55, 0.72, 0.93)]
    picks = [min(index, count - 1) for index in picks]
    table = hstack([episode.image_table[index] for index in picks])
    head = hstack([episode.image_head[index] for index in picks])
    write_png(path, vstack([table, head]))


def validate_dataset(root: Path) -> dict:
    root = Path(root)
    info = json.loads((root / "meta" / "info.json").read_text())
    if info.get("kind") == "teleop":
        raise ValueError("validate 只检查 practice 数据集。真机录制用人工成功标记和 replay。")
    if info.get("codebase_version") != STUDIO_VERSION:
        raise ValueError(f"unexpected dataset version: {info.get('codebase_version')}")
    rows = [json.loads(line) for line in (root / "meta" / "episodes.jsonl").read_text().splitlines() if line]
    if not rows:
        raise ValueError("dataset has no episodes")
    phases = info["phases"]
    required = ["approach", "grasp", "lift", "pour", "place", "release", "retreat"]
    for row in rows:
        blob = np.load(root / "data" / f"episode_{row['episode_index']:06d}.npz")
        length = int(row["length"])
        if blob["state"].shape != (length, 8) or blob["action"].shape != (length, 8):
            raise ValueError(f"episode {row['episode_index']} has a bad state/action shape")
        if blob["image_table"].shape[0] != length or blob["image_head"].shape[0] != length:
            raise ValueError(f"episode {row['episode_index']} is missing frames")
        seen = [phases[int(i)] for i in blob["phase_id"]]
        for name in required:
            if name not in seen:
                raise ValueError(f"episode {row['episode_index']} never entered phase {name}")
        grasp_rows = blob["phase_id"] == phases.index("lift")
        if float(blob["action"][grasp_rows, 7].mean()) < 20.0:
            raise ValueError("gripper is not closed while lifting")
        if float(np.mean(np.abs(blob["image_table"][0].astype(float) - blob["image_table"][length // 2].astype(float)))) < 1.0:
            raise ValueError("table camera frames do not change across the episode")
        if not row["success"]:
            raise ValueError(f"episode {row['episode_index']} was marked unsuccessful")
    return {"episodes": len(rows), "fps": info["fps"], "task": info["task"]}


def export_lerobot(src: Path, repo_id: str, root: Path | None = None) -> Path:
    """Convert a studio dataset with whatever LeRobotDataset.create the install provides."""
    import inspect

    src = Path(src)
    info = json.loads((src / "meta" / "info.json").read_text())
    rows = [json.loads(line) for line in (src / "meta" / "episodes.jsonl").read_text().splitlines() if line]
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise SystemExit(
                "没有安装 LeRobot。按 README 从源码安装 huggingface/lerobot 后，再运行 export-lerobot。"
            ) from exc

    destination = Path(root) if root is not None else src.parent / "lerobot" / repo_id.replace("/", "_")
    features = {}
    for key, spec in info["features"].items():
        features[key] = {
            "dtype": spec["dtype"] if spec["dtype"] != "image" else "image",
            "shape": tuple(spec["shape"]),
            "names": spec["names"],
        }
    create_kwargs = dict(repo_id=repo_id, fps=info["fps"], features=features, root=destination)
    signature = inspect.signature(LeRobotDataset.create)
    if "robot_type" in signature.parameters:
        create_kwargs["robot_type"] = "reachy2023"
    if "use_videos" in signature.parameters:
        create_kwargs["use_videos"] = False
    dataset = LeRobotDataset.create(**create_kwargs)
    add_params = inspect.signature(dataset.add_frame).parameters
    save_params = inspect.signature(dataset.save_episode).parameters
    for row in rows:
        blob = np.load(src / "data" / f"episode_{row['episode_index']:06d}.npz")
        for index in range(int(row["length"])):
            frame = {
                "observation.state": blob["state"][index],
                "action": blob["action"][index],
                "observation.images.table": blob["image_table"][index],
                "observation.images.head": blob["image_head"][index],
            }
            if "task" in add_params:
                dataset.add_frame(frame, task=info["task"])
            else:
                frame["task"] = info["task"]
                dataset.add_frame(frame)
        if "task" in save_params:
            dataset.save_episode(task=info["task"])
        else:
            dataset.save_episode()
    if hasattr(dataset, "finalize"):
        dataset.finalize()
    elif hasattr(dataset, "consolidate"):
        dataset.consolidate()
    return destination
