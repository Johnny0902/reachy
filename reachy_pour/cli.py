"""Command line for the Reachy 2023 pour-task lab."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from reachy_pour.bench import load_bench
from reachy_pour.dataset import export_lerobot, validate_dataset, write_dataset
from reachy_pour.demonstrator import simulate_episode
from reachy_pour.driver import ReachySession, SessionConfig
from reachy_pour.gltf_export import write_gltf
from reachy_pour.joints import joints_for
from reachy_pour.teleop import QuestMapper, QuestReceiver


def _practice(args: argparse.Namespace) -> int:
    bench = load_bench(args.bench)
    episodes = []
    for index in range(args.episodes):
        episode = simulate_episode(
            bench,
            seed=args.seed + index,
            cup_noise_m=args.noise,
            image_scale=args.image_scale,
        )
        episodes.append(episode)
        print(
            f"episode {index:03d}  frames={episode.state.shape[0]}  "
            f"ik={episode.max_ik_error_m * 100:.2f} cm  success={episode.success}"
        )
    root = write_dataset(Path(args.out), bench, episodes)
    summary = validate_dataset(root)
    print(f"wrote {root}")
    print(f"validated {summary['episodes']} episode(s), task: {summary['task']}")
    if not all(episode.success for episode in episodes):
        print("有 episode 没有完成倒出和放下。先看 ik 误差，再调 configs/pour_bench.json。")
        return 1
    return 0


def _validate(args: argparse.Namespace) -> int:
    summary = validate_dataset(Path(args.path))
    print(json.dumps(summary, indent=2))
    return 0


def _bench(args: argparse.Namespace) -> int:
    print(load_bench(args.bench).describe())
    return 0


def _export_scene(args: argparse.Namespace) -> int:
    path = write_gltf(Path(args.out), load_bench(args.bench))
    print(f"wrote {path}")
    return 0


def _export_lerobot(args: argparse.Namespace) -> int:
    destination = export_lerobot(Path(args.src), args.repo_id, Path(args.out) if args.out else None)
    print(f"wrote {destination}")
    return 0


def _check(args: argparse.Namespace) -> int:
    bench = load_bench(args.bench)
    session = ReachySession(
        SessionConfig(
            ip_address=args.host,
            mock=args.mock,
            use_external_commands=True,
            use_head_cameras=not args.no_cameras,
            head_camera_width=160 if args.mock else 640,
            head_camera_height=120 if args.mock else 480,
        ),
        bench,
    )
    session.connect()
    try:
        present = session.read_present()
        goal = session.read_goal()
        print(f"connected mock={args.mock} host={args.host}")
        for name, measured, target in zip(joints_for(bench.arm), present, goal, strict=True):
            print(f"  {name:20s}  present {measured:8.2f}   goal {target:8.2f}")
        print(f"tip xyz m: {np.array2string(session.tip(), precision=3)}")
        if not args.no_cameras:
            for key, image in session.read_images().items():
                print(f"  camera {key}: {image.shape}")
    finally:
        session.close()
    return 0


def _record_external(args: argparse.Namespace, bench) -> int:
    session = ReachySession(
        SessionConfig(
            ip_address=args.host,
            mock=False,
            use_external_commands=True,
            head_camera_width=args.width,
            head_camera_height=args.height,
            table_camera_index=args.table_camera,
        ),
        bench,
    )
    session.connect()
    print("外部控制模式：本进程不给电机发目标。请用 Quest 里的官方遥操作程序带动 Reachy。")
    print("在这台电脑上按 Enter 开始一条，输入 q 再按 Enter 结束整个录制。")
    root = Path(args.out)
    saved = 0
    try:
        while saved < args.episodes:
            answer = input(f"[{saved + 1}/{args.episodes}] Enter 开始，q 退出: ").strip().lower()
            if answer == "q":
                break
            frames = _capture_seconds(session, args.seconds, bench.fps, use_goal=not args.present)
            _write_teleop_episode(root, bench, saved, frames, _ask_success())
            saved += 1
    finally:
        session.close()
    print(f"saved {saved} episode(s) to {root}")
    return 0


def _capture_seconds(session: ReachySession, seconds: float, fps: int, *, use_goal: bool) -> list[dict]:
    frames = []
    dt = 1.0 / fps
    deadline = time.perf_counter() + seconds
    next_tick = time.perf_counter()
    while time.perf_counter() < deadline:
        present = session.read_present()
        action = session.read_goal() if use_goal else present
        images = session.read_images()
        frames.append({"state": present, "action": action, "images": images, "t": len(frames) * dt})
        next_tick += dt
        delay = next_tick - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
    return frames


def _ask_success() -> bool:
    return input("这条示范成功了吗? [y/N] ").strip().lower() == "y"


def _write_teleop_episode(root: Path, bench, index: int, frames: list[dict], success: bool) -> None:
    if not frames:
        raise ValueError("episode has no frames")
    root.mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(exist_ok=True)
    (root / "meta").mkdir(exist_ok=True)
    state = np.stack([frame["state"] for frame in frames]).astype(np.float32)
    action = np.stack([frame["action"] for frame in frames]).astype(np.float32)
    images = {}
    for key in frames[0]["images"]:
        images[key] = np.stack([frame["images"][key] for frame in frames])
    payload = {"state": state, "action": action, "timestamp": np.array([frame["t"] for frame in frames], dtype=np.float32)}
    payload.update({f"image_{key}": value for key, value in images.items()})
    np.savez_compressed(root / "data" / f"episode_{index:06d}.npz", **payload)
    info_path = root / "meta" / "info.json"
    if index == 0 or not info_path.exists():
        info_path.write_text(
            json.dumps(
                {
                    "codebase_version": "studio-v1",
                    "kind": "teleop",
                    "robot_type": "reachy2023",
                    "fps": bench.fps,
                    "task": bench.task_en,
                    "task_zh": bench.task_zh,
                },
                indent=2,
            )
        )
    with (root / "meta" / "episodes.jsonl").open("w" if index == 0 else "a") as handle:
        handle.write(json.dumps({"episode_index": index, "length": len(frames), "success": success, "task": bench.task_en}) + "\n")


def _serve(args: argparse.Namespace) -> int:
    bench = load_bench(args.bench)
    mapper = QuestMapper(bench)
    receiver = QuestReceiver(args.bind, args.port)
    receiver.open()
    session = None
    if args.arm:
        session = ReachySession(
            SessionConfig(ip_address=args.host, mock=args.mock, use_external_commands=False, max_relative_target=args.max_delta),
            bench,
        )
        session.connect()
        session.set_stiff(True)
        print("电机已上力。按 Quest 右手 B 急停。")
    else:
        print("干跑：只解算关节，不连接电机。确认无误后再加 --arm。")
    print(f"listening UDP {args.bind}:{args.port}")
    last_log = 0.0
    try:
        while True:
            packet = receiver.poll()
            if packet is None:
                time.sleep(0.01)
                continue
            step = mapper.step(packet)
            if step.estop:
                print("急停。")
                if session is not None:
                    session.emergency_stop()
                return 2
            if session is not None and step.calibrated:
                session.write_goal(step.action)
            now = time.perf_counter()
            if now - last_log > 0.5:
                flag = "ok" if step.workspace_ok else "outside workspace"
                print(f"calibrated={step.calibrated} {flag} tip_error={step.tip_error_m * 100:.1f} cm")
                last_log = now
    except KeyboardInterrupt:
        print("停止。")
    finally:
        if session is not None:
            session.close()
        receiver.close()
    return 0


def _replay(args: argparse.Namespace) -> int:
    blob = np.load(Path(args.path) / "data" / f"episode_{args.episode:06d}.npz")
    print(f"episode {args.episode}  frames={blob['action'].shape[0]}  action_dim={blob['action'].shape[1]}")
    if not args.arm:
        print("没有加 --arm，只检查文件，不驱动电机。")
        return 0
    bench = load_bench(args.bench)
    session = ReachySession(
        SessionConfig(ip_address=args.host, mock=False, use_external_commands=False, max_relative_target=args.max_delta),
        bench,
    )
    session.connect()
    session.set_stiff(True)
    dt = 1.0 / bench.fps
    try:
        for action in blob["action"]:
            session.write_goal(action)
            time.sleep(dt)
    finally:
        session.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reachy-pour", description="Reachy 2023 拿起烧杯、倒出、放下的模仿学习准备工程")
    parser.add_argument("--bench", default=None, help="实验台 JSON，默认 configs/pour_bench.json")
    sub = parser.add_subparsers(dest="cmd", required=True)

    practice = sub.add_parser("practice", help="离线生成一条可检查的示范并写成数据集")
    practice.add_argument("--episodes", type=int, default=2)
    practice.add_argument("--out", default="datasets/practice")
    practice.add_argument("--seed", type=int, default=0)
    practice.add_argument("--noise", type=float, default=0.012, help="烧杯水平位置噪声，米")
    practice.add_argument("--image-scale", type=float, default=1.0)
    practice.set_defaults(func=_practice)

    validate = sub.add_parser("validate", help="检查练习数据集的阶段、夹爪和画面")
    validate.add_argument("path")
    validate.set_defaults(func=_validate)

    bench = sub.add_parser("bench", help="打印实验台尺寸，用来对照实物和 Blender")
    bench.set_defaults(func=_bench)

    scene = sub.add_parser("export-scene", help="导出实验台 glTF")
    scene.add_argument("--out", default="assets/pour_bench.gltf")
    scene.set_defaults(func=_export_scene)

    export = sub.add_parser("export-lerobot", help="把练习数据集转成 LeRobotDataset（需要已安装 lerobot）")
    export.add_argument("--src", default="datasets/practice")
    export.add_argument("--repo-id", default="local/reachy2023-pour")
    export.add_argument("--out", default=None)
    export.set_defaults(func=_export_lerobot)

    check = sub.add_parser("check", help="连接机器人并打印关节、末端和相机")
    check.add_argument("--host", default="192.168.0.42")
    check.add_argument("--mock", action="store_true")
    check.add_argument("--no-cameras", action="store_true")
    check.set_defaults(func=_check)

    record = sub.add_parser("record", help="录制。mock 走练习数据；external 只读官方 VR；quest 用自建 Unity")
    record.add_argument("--mode", choices=["mock", "external", "quest"], default="mock")
    record.add_argument("--host", default="192.168.0.42")
    record.add_argument("--episodes", type=int, default=2)
    record.add_argument("--seconds", type=float, default=20.0)
    record.add_argument("--out", default="datasets/record")
    record.add_argument("--width", type=int, default=320)
    record.add_argument("--height", type=int, default=240)
    record.add_argument("--table-camera", type=int, default=None)
    record.add_argument("--present", action="store_true", help="用 present_position 而不是 goal_position 作为 action")
    record.add_argument("--arm", action="store_true", help="quest 模式下允许本进程上力")
    record.add_argument("--bind", default="0.0.0.0")
    record.add_argument("--port", type=int, default=29023)
    record.add_argument("--max-delta", type=float, default=5.0)
    record.set_defaults(func=_record)

    serve = sub.add_parser("serve", help="接收 Quest UDP。默认干跑，加 --arm 才上力")
    serve.add_argument("--host", default="192.168.0.42")
    serve.add_argument("--bind", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=29023)
    serve.add_argument("--arm", action="store_true")
    serve.add_argument("--mock", action="store_true")
    serve.add_argument("--max-delta", type=float, default=5.0)
    serve.set_defaults(func=_serve)

    replay = sub.add_parser("replay", help="回放一条动作。默认只读文件，加 --arm 才驱动真机")
    replay.add_argument("path")
    replay.add_argument("--episode", type=int, default=0)
    replay.add_argument("--host", default="192.168.0.42")
    replay.add_argument("--arm", action="store_true")
    replay.add_argument("--max-delta", type=float, default=5.0)
    replay.set_defaults(func=_replay)
    return parser


def _record(args: argparse.Namespace) -> int:
    bench = load_bench(args.bench)
    if args.mode == "mock":
        practice = argparse.Namespace(
            bench=args.bench,
            episodes=args.episodes,
            out=args.out,
            seed=0,
            noise=0.0,
            image_scale=0.5,
        )
        return _practice(practice)
    if args.mode == "external":
        return _record_external(args, bench)
    return _record_quest(args, bench)


def _record_quest(args: argparse.Namespace, bench) -> int:
    mapper = QuestMapper(bench)
    receiver = QuestReceiver(args.bind, args.port)
    receiver.open()
    session = ReachySession(
        SessionConfig(
            ip_address=args.host,
            mock=not args.arm,
            use_external_commands=False,
            max_relative_target=args.max_delta,
            use_head_cameras=True,
            head_camera_width=args.width,
            head_camera_height=args.height,
        ),
        bench,
    )
    if args.arm or session.config.mock:
        session.connect()
    if args.arm:
        session.set_stiff(True)
    print("自建 Unity 模式。A 校准，Y 开始或结束一条，B 急停。")
    root = Path(args.out)
    saved = 0
    recording = False
    frames: list[dict] = []
    started = 0.0
    try:
        while saved < args.episodes:
            packet = receiver.poll()
            if packet is None:
                time.sleep(0.005)
                continue
            step = mapper.step(packet)
            if step.estop:
                print("急停。")
                session.emergency_stop()
                return 2
            if step.episode_edge:
                if not recording:
                    if not step.calibrated:
                        print("先按 A，站好不要移动双脚，再按 Y。")
                        continue
                    recording = True
                    frames = []
                    started = time.perf_counter()
                    print("开始录制。")
                else:
                    if frames:
                        _write_teleop_episode(root, bench, saved, frames, True)
                        saved += 1
                        print(f"结束 episode {saved}。")
                    recording = False
            if recording:
                if session.connected:
                    applied = session.write_goal(step.action)
                    state = session.read_present()
                    images = session.read_images()
                else:
                    applied = step.action
                    state = step.action
                    images = {}
                frames.append({"state": state, "action": applied, "images": images, "t": time.perf_counter() - started})
                if time.perf_counter() - started > args.seconds:
                    _write_teleop_episode(root, bench, saved, frames, True)
                    saved += 1
                    recording = False
                    print(f"达到 {args.seconds:.0f}s，自动结束 episode {saved}。")
    except KeyboardInterrupt:
        print("停止录制。")
    finally:
        session.close()
        receiver.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
