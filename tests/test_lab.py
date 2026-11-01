import json
import socket
from pathlib import Path

import numpy as np
import pytest

from reachy_pour.bench import load_bench
from reachy_pour.cli import main
from reachy_pour.dataset import validate_dataset, write_dataset
from reachy_pour.demonstrator import PHASES, simulate_episode
from reachy_pour.driver import ReachySession, SessionConfig
from reachy_pour.gltf_export import build_gltf
from reachy_pour.ik import position_ik
from reachy_pour.joints import RIGHT_HOME_DEG
from reachy_pour.kinematics import tip_position
from reachy_pour.retarget import controller_to_hand, reachy_point_to_unity_relative
from reachy_pour.safety import clip_relative
from reachy_pour.teleop import QuestMapper, QuestReceiver, neutral_packet


def test_right_angle_pose_points_forward():
    tip = tip_position("right", [0, 0, 0, -90, 0, 0, 0])
    assert tip[0] > 0.30
    assert tip[2] < -0.20
    assert abs(tip[1]) < 0.05


def test_ik_returns_to_a_known_tip():
    target = tip_position("right", [10, -15, -10, -80, 5, -30, 10])
    solved, error = position_ik("right", target, RIGHT_HOME_DEG, max_iter=60)
    assert error < 0.01
    assert np.linalg.norm(tip_position("right", solved) - target) < 0.01


def test_relative_clip_limits_one_cycle():
    present = np.zeros(8)
    goal = np.full(8, 100.0)
    clipped = clip_relative(goal, present, 5.0)
    assert np.allclose(clipped, 5.0)


def test_quest_roundtrip_and_buttons():
    bench = load_bench()
    packet = neutral_packet(rx=0.15, ry=-0.20, rz=0.30, rtrig=1.0)
    mapper = QuestMapper(bench)
    first = mapper.step(packet)
    assert not first.calibrated
    calibrated = mapper.step(neutral_packet(a=1, rx=0.15, ry=-0.20, rz=0.30, rtrig=1.0))
    assert calibrated.calibrated
    assert "calibrated" in calibrated.events
    moved = mapper.step(neutral_packet(a=1, rx=0.15, ry=-0.20, rz=0.30, rtrig=1.0))
    assert moved.workspace_ok
    assert moved.tip_error_m < 0.03
    assert moved.action[7] == pytest.approx(bench.gripper_closed_deg)
    relative = reachy_point_to_unity_relative(
        controller_to_hand(neutral_packet(rx=0.15, ry=-0.20, rz=0.30), mapper.calibration, bench).position,
        bench,
    )
    assert np.allclose(relative, [0.15, -0.20, 0.30], atol=1e-6)
    assert mapper.step(neutral_packet(y=1)).episode_edge
    assert mapper.step(neutral_packet(b=1)).estop


def test_udp_packet_roundtrip():
    receiver = QuestReceiver("127.0.0.1", 0)
    receiver.open()
    try:
        port = receiver._sock.getsockname()[1]
        payload = json.dumps(neutral_packet(a=1)).encode()
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(payload, ("127.0.0.1", port))
        packet = None
        for _ in range(20):
            packet = receiver.poll()
            if packet is not None:
                break
        assert packet is not None
        assert packet["a"] == 1
    finally:
        receiver.close()


def test_episode_completes_the_pour(tmp_path: Path):
    bench = load_bench()
    episode = simulate_episode(bench, seed=1, cup_noise_m=0.0, image_scale=0.5)
    names = [name for name, _ in PHASES]
    seen = {names[int(index)] for index in episode.phase_id}
    assert {"approach", "grasp", "lift", "pour", "place", "release", "retreat"} <= seen
    lift = episode.phase_id == names.index("lift")
    assert float(episode.action[lift, 7].mean()) > 20.0
    assert episode.max_ik_error_m < 0.02
    assert episode.success
    root = write_dataset(tmp_path / "practice", bench, [episode])
    summary = validate_dataset(root)
    assert summary["episodes"] == 1
    assert (root / "preview.png").stat().st_size > 1000


def test_mock_session_does_not_jump():
    session = ReachySession(SessionConfig(mock=True, max_relative_target=5.0, use_head_cameras=False))
    session.connect()
    present = session.read_present()
    goal = present.copy()
    goal[:7] += 40.0
    applied = session.write_goal(goal)
    assert np.max(np.abs(applied - present)) <= 5.0 + 1e-6
    session.close()


def test_gltf_contains_the_props():
    document = build_gltf(load_bench())
    names = {node["name"] for node in document["nodes"]}
    assert {"Table", "Tray", "Beaker", "Bead_0", "ReachyOrigin"} <= names
    raw = document["buffers"][0]["uri"].split(",", 1)[1]
    import base64

    assert len(base64.b64decode(raw)) == document["buffers"][0]["byteLength"]


def test_cli_practice_and_bench(tmp_path: Path, capsys):
    assert main(["--bench", "configs/pour_bench.json", "bench"]) == 0
    text = capsys.readouterr().out
    assert "烧杯" in text
    assert main(
        [
            "practice",
            "--episodes",
            "1",
            "--noise",
            "0",
            "--image-scale",
            "0.45",
            "--out",
            str(tmp_path / "cli"),
        ]
    ) == 0
    assert (tmp_path / "cli" / "meta" / "info.json").exists()


def test_check_mock():
    assert main(["check", "--mock", "--no-cameras"]) == 0
