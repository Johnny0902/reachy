#!/usr/bin/env bash
# Run a trained ACT policy on the real Reachy 2023. The Quest app must NOT be commanding the arm.
set -euo pipefail
if ! command -v lerobot-eval >/dev/null 2>&1; then
  echo "lerobot-eval 不在 PATH 里。先安装 huggingface/lerobot 和本仓库的 reachy2023 插件。" >&2
  exit 1
fi
: "${REACHY_HOST:?set REACHY_HOST to the robot IP}"
POLICY="${POLICY:-outputs/train/reachy2023_pour/checkpoints/last/pretrained_model}"
lerobot-eval \
  --robot.type=reachy2023 \
  --robot.ip_address="${REACHY_HOST}" \
  --robot.use_external_commands=false \
  --robot.torque_on_connect=true \
  --robot.max_relative_target=5.0 \
  --robot.arm=right \
  --dataset.repo_id="${HF_USER:-local}/reachy2023_pour_eval" \
  --dataset.single_task="Pick up the beaker, pour the objects into the tray, and put the beaker down." \
  --dataset.num_episodes=10 \
  --policy.path="${POLICY}"
