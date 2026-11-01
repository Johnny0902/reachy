#!/usr/bin/env bash
# Train ACT on a dataset already converted to LeRobot format, or recorded with lerobot-record.
set -euo pipefail
if ! command -v lerobot-train >/dev/null 2>&1; then
  echo "lerobot-train 不在 PATH 里。先按 README 安装 huggingface/lerobot，再安装本仓库的两个插件。" >&2
  exit 1
fi
REPO_ID="${REPO_ID:-local/reachy2023-pour}"
ROOT="${DATASET_ROOT:-datasets/lerobot/reachy2023_pour}"
DEVICE="${DEVICE:-cuda}"
lerobot-train \
  --dataset.repo_id="${REPO_ID}" \
  --dataset.root="${ROOT}" \
  --policy.type=act \
  --output_dir=outputs/train/reachy2023_pour \
  --job_name=reachy2023_pour \
  --policy.device="${DEVICE}" \
  --policy.repo_id="${REPO_ID}_act"
