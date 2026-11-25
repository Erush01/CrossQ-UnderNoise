#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
CONFIG="$ROOT/configs/smoke_mujoco.json"
OUTPUT_ROOT="${CROSSQ_SMOKE_OUTPUT_ROOT:-$ROOT/runs/engineering-smoke}"
PYTHON="${CROSSQ_PYTHON:-$ROOT/../ARS/.venv/bin/python}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
elif [[ $# -eq 0 ]]; then
  DRY_RUN=0
else
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi

TASKS=(Hopper Walker2d Ant Humanoid HalfCheetah HumanoidStandup)
ENV_IDS=(Hopper-v4 Walker2d-v4 Ant-v4 Humanoid-v4 HalfCheetah-v4 HumanoidStandup-v4)
TARGETS=()
EXISTING=0
for index in "${!TASKS[@]}"; do
  target="$OUTPUT_ROOT/${TASKS[$index]}-seed0"
  TARGETS+=("$target")
  echo "ENGINEERING SMOKE ONLY: ${TASKS[$index]} (${ENV_IDS[$index]}), seed=0, steps=1000, eval=1 env x 1 episode every 500 steps -> $target"
  if [[ -e "$target" ]]; then
    echo "ERROR: target output already exists: $target" >&2
    EXISTING=1
  fi
done

if [[ "$EXISTING" -ne 0 ]]; then
  exit 1
fi
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run only; no smoke jobs launched."
  exit 0
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Python executable is not runnable: $PYTHON (set CROSSQ_PYTHON to override)" >&2
  exit 1
fi

for index in "${!TASKS[@]}"; do
  "$PYTHON" -c 'import sys; from crossq_repro.runner import run; run(sys.argv[1], sys.argv[2], 0, 1000, sys.argv[3])' \
    "$CONFIG" "${ENV_IDS[$index]}" "${TARGETS[$index]}"
done
