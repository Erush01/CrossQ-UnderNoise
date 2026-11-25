#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR="$ROOT/artifacts/logs"
PREFLIGHT="$ROOT/scripts/preflight_pilot.py"
CONFIG="$ROOT/configs/pilot_humanoidstandup_100k.json"
PYTHON="${CROSSQ_PYTHON:-$ROOT/../ARS/.venv/bin/python}"
DRY_RUN=0
ATTEMPT2=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --attempt2) ATTEMPT2=1 ;;
    *) echo "usage: $0 [--dry-run] [--attempt2]" >&2; exit 2 ;;
  esac
done
TARGET_NAME="HumanoidStandup-seed0"
if [[ "$ATTEMPT2" -eq 1 ]]; then TARGET_NAME="HumanoidStandup-seed0-attempt2"; fi
TARGET="$ROOT/artifacts/pilot/$TARGET_NAME"
if [[ -e "$TARGET" ]]; then echo "Refusing existing pilot target: $TARGET" >&2; exit 1; fi
if [[ ! -x "$PYTHON" ]]; then echo "CrossQ Python runtime is not executable: $PYTHON (set CROSSQ_PYTHON to override)" >&2; exit 1; fi
mkdir -p "$LOGDIR"
STAMP="$(date -u +%Y%m%dT%H%M%S%N)"
PREFLIGHT_JSON="$LOGDIR/pilot-preflight-$STAMP.json"
PREFLIGHT_LOG="$LOGDIR/pilot-preflight-$STAMP.log"
if [[ -e "$PREFLIGHT_JSON" || -e "$PREFLIGHT_LOG" ]]; then echo "Refusing to overwrite preflight artifacts" >&2; exit 1; fi
"$PYTHON" "$PREFLIGHT" --target "$TARGET" --output "$PREFLIGHT_JSON" >"$PREFLIGHT_LOG" 2>&1 || { cat "$PREFLIGHT_LOG"; exit 1; }
if [[ "$DRY_RUN" -eq 1 ]]; then
  printf 'DRY RUN: HumanoidStandup-v4 seed=0 steps=100000 device=auto\n'
  if [[ "$ATTEMPT2" -eq 1 ]]; then printf 'Attempt: 2 target: %s\n' "$TARGET"; fi
  printf 'Preflight: %s\n' "$PREFLIGHT_JSON"
  exit 0
fi
RUN_LOG="$LOGDIR/pilot-run-$STAMP.log"
TIME_LOG="$LOGDIR/pilot-time-$STAMP.txt"
if [[ -e "$RUN_LOG" || -e "$TIME_LOG" ]]; then echo "Refusing to overwrite run statistics" >&2; exit 1; fi
set +e
/usr/bin/time -v -o "$TIME_LOG" "$PYTHON" -c 'import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); from crossq_repro.runner import run; run(sys.argv[2], "HumanoidStandup-v4", 0, 100000, sys.argv[3], device="auto", provenance_paths=sys.argv[4:])' \
  "$ROOT" "$CONFIG" "$TARGET" "$0" "$PREFLIGHT" >"$RUN_LOG" 2>&1
STATUS=$?
set -e
exit "$STATUS"
