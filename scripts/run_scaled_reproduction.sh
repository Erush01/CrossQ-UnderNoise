#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOBS_SCRIPT="$ROOT/scripts/scaled_reproduction_jobs.py"
PYTHON="${CROSSQ_PYTHON:-$ROOT/../ARS/.venv/bin/python}"
DRY_RUN=0
CONFIRM=0
BATCH=""
SINGLE=""
SEED=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --confirm-scaled-run) CONFIRM=1; shift ;;
    --batch)
      [[ $# -ge 2 ]] || { echo "--batch requires seed0 or remaining" >&2; exit 2; }
      BATCH="$2"
      shift 2
      ;;
    --single)
      [[ $# -ge 2 ]] || { echo "--single requires Hopper" >&2; exit 2; }
      SINGLE="$2"
      shift 2
      ;;
    --seed)
      [[ $# -ge 2 ]] || { echo "--seed requires 0" >&2; exit 2; }
      SEED="$2"
      shift 2
      ;;
    *) echo "usage: $0 --dry-run | --batch seed0|remaining --confirm-scaled-run | --single Hopper --seed 0 [--dry-run|--confirm-scaled-run]" >&2; exit 2 ;;
  esac
done

if [[ -n "$SINGLE" ]]; then
  if [[ -n "$BATCH" || ( "$SINGLE" != "Hopper" && "$SINGLE" != "HalfCheetah" ) || "$SEED" != "0" ]]; then
    echo "Single retry supports only Hopper or HalfCheetah seed 0 and cannot be combined with --batch" >&2
    exit 2
  fi
  if [[ "$DRY_RUN" -eq 1 ]]; then
    if [[ "$CONFIRM" -eq 1 ]]; then
      echo "--dry-run cannot be combined with --confirm-scaled-run" >&2
      exit 2
    fi
    exec "$PYTHON" "$JOBS_SCRIPT" --single "$SINGLE" --seed "$SEED" --dry-run
  fi
  if [[ "$CONFIRM" -ne 1 ]]; then
    echo "Refusing single retry without --confirm-scaled-run" >&2
    exit 2
  fi
  if [[ -z "${TMUX:-}" ]]; then
    echo "Refusing live single retry outside an existing tmux session" >&2
    exit 2
  fi
  ACTUAL_SESSION="$(tmux display-message -p '#{session_name}' 2>/dev/null || true)"
  EXPECTED_SESSION="cmp620-crossq-scaled-single-$SINGLE-seed$SEED"
  if [[ "$ACTUAL_SESSION" != "$EXPECTED_SESSION" ]]; then
    echo "Refusing single retry: requires tmux session $EXPECTED_SESSION, got ${ACTUAL_SESSION:-none}" >&2
    exit 2
  fi
  if [[ ! -x "$PYTHON" ]]; then
    echo "CrossQ Python runtime is not executable: $PYTHON" >&2
    exit 1
  fi
  exec "$PYTHON" "$JOBS_SCRIPT" --single "$SINGLE" --seed "$SEED" --confirm-scaled-run
fi

if [[ -n "$SEED" ]]; then
  echo "--seed requires --single Hopper" >&2
  exit 2
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  if [[ "$CONFIRM" -eq 1 || -n "$BATCH" ]]; then
    echo "--dry-run lists the entire matrix and cannot be combined with launch options" >&2
    exit 2
  fi
  exec "$PYTHON" "$JOBS_SCRIPT" --dry-run
fi

if [[ "$CONFIRM" -ne 1 || ( "$BATCH" != "seed0" && "$BATCH" != "remaining" ) ]]; then
  echo "Refusing scaled launch; explicitly select --batch seed0|remaining --confirm-scaled-run" >&2
  exit 2
fi
if [[ -z "${TMUX:-}" ]]; then
  echo "Refusing live scaled execution outside an existing tmux session" >&2
  exit 2
fi
ACTUAL_SESSION="$(tmux display-message -p '#{session_name}' 2>/dev/null || true)"
EXPECTED_SESSION="cmp620-crossq-scaled-$BATCH"
if [[ "$ACTUAL_SESSION" != "$EXPECTED_SESSION" ]]; then
  echo "Refusing batch $BATCH: requires tmux session $EXPECTED_SESSION, got ${ACTUAL_SESSION:-none}" >&2
  exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "CrossQ Python runtime is not executable: $PYTHON" >&2
  exit 1
fi
exec "$PYTHON" "$JOBS_SCRIPT" --run-batch "$BATCH" --confirm-scaled-run
