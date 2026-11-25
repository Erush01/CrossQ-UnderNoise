# CrossQ Scaled Reproduction Implementation Plan

> **Historical plan:** Initial implementation design from 2026-10-24. Its consent and progress statements describe the state at creation. The subsequent 3-task plan is also superseded by `CrossQ/docs/crossq.pdf` and its proposal-aligned execution plan.

## Status update — 2026-10-26

- The guarded launcher, matrix, validator, and summary tooling were implemented and tested; the 9-job dry-run passed.
- Strictly validated 5M CrossQ runs now exist for Hopper-v4 seed 0 and HalfCheetah-v4 seed 0 under `CrossQ/artifacts/scaled_reproduction_retry/`.
- The 400k Hopper attempt under `CrossQ/artifacts/scaled_reproduction/Hopper-seed0/` remains incomplete and is excluded.
- The selected scaled 3-task × 3-seed baseline is 2/9 complete. No REDQ/DroQ/SAC comparison arms have been run.
- The remaining execution plan and approval gates are recorded in `CrossQ/docs/plans/2026-10-26-crossq-scaled-baseline-completion.md`; no remaining jobs are authorized by that plan alone.

**Goal:** Prepare a safe, reproducible CrossQ-only experiment covering Hopper, HalfCheetah, and HumanoidStandup, seeds 0–2, with the paper's 5M-step horizon.

**Architecture:** Add a frozen scaled manifest, a deterministic nine-job matrix, and a guarded sequential launcher that reuses the existing runner and artifact pipeline. Validate every 5M run before advancing; aggregate seed-level results only after the authorized batch completes.

**Tech Stack:** Python 3.12, SB3-Contrib/Stable-Baselines3 2.9.0, Gymnasium 1.4.0, MuJoCo 3.16.0, pytest, NumPy; existing `ARS/.venv` is read-only.

---

## Scope and consent gates

- **Approved design:** CrossQ only; `Hopper-v4`, `HalfCheetah-v4`, `HumanoidStandup-v4`; seeds `0,1,2`; 5,000,000 environment steps per run; nine total runs.
- Preserve the Table 1 settings in `CrossQ/configs/paper_mujoco.yaml`. Keep the existing modern evaluation protocol (25k steps, 20 episodes, five eval envs), labelled as not paper-specified.
- This subset is exploratory, not a full reproduction: it omits three tasks, seven seeds per task, and all paper baseline runs. It cannot independently validate the relative CrossQ-vs-REDQ/DroQ performance claim.
- The 100k HumanoidStandup seed-0 pilot is resource/stability evidence only. It is not part of these nine jobs. All nine runs start from scratch; replay/RNG state was not saved, so do not resume the pilot checkpoint.
- The user approved the design and this plan only. No implementation or training is authorized yet. Obtain separate approval before implementing the launcher, and separate approval before the first full-length training batch.
- Keep `ARS/.venv` read-only; no package installation, version substitution, or algorithm modification. CMP620 is not a Git repo, so do not initialize one or claim a commit.

## Budget estimate

- Total: `3 × 3 × 5M = 45M` training environment steps.
- Each run: 200 evaluations at 25k intervals; 20 episodes per evaluation.
- Nine runs: 1,800 evaluation points and 36,000 episodes.
- At the measured 100k HumanoidStandup pilot rate, one 5M run is about 21.49 hours and nine serial runs about 193.4 GPU-hours (~8 days). This is a rough extrapolation from one task/seed, not a validated per-task estimate.
- Two model checkpoints at the observed pilot sizes imply about 2.21 GB for nine runs, excluding logs, evaluations, configs, and partial outputs.
- Repeat disk/RAM/GPU preflight immediately before each approved batch. Use one training process at a time; no concurrent GPU jobs.

## Task 1: Frozen scaled manifest and matrix tests (no training)

**Files:**
- Create `CrossQ/configs/scaled_3tasks_3seeds_5m.yaml`.
- Create `CrossQ/tests/test_scaled_reproduction.py`.

**Steps:**
1. Add failing tests requiring exactly the three approved task IDs, seeds `[0,1,2]`, 5M steps, the existing Table 1 settings, and modern evaluation settings.
2. Run `PYTHONPATH=ARS/.venv/lib/python3.12/site-packages python -m pytest CrossQ/tests/test_scaled_reproduction.py -q`; confirm the new tests fail because the scaled manifest/matrix helper is absent.
3. Create a separate JSON-compatible YAML manifest by copying the approved settings and narrowing only tasks/seeds. Do not edit `paper_mujoco.yaml`.
4. Add a deterministic job builder that returns nine unique `(task, seed)` runs and paths `CrossQ/artifacts/scaled_reproduction/<Task>-seed<N>/`.
5. Re-run the focused tests; expect nine jobs, no duplicate IDs/paths, and no implicit pilot inclusion.

## Task 2: Guarded sequential launcher and strict per-run checks (no training)

**Files:**
- Create `CrossQ/scripts/scaled_reproduction_jobs.py`.
- Create `CrossQ/scripts/run_scaled_reproduction.sh`.
- Extend `CrossQ/tests/test_scaled_reproduction.py`.

**Steps:**
1. Add tests that default launcher invocation refuses; `--dry-run` prints all nine fixed jobs and creates no run targets; an existing target or active tmux session refuses the entire launch before job one.
2. Add a mandatory `--confirm-scaled-run` flag for training. It must never be set by default, environment variable, or dry-run.
3. Implement deterministic serial execution with a unique detached tmux session and per-job run/time logs; do not launch more than one job at once.
4. After each run, require metadata `status=completed`, requested and actual steps both exactly 5M, expected task/seed/config/source hashes and package versions, and 200 evaluation points `[25k, 50k, …, 5M]` with 20 finite returns and positive episode lengths each.
5. Load both best/final checkpoints and verify two critics with `[2048,2048,1]` linear widths.
6. On the first preflight, runtime, hash, metric, or checkpoint failure: stop the queue, preserve partial artifacts, and do not retry/resume automatically.
7. Run all tests, environment checks, and the scaled launcher dry-run. Expected: nine planned jobs, zero training, zero run-target directories created.

## Task 3: First authorized batch — seed 0 across the three tasks

This 15M-step batch is **not currently authorized**.

1. After explicit user consent, re-check disk/RAM/CUDA capacity, all nine output paths, package versions, and tmux/process state.
2. Run only Hopper, HalfCheetah, and HumanoidStandup seed 0, one at a time, each from scratch for 5M steps.
3. Strictly validate each run before starting the next. If any run fails, stop and preserve its artifacts; do not automatically resume or replace it.
4. Report the three measured runtimes, peak RSS, disk use, evaluation curves, and updated estimate for the remaining six runs.
5. Pause for user review and separate approval before seeds 1 and 2.

## Task 4: Seeds 1–2 after separate approval

1. Re-run dry-run and resource checks; require exactly six remaining jobs and no path/session collision.
2. Run the six jobs sequentially, validating each before the next.
3. Stop on first failure; no automatic retry, resume, or budget change.
4. Do not continue to the full 6×10 matrix under this plan.

## Task 5: Scaled result summary

**Files:**
- Create `CrossQ/scripts/summarize_scaled_reproduction.py`.
- Extend `CrossQ/tests/test_scaled_reproduction.py`.

**Steps:**
1. Test rejection of missing tasks/seeds, partial steps, missing evaluation points, non-finite returns, config/source mismatch, and invalid checkpoints.
2. When all nine artifacts validate, retain every seed's curve and final 20-episode score; compute per-task across-seed mean/std and, if the paper statistic is implemented, report it with its three-seed limitation. Never pool episode observations as independent training seeds.
3. Write `CrossQ/results/scaled_reproduction/summary.json` and `report.md` with versions, settings, deviations, per-seed results, runtime/resources, and explicit evidence limits.
4. Require 9/9 complete runs for a complete matrix summary. Otherwise report incomplete status without imputing missing results.

## Verification commands (after implementation)

Run from `/home/erush/CMP620` using the existing environment:

```bash
PYTHONPATH=ARS/.venv/lib/python3.12/site-packages python -m pytest CrossQ/tests -q
CROSSQ_PYTHON=ARS/.venv/bin/python CrossQ/scripts/check_envs.py --config CrossQ/configs/scaled_3tasks_3seeds_5m.yaml
CROSSQ_PYTHON=ARS/.venv/bin/python bash CrossQ/scripts/run_scaled_reproduction.sh --dry-run
```

Expected dry-run: nine unique jobs, each 5M steps, seeds 0–2; no training process and no run-target directory created.

## Reference documents

- Approved design: `CrossQ/docs/plans/2026-10-24-crossq-scaled-reproduction-design.md`.
- Paper/config mapping: `CrossQ/configs/paper_mujoco.yaml`.
- Claim audit: `CrossQ/docs/research/crossq-paper-claims.md`.
- Pilot protocol/evidence: `CrossQ/docs/pilot-protocol.md` and `CrossQ/artifacts/pilot/HumanoidStandup-seed0-attempt2/`.
