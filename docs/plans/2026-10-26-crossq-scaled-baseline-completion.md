# Complete the Scaled CrossQ Baseline — Implementation Plan

> **Superseded by the latest proposal:** `CrossQ/docs/crossq.pdf` changes the project target to a five-seed Hopper reproduction followed by action-noise robustness evaluation. Retain this file as the historical 3-task × 3-seed plan; do not execute it.

**Goal:** Complete the existing CrossQ-only three-task × three-seed × 5M-step scaled baseline without rerunning the two already validated runs.

**Architecture:** Keep the frozen scaled manifest and current validated artifacts in `CrossQ/artifacts/scaled_reproduction_retry/`. Add a new single-task entry point for the remaining HumanoidStandup seed-0 run, while leaving existing source files whose hashes are recorded in the completed artifacts unchanged. After seed 0 is complete and reviewed, use the existing serial `remaining` batch implementation against the same retry root, only after separate approval.

**Tech Stack:** Python 3.12, SB3-Contrib/Stable-Baselines3 2.9.0, Gymnasium 1.4.0, MuJoCo 3.16.0, pytest, NumPy, existing read-only `ARS/.venv`.

---

## Claim target and limits

- This plan completes a **scaled CrossQ-only baseline**. It does not test the paper's comparative CrossQ-vs-REDQ/DroQ claim because there are no baseline algorithm arms.
- The official paper reports six MuJoCo tasks, ten seeds, and a 5M-step endpoint. The selected scaled matrix uses three tasks—Hopper, HalfCheetah, HumanoidStandup—and seeds 0–2. Label all results as scaled/exploratory, not a full reproduction.
- Keep the frozen Table 1 mappings unchanged. Evaluation remains the modern SB3 choice (25k steps, 20 episodes, five eval envs); the paper does not specify this exact cadence.
- The paper-claim audit was refreshed on 2026-10-26 at `CrossQ/docs/research/crossq-paper-claims.md`.

## Verified run inventory

| Task / seed | State | Evidence |
|---|---|---|
| Hopper-v4 / 0 | Complete, 5M | `CrossQ/artifacts/scaled_reproduction_retry/Hopper-seed0/`; strict validation passed. |
| HalfCheetah-v4 / 0 | Complete, 5M | `CrossQ/artifacts/scaled_reproduction_retry/HalfCheetah-seed0/`; strict validation passed. |
| HumanoidStandup-v4 / 0 | Not run at 5M | There is a separate 100k resource/stability pilot; it does not count as this run. |
| All three tasks / seeds 1 and 2 | Not run | Six additional 5M jobs remain. |

The earlier 400k Hopper partial under `CrossQ/artifacts/scaled_reproduction/Hopper-seed0/` is preserved and excluded. Existing completed-run source hashes include `scaled_reproduction_jobs.py` and `run_scaled_reproduction.sh`; do not modify those files during this plan unless a source-snapshot migration is separately designed and tested.

## Remaining workload and resources

- Remaining: 7 runs × 5M = **35M environment steps**.
- With 200 evaluation points ×20 episodes per run: **1,400 evaluation points and 28,000 episodes** remain.
- Observed completed 5M runtimes: Hopper seed 0 = **21.96 h**; HalfCheetah seed 0 = **20.48 h**. Uniform-rate projection for seven runs is about **148.5 hours (~6.2 days) serial**. This is a rough projection from two tasks and one seed each; HumanoidStandup and seed variation are not measured at 5M.
- Approximate remaining checkpoint storage is **1.7 GB** based on the current per-run checkpoint size; require the existing 20 GiB free-disk threshold and rerun RAM/CUDA/environment preflight immediately before each authorized batch.
- One training process at a time. Stop on the first failure; preserve partial outputs; no automatic retry, resume, or budget increase.

## Task 1: Add a hash-safe HumanoidStandup seed-0 entry point — no training

**Files:**
- Create `CrossQ/scripts/run_scaled_completion.py`.
- Create `CrossQ/tests/test_scaled_completion.py`.
- Do not modify `CrossQ/scripts/scaled_reproduction_jobs.py` or `CrossQ/scripts/run_scaled_reproduction.sh` because those exact source hashes are recorded in the two completed runs.

**Steps:**
1. Write failing tests for exactly one allowed job: `HumanoidStandup-v4`, seed 0, 5M steps, output `CrossQ/artifacts/scaled_reproduction_retry/HumanoidStandup-seed0`.
2. Test refusal without explicit confirmation, wrong task/seed, mismatched named tmux session, package mismatch, existing target, insufficient resource preflight, and failed environment/API check.
3. Test dry-run prints exactly the one HumanoidStandup job and creates no target directory; test the execution seam with a stubbed runner so no environment steps occur.
4. Implement the minimal entry point using the existing frozen config and runner, with RAM/disk/CUDA preflight, configured environment checks, unique logs, strict post-run validation, and this tmux name: `cmp620-crossq-scaled-single-HumanoidStandup-seed0`.
5. Include the new helper’s hash in this run’s provenance. Preserve the recorded hashes of existing source files.
6. Run `PYTHONPATH=ARS/.venv/lib/python3.12/site-packages pytest -q CrossQ/tests` and the helper’s dry-run. Expected: tests pass, exactly one planned job, target absent, no training process.

## Task 2: HumanoidStandup seed-0 run — separate user approval required

1. Before launch, recheck the target is absent, the named tmux session is absent, no scaled job is active, versions match, and the live RAM/disk/CUDA/API preflight passes.
2. Start only HumanoidStandup-v4 seed 0 for 5M steps in the named tmux session.
3. Require metadata `status=completed`, requested/actual steps 5M, 200 finite evaluation points ×20 episodes, expected config/source hashes, and loadable best/final checkpoints with critic widths `[2048, 2048, 1]`.
4. Stop on any error and preserve artifacts. Do not retry without another explicit user decision.
5. Present the three seed-0 task results and pause for review.

## Task 3: Seeds 1–2 — separate user approval required after Task 2 review

The existing Python batch runner supports `--output-root`; use the retry root so the two completed seed-0 artifacts are reused. Do not edit the hash-recorded batch launcher.

1. After explicit approval, verify all three seed-0 runs in `CrossQ/artifacts/scaled_reproduction_retry/` with the strict validator.
2. Launch exactly the six remaining jobs sequentially under `cmp620-crossq-scaled-remaining`:

```bash
cd /home/erush/CMP620/CrossQ
/home/erush/CMP620/ARS/.venv/bin/python scripts/scaled_reproduction_jobs.py \
  --run-batch remaining \
  --config configs/scaled_3tasks_3seeds_5m.yaml \
  --output-root artifacts/scaled_reproduction_retry \
  --confirm-scaled-run
```

3. Validate each run before the queue advances. On failure, stop and preserve all artifacts; no automatic retry or resume.

## Task 4: Aggregate only after 9/9 strict validation

1. From `CrossQ/`, run the all-nine summary into a separate results directory:

```bash
/home/erush/CMP620/ARS/.venv/bin/python scripts/summarize_scaled_reproduction.py \
  --batch all \
  --config configs/scaled_3tasks_3seeds_5m.yaml \
  --output-root artifacts/scaled_reproduction_retry \
  --results-dir results/scaled_reproduction_retry
```

2. Preserve all seed-level curves. Report per-task, per-seed scores and across-seed uncertainty; never treat 20 evaluation episodes as 20 independent training seeds.
3. The paper's Figure 5 uses IQM with 70% quantile intervals, while the current local summary reports mean/std. Before claiming a paper-statistic match, verify the exact aggregation procedure and add it with tests; otherwise report the local statistic and difference explicitly.
4. State that the result remains a 3-task/3-seed CrossQ-only scaled baseline and does not test the REDQ/DroQ UTD=20 comparison.

## Current approval state

This plan records the next work only. **No HumanoidStandup or seed-1/2 training is authorized by this document.** Obtain separate user approval before Task 1 implementation, Task 2 launch, and Task 3 launch as specified above. CMP620 is not a Git repository; do not initialize one or claim a commit.
