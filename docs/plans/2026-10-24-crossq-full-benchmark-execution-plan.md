# CrossQ Full Benchmark Execution Plan

> **For Codex:** Use the `executing-plans` skill task-by-task after explicit user authorization. This file is a plan only; do not start training while approval is pending.

**Goal:** Run a version-pinned, six-task × ten-seed × 5M-step CrossQ benchmark under the already approved SB3-Contrib 2.9.0 protocol, then aggregate and compare its results with the CrossQ paper.

**Architecture:** Reuse the frozen manifest and single-run runner. Add a tested, sequential job-matrix launcher and a strict aggregation/validation script; preserve each task/seed as an independent result. Launch only in explicit batches, validate after every run, and stop on any incomplete or mismatched artifact.

**Tech Stack:** Python 3.12, SB3-Contrib/Stable-Baselines3 2.9.0, Gymnasium 1.4.0, MuJoCo 3.16.0, pytest, NumPy; existing `ARS/.venv` is read-only.

---

## Approval and scope gates

- **Current user choice:** Prepare this execution plan only. No full-benchmark training is authorized or started.
- **Current matrix:** CrossQ only; six paper tasks, seeds `0–9`, `5,000,000` environment steps per run. This does **not** reproduce the paper's SAC/REDQ/DroQ/other baseline comparisons or ablations. Before implementation or launch, confirm that a CrossQ-only reproduction is the intended deliverable. Baselines add separate algorithms, validation, and compute.
- **Separate launch consent:** Do not create or start the full run queue until the user explicitly approves it. Recommended staged consent: first authorize the six seed-0 full-length jobs (30M steps); after those validate and update the cost estimate, ask again before seeds `1–9` (54 additional jobs, 270M steps).
- **No automatic recovery:** The training replay buffer and RNG state are not checkpointed. A partial run cannot be resumed exactly. Preserve its output, stop the queue, diagnose, and ask before starting a fresh replacement run.
- **Environment/version scope:** Keep `CrossQ/configs/paper_mujoco.yaml` frozen. Do not install packages or modify `ARS/.venv`; do not substitute `-v5` for the paper-configured `-v4` environments or change hyperparameters silently.
- **Workspace constraint:** CMP620 is not a Git repository. Do not initialize a repository or claim a commit; keep all generated scripts, plans, and results under `CrossQ/`.

## Measured cost and resource envelope

### Observed evidence

The completed `HumanoidStandup-v4`, seed-0, 100k pilot used the same 25k / 20-episode / five-evaluation-env cadence planned for the full runs:

- Runtime: `1,547.127777 s` (`25m 47.13s`) for 100,000 steps, including four evaluations.
- Maximum process RSS: `2,437,420 kB`; no swaps; device resolved to CUDA.
- Two checkpoints: `final_model.zip` 122,595,394 bytes and `best_model.zip` 122,595,366 bytes.
- Pilot target: `CrossQ/artifacts/pilot/HumanoidStandup-seed0-attempt2/`.

The six 1k engineering smokes took the following times, but used one evaluation episode/one evaluation env every 500 steps. They are not suitable for full-run timing forecasts.

| Task | Smoke runtime |
|---|---:|
| Ant | 71.14 s |
| HalfCheetah | 63.83 s |
| Hopper | 63.87 s |
| Humanoid | 57.80 s |
| HumanoidStandup | 69.98 s |
| Walker2d | 57.88 s |

### Transparent arithmetic (planning estimate only)

The full matrix has `6 × 10 = 60` runs and `60 × 5,000,000 = 300,000,000` training environment steps.

Using only the HumanoidStandup pilot rate as a uniform scenario:

```text
1 run at 5M       = 1,547.127777 s × 50 = 77,356.38885 s = 21.4879 h
60 sequential runs = 77,356.38885 s × 60 = 4,641,383.331 s
                   = 1,289.27 GPU-hours ≈ 53.72 days
```

This is **not** a validated task-by-task forecast or a guaranteed upper bound. It extrapolates 50× beyond one 100k run, assumes the other environments and seeds have comparable throughput, and assumes no interruptions or resource contention. The six smokes are too short and used a different evaluation schedule to improve that estimate reliably. Record actual per-run time and revise the estimate after the seed-0 batch.

### Evaluation and storage

- At 5M steps and 25k evaluation frequency: `200` evaluation checkpoints per run; `12,000` checkpoints across 60 runs.
- With 20 episodes per evaluation: `4,000` evaluation episodes per run; `240,000` total episodes. This frequency/episode count is a modern protocol, not paper-specified.
- At the pilot checkpoint sizes: two ZIPs cost about `245,190,760 bytes/run`; 60 runs need about `14.71 GB` for model ZIPs alone. Budget extra space for logs, configs, evaluation arrays, and partial failures; preflight must require at least `20 GiB` free before launch.
- HumanoidStandup's one-million-entry replay estimate is `3,084,000,000` bytes. Preflight must calculate each task's observation/action-based estimate and require available RAM of at least `1.5×` the largest estimate.
- Use **one training process at a time** on the measured GPU. The pilot preflight's free CUDA memory was about `11.55 GB`; it checks a minimum of `2 GiB`, but concurrent runs are not authorized or estimated.
- All resource values are point-in-time observations. Recheck disk, RAM, GPU memory, target paths, and tmux session immediately before each approved batch.

## Execution stages

### Stage 0: Confirm scope and freeze inputs

1. Confirm the deliverable is CrossQ-only or revise the design and cost for baseline comparisons; do not assume.
2. Re-run the existing test suite and six environment checks with the existing ARS Python environment.
3. Record a fresh hash of the manifest, runner, evaluation, validation, and launcher sources. Preserve the current `paper_mujoco.yaml` unchanged.
4. Generate a dry-run matrix and require exactly 60 unique `(environment, seed)` jobs, each with 5M steps, seeds `0–9`, expected evaluation settings, and a unique output directory.
5. Dry-run must refuse if any target exists, an existing benchmark tmux session is active, or versions/config differ. Dry-run must not create run targets or start training.

### Stage 1: Implement guarded batch execution (no training during implementation)

**Planned files:**

- Create `CrossQ/scripts/full_benchmark_jobs.py` to construct and validate the 60-job matrix from the frozen manifest.
- Create `CrossQ/scripts/run_full_benchmark.sh` with `--dry-run`, `--batch seed0`, and an explicit `--confirm-full-matrix` gate for the remaining seeds. It must be sequential, refuse overwrites, and never retry/resume automatically.
- Create `CrossQ/scripts/summarize_full_benchmark.py` to validate per-run artifacts and aggregate seed-level results.
- Create `CrossQ/tests/test_full_benchmark.py` for matrix cardinality, unique IDs/paths, fixed config, dry-run safety, refusal cases, per-run completeness, and aggregation semantics.
- Update `CrossQ/README.md` with the exact CrossQ-only scope and paper-vs-modern deviations.

**Run target convention:** `CrossQ/artifacts/full_benchmark/<TaskName>-seed<N>/`, for example `CrossQ/artifacts/full_benchmark/Hopper-seed0/`.

**Steps:**

1. Write tests asserting exactly 60 jobs, six configured env IDs, ten seeds per env, 5M steps, `eval_freq=25,000`, `n_eval_episodes=20`, and unique non-existing targets. Expected: tests fail before the matrix helper exists.
2. Implement the minimal deterministic matrix helper and run the focused tests. Expected: 60 jobs and zero duplicates.
3. Add launcher tests: default invocation refuses; `--dry-run` prints the plan only; live launch requires the explicit consent flag, and any existing target blocks the whole batch before job one.
4. Implement the sequential launcher. Use a unique detached tmux session so the workload survives terminal/server-session loss; do not exceed one active run.
5. Add per-job completion records and `time -v` logs. A job is complete only when runner metadata reports requested and actual steps both equal 5M and strict evaluation/checkpoint/hash validation passes.
6. Add aggregation tests that retain each run/seed and compute across-seed summaries from the ten per-seed evaluation means; never pool all episodes as if they were independent training seeds.
7. Run the full tests, six environment checks, dry-run matrix, and validator against deliberately incomplete/mismatched fixtures. No full training during these implementation steps.

### Stage 2: First approved full-length batch — seed 0 across six tasks

This is a proposed checkpoint batch, **not currently authorized**.

1. Immediately before launch, verify explicit user approval, resource preflight, no existing run targets, no active GPU benchmark process, and no tmux-session collision.
2. Start only the six seed-0 jobs (one environment at a time) under a named tmux session; each is exactly 5M steps. Do not start seeds 1–9 in this stage.
3. After each job, validate `actual_timesteps=5,000,000`, exactly 200 evaluation points `[25k, 50k, …, 5M]`, 20 finite episodes and positive lengths per point, expected environment/seed/config/source hashes, package versions, and loadable best/final `[2048,2048,1]` critic checkpoints.
4. On the first failure, stop the queue; preserve logs and partial files; do not resume or retry.
5. After all six validate, report actual wall time, peak RSS, CUDA use, disk usage, per-task curves, and a revised 60-run estimate. Pause for user review before any more training.

### Stage 3: Remaining seeds after separate approval

1. Obtain explicit approval for seeds `1–9` only after the seed-0 report and revised resource estimate are presented.
2. Re-run dry-run and resource/target checks. Confirm exactly 54 remaining jobs and no conflicting process/session.
3. Run serially, stopping on the first failure. Validate every job before starting the next. No retry, resume, or budget change without user consent.
4. Do not include the 100k pilot as a 5M seed result; the full matrix must start every 5M run from scratch at its declared seed.

### Stage 4: Analysis and report

1. Require 60/60 complete, matching, validated run artifacts for a complete matrix report. Otherwise report incomplete status and per-run failures without filling missing seed values.
2. Retain per-seed evaluation curves and final 20-episode means. Compute the paper's stated across-seed statistic/uncertainty after checking the figure caption; if not stated, report mean and standard deviation across ten independent seed-level means.
3. Report per-task/per-seed results, evaluation protocol, exact versions, wall time, resource use, checkpoint/hash audit, and deviations from the original JAX setup.
4. State prominently that this is a modern SB3-Contrib CrossQ-only reproduction. Do not claim replication of baseline comparison curves or exact historical equivalence.

## Failure and stop policy

- Fail closed on failed preflight, unexpected environment IDs, package/version drift, config/source hash mismatch, non-finite evaluation, missing checkpoint, wrong critic architecture, or a target/session collision.
- One run at a time. Do not add parallel workers to reduce wall time without a separate resource analysis and explicit approval.
- Incomplete runs are preserved and excluded from aggregates. Since replay/RNG state is not persisted, never call them resumable.
- A task/environment deprecation warning does not trigger automatic version substitution. Preserve the configured `-v4` ID and report the warning.
- The current user authorization covers only preparation of this plan. It does not authorize implementation of the full launcher, a seed-0 5M batch, or the 60-run matrix.

## Verification commands (after the planned tooling exists)

Run from `/home/erush/CMP620` using the existing Python environment:

```bash
PYTHONPATH=ARS/.venv/lib/python3.12/site-packages python -m pytest CrossQ/tests -q
CROSSQ_PYTHON=ARS/.venv/bin/python CrossQ/scripts/check_envs.py --config CrossQ/configs/paper_mujoco.yaml
CROSSQ_PYTHON=ARS/.venv/bin/python bash CrossQ/scripts/run_full_benchmark.sh --dry-run
```

Expected dry-run output: exactly 60 planned jobs, zero launch, zero run-target directories created. These checks do not authorize training.

## Source of measured pilot figures

- Pilot run: `CrossQ/artifacts/pilot/HumanoidStandup-seed0-attempt2/`
- Pilot preflight/run/time logs: `CrossQ/artifacts/logs/`
- Frozen matrix: `CrossQ/configs/paper_mujoco.yaml`
- Paper protocol/design: `CrossQ/docs/plans/2026-10-23-crossq-mujoco-design.md`
