# CrossQ Hopper Action-Noise Study — Execution Plan

> **For Codex:** This plan follows the latest project proposal in `CrossQ/docs/crossq.pdf`. Proposal-specific code, tests, dry-runs, five nominal training seeds, and the authorized action-noise evaluation are complete. Do not start further training. The ±20% threshold remains unapproved and was not applied.

**Goal:** Reproduce the Hopper-v4 CrossQ learning curve over five training seeds, then test the completed policies under controlled test-time action noise.

**Architecture:** Reuse the completed Hopper seed-0 5M run after checking its per-run settings against this proposal; record that its stored manifest was the earlier 3-task/3-seed subset. Train only seeds 1–4 with a separate proposal-specific runner so existing run-hashed source files and artifacts remain unchanged. The existing scaled validator hard-requires the old 3-task manifest, so add a proposal-specific validator that accepts the documented seed-0 manifest difference without weakening the old validator. After all five final policies validate, evaluate each policy at σ=0, 0.05, and 0.10 using paired reset seeds and common standard-normal noise streams.

**Tech Stack:** Python 3.12, SB3-Contrib/Stable-Baselines3 2.9.0, Gymnasium 1.4.0, MuJoCo 3.16.0, NumPy, pytest, existing read-only `ARS/.venv`.

---

## Proposal scope

- Reproduction: Hopper-v4, training seeds 0–4, 5,000,000 environment steps each; 25M total.
- Training settings from the proposal: UTD=1, two critics, actor `[256,256]`, critic `[2048,2048]`, BRN with 100k-step warm-up and SB3 running-stat update rate 0.01, policy delay 3, Adam β₁=0.5, batch size 256, discount 0.99.
- Evaluation during training: every 25k steps, 20 held-out episodes. Five evaluation environments are the modern SB3 implementation choice; disclose it as not specified by the paper.
- Extension, after training only: evaluate the five final checkpoints with deterministic actions plus Gaussian perturbations σ∈{0,0.05,0.10} in normalized action coordinates; clip to action bounds; use the same reset seeds 90000–90019 and matched standard-normal noise streams for σ=0.05 and σ=0.10. This produces 5×20×3=300 episodes.
- Primary metric: episodic return. Also report episode length, action-clipping frequency, and paired return differences from σ=0.
- Report per-seed results and 95% bootstrap intervals across training seeds as the proposal specifies. The paper's Figure 5 uses IQM with 70% quantile intervals; report that paper-style interval separately if the exact procedure is confirmed. Do not present the two interval types as equivalent.
- The proposed ±20% final-IQM tolerance is provisional (“subject to approval” in the proposal). Do not use it as a success/failure gate until the published endpoint is verified and the threshold is approved.
- This proposal tests robustness to action noise, not the paper's REDQ/DroQ UTD=20 comparison.

## Existing artifacts and reuse decision

| Artifact | Use in proposal | Caveat |
|---|---|---|
| `CrossQ/artifacts/scaled_reproduction_retry/Hopper-seed0/` | Reuse as training seed 0; 5M Hopper-v4 run and Table 1 mappings match the proposal's per-run settings. | Its saved config manifest lists the earlier three-task/seeds-0–2 subset. Preserve this manifest-scope difference in the report; do not rewrite its metadata. |
| `CrossQ/artifacts/scaled_reproduction_retry/HalfCheetah-seed0/` | Preserve, but exclude from this proposal's Hopper analysis. | It is a valid run for the superseded scaled plan, not a proposal task. |
| `CrossQ/artifacts/pilot/HumanoidStandup-seed0-attempt2/` | Preserve as a resource/stability pilot only. | Not part of the Hopper proposal. |
| `CrossQ/artifacts/scaled_reproduction/Hopper-seed0/` | Preserve as an incomplete historical attempt; exclude from all seed statistics. | It did not reach the 5M endpoint. |

The reusable Hopper seed-0 result had best mean return 4,018.13 at 4.95M and final mean 2,245.59 at 5M. This is one seed, not an across-seed estimate. The completed run took 21.96 hours; four remaining seeds are roughly 88 hours at that observed rate, with task/device variation possible.

The seed-0 metadata records hashes for the old `scaled_reproduction_jobs.py` and `run_scaled_reproduction.sh`; those files were later changed for single-run CLI support, and no byte-for-byte snapshot of their old contents was retained. The run passed strict validation at completion; current proposal validation confirms the core training sources and reports the two historical launcher-hash differences as warnings. If the course requires a currently reproducible byte-for-byte launcher snapshot, seed 0 must be rerun; otherwise retain it with this provenance limitation disclosed.

## Task 1: Proposal-specific config and runner for seeds 1–4 — no training

**Files:**
- Create `CrossQ/configs/proposal_hopper_5seeds_5m.yaml` from the frozen Table 1 mappings, narrowed to Hopper-v4 and seeds 0–4.
- Create `CrossQ/scripts/run_proposal_hopper.py`.
- Create `CrossQ/scripts/validate_proposal_hopper.py`.
- Create `CrossQ/tests/test_proposal_hopper.py`.

**Source-hash constraint:** Do not modify `CrossQ/scripts/scaled_reproduction_jobs.py` or `CrossQ/scripts/run_scaled_reproduction.sh`; their exact hashes are recorded in the completed seed-0 artifacts. The new runner must hash itself and the unchanged shared training sources.

**Steps:**
1. Add failing tests for the exact Hopper-only manifest, five seeds, 5M horizon, and unchanged per-run Table 1 settings.
2. Add failing tests that permit only fresh seeds 1–4, require explicit confirmation and a seed-specific tmux session, reject an existing target, and refuse package/resource/environment mismatches.
3. Test `--dry-run` lists exactly four jobs under a new `CrossQ/artifacts/proposal_hopper/` root without creating targets or starting training.
4. Implement the minimal runner using the existing `crossq_repro.runner.run`; enforce versions, RAM/disk/CUDA preflight, one-at-a-time execution, explicit confirmation, and seed-specific tmux sessions.
5. Run `PYTHONPATH=ARS/.venv/lib/python3.12/site-packages pytest -q CrossQ/tests` and the proposal runner dry-run. Expected: tests pass, four planned seed jobs, no training process and no run-target directories created.

## Task 2: Proposal-specific validation and summary — no training

**Files:**
- Modify `CrossQ/scripts/validate_proposal_hopper.py`.
- Extend `CrossQ/tests/test_proposal_hopper.py`.

**Steps:**
1. Add failing tests for the reused seed-0 artifact, proposal-config runs for seeds 1–4, wrong task/seed/steps, config/source mismatch, missing eval points, non-finite returns, and invalid checkpoints.
2. For seed 0, require the actual Hopper task, seed, 5M horizon, expected package versions, 25k/20 evaluation schedule, and matching resolved settings. Permit only the recorded top-level task/seed-manifest difference and retain its original config hash.
3. For seeds 1–4, require the proposal-specific manifest/config hash and exact task, seed, horizon, versions, evaluation, and source hashes.
4. Keep each seed's curve and final 20-episode score. Report per-seed results, IQM, and 95% bootstrap intervals resampled at the training-seed level. Never treat episodes as independent seeds.
5. Run the full tests and summary validation against the reused seed 0 artifact. No training.

## Task 3: Train seeds 1–4 — separate user approval required

1. Before launch, verify the four targets, exact package versions, tmux sessions, other training processes, RAM, disk, CUDA memory, and Hopper environment/API smoke.
2. After explicit approval, run Hopper seeds 1–4 sequentially, each for 5M steps, under seed-specific named tmux sessions.
3. Strictly validate each run before proceeding: metadata complete, requested/actual steps 5M, 200 evaluations ×20 finite episodes, config/source hashes, and loadable best/final checkpoints with the expected critic shape.
4. On any failure, stop, preserve artifacts, and do not automatically retry or resume. Ask before replacing an incomplete run.
5. Present the five-seed nominal results and pause before the action-noise evaluation.

## Task 4: Final action-noise evaluation — no training, after all five checkpoints validate

1. Create `CrossQ/scripts/evaluate_hopper_action_noise.py` and tests without modifying the trained models or replay state.
2. Load each final seed checkpoint; for each held-out reset seed 90000–90019, evaluate deterministic policy actions under σ=0, 0.05, 0.10. For nonzero σ, reuse the same standard-normal stream per training-seed/reset-seed pair and scale it by σ.
3. Record the pre-clip noisy action and compute action-clipping frequency; report episodic return and length for every seed/noise/reset-seed tuple.
4. Require exactly 300 completed episodes and no extra training. Report paired return differences against σ=0, per-seed scores, proposal 95% bootstrap intervals, and paper-style IQM/70% intervals separately where verified.
5. Compare σ=0 Hopper results to the published Figure 5 endpoint using a verified figure value or source data. Keep the ±20% tolerance pending approval.

## Current approval state

The user asked to follow this proposal and chose to reuse the completed Hopper seed-0 run. All five final policies (seed0 reused; seeds1–4 trained under the proposal config) pass strict validation. Final mean returns by seed are 2,245.59, 2,102.30, 2,049.57, 2,088.46, and 1,968.29; the seed-level IQM is 2,080.95 with a 95% bootstrap CI of [1,992.68, 2,202.60], resampled by training seed. The authorized 300-episode action-noise evaluation completed and passed coverage/finite-value validation; results are in `CrossQ/results/proposal_hopper/action_noise.json`. Mean paired return change at σ=0.05 is −1.91 (95% training-seed bootstrap CI [−87.95, 56.40]); at σ=0.10 it is −188.49 ([−345.96, −64.05]). Mean action-clipping fractions are 0.0645 and 0.0945 for σ=0.05 and0.10; corresponding mean episode lengths are 461.89 and425.61. The proposal's ±20% tolerance remains pending approval and was not applied. Seed3 attempts1 and2 stopped at75k and1.675M; both partial attempts are preserved under `CrossQ/artifacts/proposal_hopper_incomplete/` because replay/RNG state was not saved. Attempt2 coincided with a host/WSL reboot; exact reboot trigger is unknown. The full CrossQ suite passes 123 tests; dry-runs list the planned 300 evaluation episodes. The older 3-task ×3-seed plans are superseded. CMP620 is not a Git repository; do not initialize one or claim a commit.
