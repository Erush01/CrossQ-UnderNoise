# CrossQ Scaled Reproduction Design

## Goal

Obtain a bounded, paper-structured CrossQ result that preserves the paper's 5M-step horizon while reducing task and seed coverage to three tasks and three seeds.

## Scope

- **Algorithm:** stock SB3-Contrib CrossQ 2.9.0 only; no custom loss or replay modification.
- **Tasks:** Hopper-v4, HalfCheetah-v4, HumanoidStandup-v4.
- **Seeds:** 0, 1, 2 for each task (nine fresh training runs).
- **Budget:** 5,000,000 environment steps per task/seed.
- **Evaluation:** every 25,000 steps, 20 episodes across five eval environments. This is the selected modern SB3-Contrib/RL-Zoo protocol, not a paper-specified cadence.
- **Settings:** preserve the frozen CrossQ Table 1 configuration in `CrossQ/configs/paper_mujoco.yaml`; do not tune the selected tasks or seeds after seeing results.
- **Baselines:** excluded. This design cannot validate comparative claims against REDQ, DroQ, or SAC.

## Rationale and expected evidence

This option keeps the paper's 5M-step horizon and uses a small, diverse subset: a standard locomotion task (Hopper), HalfCheetah, and the task used by the completed 100k resource pilot (HumanoidStandup). Three seeds allow limited within-task variability reporting, but are fewer than the paper's ten and do not support the same statistical confidence. Results should be described as an exploratory scaled reproduction, not a full replication.

The completed 100k HumanoidStandup seed-0 pilot is **not** one of the nine runs and is not a checkpoint to resume from. Replay and RNG state were not saved. All nine runs start from scratch with their declared seeds; seed 0 is therefore a fresh 5M run.

## Budget and resource estimate

- Training budget: `3 tasks × 3 seeds × 5M = 45M` environment steps.
- Evaluation workload: `200` eval points per run, `1,800` total; `20` episodes per point, `36,000` episodes total.
- Pilot-rate estimate: `1,547.13s` per 100k steps implies about `21.49h` per 5M run and `193.4 GPU-hours` (about eight days) for nine serial runs.
- This is a rough planning estimate from one HumanoidStandup pilot, not a validated task-by-task forecast. Recalculate after each completed task/seed.
- Two model checkpoints at observed sizes imply about `2.21 GB` for nine runs, excluding evaluation data, logs, metadata, and incomplete outputs.

## Analysis and limitations

- Retain raw per-seed evaluation curves and report individual seed results plus across-seed summaries for each task.
- With only three seeds, label uncertainty estimates exploratory. Do not imply equivalence to the paper's ten-seed statistics.
- Compare the three selected tasks to paper curves at the same 5M endpoint, but report the modern evaluation choice and SB3-Contrib/JAX differences.
- A CrossQ-only run cannot validate the paper's relative sample-efficiency/compute claims against REDQ or DroQ. That would require separate, source-verified baseline implementations and matched runs.

## Approval boundary

The user approved this design and an execution plan only. Implementation of a launcher and all nine training runs remain unstarted and require separate explicit authorization. Run serially, stop at the first invalid/incomplete artifact, and never resume/retry automatically because replay/RNG state is not checkpointed.
