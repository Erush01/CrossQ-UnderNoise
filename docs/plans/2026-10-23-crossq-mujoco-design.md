# CrossQ MuJoCo Reproduction Design

## Goal and scope

Reproduce the CrossQ ICLR 2024 continuous-control results using SB3-Contrib's modern PyTorch implementation on the paper's six MuJoCo tasks: Hopper, Walker2d, Ant, Humanoid, HalfCheetah, and HumanoidStandup. This is a modern, version-pinned reproduction, not an exact historical rerun: the paper does not specify its Gymnasium/MuJoCo package versions, and SB3-Contrib differs in implementation details.

The paper's main MuJoCo results use ten random seeds per environment. The implementation path will preserve that target, but no full multi-seed benchmark is launched until the user approves it after the smoke tests and pilot.

## Implementation choice

Use the installed SB3-Contrib 2.9.0 CrossQ implementation in `ARS/.venv` as the starting point, but create an isolated `CrossQ/` experiment workspace and its own scripts/artifact tree; do not change the existing ARS environment. Match paper Table 1: gamma 0.99, learning rate 0.001, replay 1e6, batch 256, ReLU, BRN with paper moving-average momentum 0.99 and 100k warm-up steps, critic width `[2048, 2048]`, Adam beta1 0.5, UTD 1, policy delay 3, two critics, and no target networks. SB3-Contrib encodes the paper's BRN moving-average momentum as PyTorch update momentum 0.01 and no target networks as `tau=1.0`; verify these mappings against v2.9.0 source. SB3-Contrib's actor default is `[256, 256]`; actor width is not listed in Table 1 and must be recorded as an implementation default, not a paper-matched value. Keep stock CrossQ; do not add custom algorithm changes.

The official JAX implementation is not selected because the existing modern PyTorch implementation lowers setup risk. The reproduction report must state this choice and document differences from both the original JAX code and SB3-Contrib defaults. The paper reports a 5M-step endpoint and ten seeds per task, but does not specify evaluation frequency/episode count in the main text. Use the documented SB3-Contrib/RL-Zoo recipe (evaluate every 25k steps, 20 episodes, 5 eval environments) only as an explicitly modern evaluation protocol, not as an original-paper setting.

## Experiment stages

1. **Protocol and environment validation:** Extract and record paper hyperparameters and evaluation protocol from the official paper and SB3-Contrib v2.9.0 implementation/docs. Pin package versions and explicitly name Gymnasium environment IDs. Confirm each environment can reset/step and that CrossQ accepts it.
2. **Smoke tests:** Run short, bounded training checks on all six tasks; verify finite learning, expected critic width, saved configuration/checkpoint, and evaluation execution. These are engineering checks, not reproduction results.
3. **Pilot:** Run a single-seed bounded pilot under the frozen protocol. Inspect runtime, memory, numerical stability, and evaluation artifacts before requesting authorization for the full run.
4. **Full benchmark after approval:** Run ten seeds per environment under the frozen settings, with periodic evaluation and per-seed artifacts. Do not silently change hyperparameters or environment versions mid-run.
5. **Analysis:** Report per-environment/per-seed scores, aggregate uncertainty, package versions, deviations, and comparisons to the paper. Label results as modern-version reproduction evidence, not exact historical replication.

## Evaluation and evidence

Before implementation, verify exact paper settings and evaluation details from the official ICLR PDF (especially Table 1 and the MuJoCo experimental section) and the SB3-Contrib 2.9.0 CrossQ docs. The run configuration must record environment ID/version, seed, policy/network architecture, all algorithm settings, evaluation seeds/frequency/episodes, and dependency versions. Smoke/pilot validation must check process completion, finite metrics, configuration consistency, and loadable checkpoints. Full-run summaries must retain individual seed results rather than only a pooled score.

## Risks and caveats

- The paper does not identify the exact Gymnasium/MuJoCo versions; environment version differences can affect scores.
- SB3-Contrib's CrossQ is a modern PyTorch port, not the authors' JAX implementation.
- Matching critic width does not eliminate other implementation differences.
- Six tasks × ten seeds may be computationally expensive; runtime is measured on the pilot before approval to scale.
- The CMP620 workspace root is not a Git repository, so this design cannot be committed there; keep the design and experiment artifacts in the isolated `CrossQ/` directory.

## Authoritative sources

- CrossQ ICLR 2024 paper: https://proceedings.iclr.cc/paper_files/paper/2024/file/f381114cf5aba4e45552869863deaaa7-Paper-Conference.pdf
- ICLR paper record: https://openreview.net/forum?id=PczQtTsTIX
- Authors' code: https://github.com/adityab/CrossQ
- SB3-Contrib v2.9.0 CrossQ documentation: https://sb3-contrib.readthedocs.io/en/v2.9.0/modules/crossq.html
- SB3-Contrib v2.9.0 installation documentation: https://sb3-contrib.readthedocs.io/en/v2.9.0/guide/install.html
