# CrossQ MuJoCo Reproduction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a reproducible SB3-Contrib CrossQ harness for the six MuJoCo tasks in the CrossQ ICLR 2024 paper, validate it with bounded smoke tests and a single-seed pilot, then request authorization before the ten-seed benchmark.

**Architecture:** Keep all CrossQ code, configuration, tests, and run artifacts under `CrossQ/`, separate from the existing ARS and HER work. Use SB3-Contrib 2.9.0 CrossQ with the paper's Table 1 settings (gamma 0.99, LR 0.001, replay 1e6, batch 256, ReLU, BRN, critic `[2048, 2048]`, Adam beta1 0.5, UTD 1, policy delay 3, two critics, no target networks); keep the SB3 actor default `[256,256]` explicitly labeled because Table 1 does not report actor width. A configuration-driven runner records exact versions and experiment settings, while separate evaluation and summary scripts consume saved checkpoints/logs. Do not alter CrossQ internals or automatically launch the full benchmark.

**Tech Stack:** Python 3.12, Stable-Baselines3 2.9.0, SB3-Contrib 2.9.0, Gymnasium, MuJoCo, pytest, NumPy, pandas.

---

## Research and protocol gates

- Paper values verified from official PDF Appendix A.2, Table 1: gamma .99, actor/critic LR .001, replay 1e6, batch 256, ReLU, BRN, paper moving-average momentum .99, warm-up 1e5, critic width 2048, Adam beta1 .5, UTD 1, policy delay 3, two critics, no target update rate. SB3 encodes paper BRN moving-average momentum .99 as PyTorch update momentum .01 and no target networks as tau=1.0; verify this mapping against installed v2.9.0 source. Record citations in README.
- Paper Appendix A.2 names the six-task protocol and 10 seeds; Sections 4.1–4.2 use periodic evaluations, 5M endpoint, and mean/std over seeds. Evaluation frequency and episode count are not specified by the paper; any choice (e.g. SB3-Contrib's documented 25k / 20 episode / 5 eval-env recipe) must be labeled modern protocol, not paper-exact.
- Confirm exact supported environment IDs and their installed Gymnasium/MuJoCo versions. The paper does not report those package versions, so use and disclose the selected current versions rather than claiming exact historical equivalence.
- Confirm `CrossQPolicy` architecture parameters and training/evaluation APIs from the installed SB3-Contrib 2.9.0 source/docs before writing the runner.
- Confirm the SB3 default actor architecture and whether paper settings require explicit overrides. Do not assume undocumented defaults match the paper.

## Task 1: Isolated workspace and dependency provenance

**Files:**
- Create: `CrossQ/README.md`
- Create: `CrossQ/requirements-lock.txt` (or another exact lock format chosen after checking the existing repository convention)
- Create: `CrossQ/.gitignore`

**Steps:**
1. Inspect `ARS/README.md`, `ARS/requirements-lock.txt`, and available lock tooling for project conventions and exact installed versions.
2. Write a failing workspace validation test asserting the declared CrossQ package versions and all six environment IDs are present in the experiment manifest.
3. Run the test and verify it fails because the workspace does not yet exist.
4. Create the isolated workspace, record the SB3 2.9.0/SB3-Contrib 2.9.0/Gymnasium/MuJoCo versions, and document that CrossQ was selected over the authors' JAX code for setup risk.
5. Run the validation test; expected: PASS. Do not install or upgrade packages in `ARS/.venv`.

## Task 2: Frozen experiment manifest and paper settings

**Files:**
- Create: `CrossQ/configs/paper_mujoco.yaml`
- Create: `CrossQ/tests/test_config.py`
- Modify: `CrossQ/README.md`

**Steps:**
1. Write tests requiring six paper task names, explicit Gymnasium environment IDs, ten planned seeds, explicit evaluation cadence/episode count, `[2048, 2048]` critic width, and pinned implementation versions.
2. Run `pytest CrossQ/tests/test_config.py -q`; expected: FAIL because manifest/parser is absent.
3. Transcribe verified paper settings from Appendix A.2 Table 1; annotate each value with its source. Keep actor width and evaluation frequency/episode count clearly marked as unspecified by the paper; choose a documented modern protocol for those.
4. Implement a minimal YAML loader/validator that rejects missing or unknown required fields and disallows an empty seed list.
5. Run the config tests; expected: PASS. Review all values against the primary PDF and versioned CrossQ docs.

## Task 3: Environment and CrossQ API smoke validation

**Files:**
- Create: `CrossQ/scripts/check_envs.py`
- Create: `CrossQ/tests/test_env_manifest.py`

**Steps:**
1. Write tests for the six manifest entries and the expected continuous `Box` action space requirement.
2. Run the tests; expected: FAIL until the checker exists.
3. Implement a short check that creates each environment, resets with a fixed seed, takes one valid action, verifies the step result and finite observations/rewards, and closes the environment.
4. Add a CrossQ construction check that asserts the effective critic width is `[2048, 2048]`; do not train yet.
5. Run the tests and then `python CrossQ/scripts/check_envs.py --config CrossQ/configs/paper_mujoco.yaml`; expected: six environments pass or a precise dependency/environment blocker is reported. No automatic version substitutions.

## Task 4: Training and evaluation harness

**Files:**
- Create: `CrossQ/crossq_repro/runner.py`
- Create: `CrossQ/crossq_repro/evaluate.py`
- Create: `CrossQ/crossq_repro/artifacts.py`
- Create: `CrossQ/tests/test_runner_config.py`
- Create: `CrossQ/tests/test_artifacts.py`

**Steps:**
1. Write unit tests using a tiny test environment to verify model kwargs, fixed-seed handling, artifact naming, metadata persistence, and evaluation result schema.
2. Run the tests; expected: FAIL because runner/artifact functions are missing.
3. Implement the minimum runner that loads the frozen manifest, creates CrossQ, learns a caller-specified number of steps, evaluates at configured intervals, and saves checkpoints and JSON metadata under a unique run directory.
4. Persist resolved model settings, seed, package versions, environment ID, timestamps, and source/config hashes with each run; refuse to overwrite an existing run directory.
5. Implement evaluation that loads a checkpoint and reports raw episode returns/lengths and aggregate mean/std without collapsing seed-level data.
6. Run tests; expected: PASS. Verify no observer, replay-buffer modification, or custom algorithm loss has been added.

## Task 5: Bounded smoke tests across six tasks

**Files:**
- Create: `CrossQ/scripts/run_smokes.sh`
- Create: `CrossQ/tests/test_smoke_artifacts.py`

**Steps:**
1. Write artifact checks for complete status, finite evaluation values, loadable model ZIP, correct critic width, and matching config hash.
2. Run the artifact test against an empty fixture; expected: FAIL.
3. Add a bounded smoke launcher for all six environments with a short, explicit step budget and one fixed seed; name results as engineering smoke tests.
4. Run smoke jobs sequentially to avoid resource contention; inspect completion markers, model loading, and evaluation logs.
5. Run artifact tests; expected: PASS for all six, or stop at the first failure and diagnose before continuing. Do not increase step budget automatically.

## Task 6: Single-seed pilot and approval gate

**Files:**
- Create: `CrossQ/scripts/run_pilot.sh`
- Create: `CrossQ/scripts/summarize_pilot.py`
- Create: `CrossQ/docs/pilot-protocol.md`

**Steps:**
1. Define a bounded pilot protocol and acceptance limits for wall time, memory, finite training/evaluation metrics, and complete artifacts; use one task/seed chosen to be representative and disclose the choice.
2. Run a preflight that checks disk/RAM availability and confirms the target output directory is absent.
3. Run the single-seed pilot only; keep the process and output isolated from ARS/HER runs.
4. Summarize runtime, resource use, evaluation trajectory, checkpoint validity, and deviations. Confirm no full ten-seed jobs were launched.
5. Present the pilot evidence and ask the user before creating or launching the full ten-seed matrix.

## Verification commands

Run from `/home/erush/CMP620`:

```bash
python -m pytest CrossQ/tests -q
python CrossQ/scripts/check_envs.py --config CrossQ/configs/paper_mujoco.yaml
bash CrossQ/scripts/run_smokes.sh --dry-run
python CrossQ/scripts/summarize_pilot.py --validate-only
```

Expected before any long run: unit tests pass, all six environment checks pass, smoke dry-run lists only bounded jobs, and pilot summary validation rejects incomplete/mismatched artifacts. Full ten-seed benchmark remains disabled until explicit user approval.

## Source references

- Official ICLR paper PDF: https://proceedings.iclr.cc/paper_files/paper/2024/file/f381114cf5aba4e45552869863deaaa7-Paper-Conference.pdf
- Authors' repository: https://github.com/adityab/CrossQ
- SB3-Contrib 2.9.0 CrossQ docs: https://sb3-contrib.readthedocs.io/en/v2.9.0/modules/crossq.html
- SB3-Contrib 2.9.0 installation docs: https://sb3-contrib.readthedocs.io/en/v2.9.0/guide/install.html
