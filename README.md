# CrossQ MuJoCo reproduction

This workspace isolates the CrossQ code, configuration, and run artifacts for
a modern reproduction of the CrossQ MuJoCo experiments. It uses the PyTorch implementation in SB3-Contrib rather than the
authors' official JAX repository, which remains an alternative implementation
but is not selected here. The separate `requirements-lock.txt` records the
exact installed package snapshot used as the dependency provenance source; it
follows the `ARS/requirements-lock.txt` `pip freeze` convention.

The validated runtime is the existing read-only `ARS/.venv`: Python 3.12.3 with Gymnasium 1.4.0, MuJoCo 3.16.0,
Stable-Baselines3 2.9.0, SB3-Contrib 2.9.0, and PyTorch 2.14.1. These are
current, pinned reproduction dependencies, not versions reported by the paper.
Environment APIs and results may therefore differ from the historical setup;
do not interpret this as an exact reproduction of its original software stack.

The method is based on the CrossQ ICLR 2024 paper:

- Paper: https://proceedings.iclr.cc/paper_files/paper/2024/file/f381114cf5aba4e45552869863deaaa7-Paper-Conference.pdf
- Authors' official JAX implementation (not selected): https://github.com/adityab/CrossQ
- Selected PyTorch implementation: [SB3-Contrib CrossQ](https://sb3-contrib.readthedocs.io/en/v2.9.0/modules/crossq.html)

## Frozen MuJoCo experiment manifest

`configs/paper_mujoco.yaml` is the frozen experiment manifest. It records the
six Gymnasium-v4 tasks, intended seeds 0–9, a 5,000,000-step budget, and the
SB3-Contrib 2.9.0 / SB3 2.9.0 / Gymnasium 1.4.0 / MuJoCo 3.16.0 implementation
versions. The paper settings are transcribed from Appendix A.2, Table 1 of the
[official ICLR 2024 paper](https://proceedings.iclr.cc/paper_files/paper/2024/file/f381114cf5aba4e45552869863deaaa7-Paper-Conference.pdf).
The manifest records provenance per setting. In particular, the paper's BRN
running-average momentum is stored as 0.99; the separate SB3 PyTorch
`batch_norm_momentum: 0.01` mapping uses the complementary update convention.
UTD 1 is represented by `train_freq: 1` and `gradient_steps: 1`.

`settings.sb3_mappings` records the realizable SB3-Contrib v2.9.0 API: one
`CrossQ` constructor `learning_rate: 0.001` for both actor and critic; the
constructor's `policy_delay: 3`; and `policy_kwargs` for `net_arch` (`pi` and
`qf`), ReLU, batch normalization, BRN warm-up, two critics, and Adam betas.
These names and defaults were checked against the installed v2.9.0 source
(`sb3_contrib.crossq.crossq.CrossQ.__init__` and
`sb3_contrib.crossq.policies.CrossQPolicy.__init__`). CrossQ does not expose
`tau` as a caller kwarg: its constructor passes `tau=1.0` internally, which
disables target-network updates. `ent_coef: auto`, `learning_starts: 100`, and
the SB3 default actor width are explicitly marked non-paper choices.

The 5,000,000-step endpoint is sourced to Figures 5–6 and Appendix A.3. Task
IDs are sourced individually to Appendix A.3, Table 2; Humanoid-v4 is identified
as the Gymnasium task-ID counterpart and cross-referenced to the Section 4
environment list.

The `[256, 256]` actor architecture is the SB3-Contrib 2.9.0 documented default,
not a value reported in Table 1. Likewise, automatic entropy coefficient and
`learning_starts: 100` are modern SB3 defaults, not paper settings. Evaluation
uses the versioned SB3-Contrib/RL-Zoo modern protocol (25,000-step frequency,
20 episodes, 5 evaluation environments); the paper does not specify these
evaluation details, so they must not be described as paper-exact.
Each run seeds its evaluation vector environment with the training seed plus
10,000 as an explicit modern reproducibility choice; this offset is not
specified by the paper and is recorded in run metadata.
Each raw evaluation environment is wrapped with SB3 `Monitor` before entering
`DummyVecEnv`, making episode reward and length reporting explicit for
`EvalCallback`. Training environments are unchanged. This addresses SB3's
advisory warning that unmonitored evaluation environments can report modified
episode values when wrappers intervene.

The manifest uses JSON syntax, a dependency-free loader for the JSON-compatible
YAML subset. Caller-supplied config dictionaries are serialized canonically to
each run's `config.json`; the stored snapshot is the source of its recorded hash.
This keeps loading and schema validation dependency-free because the existing
ARS virtual environment has no PyYAML parser; no package was installed for this
task. `requirements-lock.txt` is a package snapshot of that runtime, not a
separately provisioned CrossQ virtual environment.

CrossQ source/configuration and run artifacts remain separate from ARS/HER.
The shared runtime is used without package changes: do not install or upgrade
dependencies in `ARS/.venv` for this reproduction. The selected device is
recorded both as requested (`auto` by default) and as resolved by PyTorch.
