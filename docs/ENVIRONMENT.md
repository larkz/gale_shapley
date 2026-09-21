# Environment Construction — LLM-Task Stable Matching

This document specifies exactly how the learning environment is built:
from raw LLM benchmark data to a two-sided matching market with noisy
pairwise feedback, for every configuration used in the experiments.

All construction is deterministic given (config, PYTHONHASHSEED=0).
Entry point: `llm_matching.runner.build_context(config)` (offline
studies) and `llm_matching.bandit.build_bandit_context(config)`
(online bandit studies).

---

## 1. Source data: LLMRouterBench (fully offline)

* Pre-collected evaluation results (HuggingFace `NPULH/LLMRouterBench`,
  `bench-release.tar.gz`, ~1.28 GB), fetched once with
  `scripts/fetch_routerbench_results.py`.
* One JSON per (dataset, split, model): per-instance records
  `{index, score, cost, prompt_tokens, completion_tokens}`.
* Full pool: **20 models x 15 datasets** (`data/market_analysis_20x15/`).
* No LLM inference anywhere in the loop — the environment only reads
  stored scores. Every pairwise "comparison" the learner requests is a
  Bernoulli draw computed from these scores.

## 2. Alignment (`llm_matching/routerbench_loader.py`)

1. Load every (dataset, split, model) JSON into score records.
2. Per dataset: intersect record indices across the SELECTED models
   (any missing entry is a hard error; per-dataset alignment stats are
   written to `data_alignment.csv`).
3. Quality gate: `min_aligned_records: 100`.
4. All markets used in the experiments have **zero missing entries**
   after alignment.

## 3. Splits (offline studies only)

Deterministic 60/20/20 train/val/test per dataset — index-arithmetic
hashing with `split.seed` (default 3407), NO random number generator,
so the same record lands in the same split for every model.
The ONLINE bandit studies (`bandit`, upstream replication) use NO
split: utilities are computed on the FULL aligned data.

## 4. Utilities (the market's "true" values)

For any evaluation set S (train split, or full data):

* **Task side** (task = woman, dataset d): mean score of model m on d
  over S:  `U_d(m) = mean_{x in S} score[m, d, x]`.
* **Model side** (model = man), comparative-advantage mode:
  `V_m(d) = U_d(m) - (1/(M-1)) * sum_{m' != m} U_d(m')`
  — "on which task am I strong RELATIVE to rivals", which avoids the
  degenerate "everyone prefers the easy task" market.
* Symmetric/mirror mode (`preferences.mode: symmetric`):
  `W(d, m) = U_d(m)` shared by BOTH sides (model side = W.T) — see §5.

## 5. Preferences and the oracle

Two preference modes:

* **comparative** (default): task d ranks models by `U_d`; model m
  ranks tasks by `V_m`.
* **symmetric (mirror)**: both sides rank by the SAME matrix
  `W = U` (model side reads W.T). With a strictly ordered W the stable
  matching is provably UNIQUE (global-max pair is a mutual top choice;
  induct on the remaining submatrix). `build_context` verifies this at
  construction: men-GS == women-GS == mutual-best cascade, else
  AssertionError.

**Tie handling.** Real utilities contain EXACT ties (e.g. the original
market has math500: DS-R1 == Nemotron == 0.9466667). Numerically
indistinguishable values (within `ties.epsilon = 1e-6`, transitive
grouping) receive a deterministic SHA256 jitter (<= eps/4 in
comparative mode; <= eps/8 per cell in symmetric mode, which breaks
row AND column ties jointly). Preferences, oracle, and reward all use
the SAME jittered values. No Python `hash()` is used.

**Oracle** `H*_S` = men-proposing Gale-Shapley (models propose) on the
strictified preferences. Baselines: Hungarian on utilities (welfare
optimum), random bijection (200 seeds), and the test-side welfare
ceiling where splits exist. Oracle JSONs ship in each
`data/<study>/oracle_matching_*.json`.

## 6. Reward: pairwise feedback providers (`llm_matching/providers.py`)

The learner never sees utilities — only binary answers to "does agent
a prefer b1 over b2?" via the `SignalProvider` interface.

### 6.1 Bradley-Terry (parametric; default for all headline studies)

```
theta_side(x) = exp(eta_side * utility_side(x))
P(b1 beats b2 | agent a) = theta_a(b1) / (theta_a(b1) + theta_a(b2))
```

`eta_task`, `eta_model` are calibrated so the **median positive
utility gap** maps to win probability `target_median_win_probability`
(default 0.70; calibration constants ship in `bt_calibration.json`).
Calibration: `eta = ln(p/(1-p)) / median_gap`.

### 6.2 No-tie guarantee (`feedback.probability_floor`, optional)

Every win probability is clamped to `|p - 1/2| >= floor` **in the
direction of the (jittered) strict preference**:

* exact ties (previously p = 0.5000002, structurally unresolvable)
  become p = 0.5 ± floor;
* weak-but-real signals below the floor are boosted to the floor;
* strong signals (|p - 1/2| >= floor) are untouched.

This is a REWARD-CHANNEL-ONLY change: preferences, the oracle, and
welfare are unchanged. Resolution guarantee: every arm resolves at
`n ~ c*ln(t)/floor^2` samples (c = CI constant), which is what makes
certified stopping feasible at 8x8 within a 200K horizon.

### 6.3 Replay (empirical robustness mode)

* Task side: sample a TRAIN instance x of d; the model with the higher
  stored score on x wins (exact score tie -> fair coin).
* Model side: sample instances x1 ~ d1, x2 ~ d2; per-instance
  comparative advantages a_i; `P(d1 wins) = clip(0.5 + gamma*(a1-a2),
  eps, 1-eps)` with gamma = 0.25.
* Substantially noisier than BT (resolution ~0.40 vs ~0.85 at 200K on
  the original 8x8) — used to check that BT-specific conclusions
  survive a more realistic feedback channel.

## 7. Learners (what differs between algorithms)

All learners share the estimator (per-agent BT-MLE ranking + Hoeffding
CIs `sqrt(c ln t / n)`, c = 0.1). They differ ONLY in sampling and
stopping:

| | P2ETG (`p2etg.py`) | PrefLID (`preflid.py`) | Matching-ID (`llm_matching/matching_id.py`) |
|---|---|---|---|
| sampling | uniform round-robin over all arms | RRT: one center agent per iteration round-robins its partners (`center_policy: round_robin` recommended; upstream default `random` has a multinomial straggler problem) | identical to P2ETG |
| stopping | ALL pairwise CIs exclude 1/2 (full-preference identification) | all CI-consistent configurations merge into ONE lattice island (structural certificate; |Omega| <= budget) | all CI-consistent profiles yield the SAME GS matching (exact certificate) |

## 8. Markets used in the experiments

| market | mode | datasets x models | selection | key facts |
|---|---|---|---|---|
| original 8x8 | comparative | math500, livecodebench, bbh, mmlupro, finqa, medqa, meld, korbench x DS-R1, Fin-R1, Llama-UltraMedical, Qwen2.5-Coder, Qwen3-8B, Llama-Inst, Nemotron, gemma | hand | P_bootstrap=0.114; 7 critical arms; 2 EXACT ties (structural blockers) |
| smoke 4x4 | comparative | math500, livecodebench, finqa, medqa x 4 models | subset | the certification lab |
| diverse korbench 8x8 | comparative | emorynlp, finqa, humaneval, korbench, math500, mbpp, medqa, meld x DS-R1, Fin-R1, GLM-Z1, Intern-S1, Qwen2.5-Coder, Qwen3-8B, gemma, glm-4 | `scripts/select_diverse_market.py` (distinct-winner greedy, variant preflight) | P_bootstrap=0.538; 0 ambiguous tasks; min critical gap 0.010 |
| mirror markets | symmetric | same as above (and original) | — | unique stable matching (verified) |
| scaling 3x3/5x5/10x10 | symmetric | greedy distinct-winner prefixes of the full pool | `configs/bandit_scaling_*.yaml` | floor 0.1 + round-robin |

Selection preflight (500 bootstrap replicates + 448-arm sensitivity)
lives in each study's `bootstrap/` and `sensitivity/` outputs; the
kandk-variant rejection note in
`data/diverse_korbench_8x8/market_selection_note.md`.

## 9. Reproducing an environment from scratch

```bash
pip install -r requirements-llm-matching.txt
python scripts/fetch_routerbench_results.py          # once, ~1.3GB

# offline study (train/test split, oracle generalization)
PYTHONHASHSEED=0 python -m llm_matching.cli \
    --config configs/llm_matching_8x8.yaml --algorithm p2etg --feedback bt --seed 0

# online matching bandit (no split, mirror preferences, floor)
PYTHONHASHSEED=0 python -m llm_matching.cli bandit \
    --config configs/llm_matching_8x8_diverse_korbench_symmetric_floor.yaml

# diagnostics: bootstrap / sensitivity / market analysis
PYTHONHASHSEED=0 python -m llm_matching.cli bootstrap \
    --config configs/llm_matching_8x8_diverse_korbench.yaml --num-bootstrap 500
```

`PYTHONHASHSEED=0` is REQUIRED: `gs_lib.bt._canonical` orders pairs by
Python `hash()`, which is otherwise randomised per process and makes
runs irreproducible.

## 10. Config reference (all knobs)

```yaml
routerbench_root: ../LLMRouterBench
datasets: [...]            # tasks (women)
models: [...]              # models (men)
min_aligned_records: 100
split: {train: .6, val: .2, test: .2, seed: 3407}
preferences: {mode: comparative | symmetric}   # symmetric => unique H*
feedback:
  mode: bt | replay
  target_median_win_probability: 0.70   # BT sharpness (eta calibration)
  probability_floor: 0.0                # no-tie reward guarantee (>=0)
  replay_gamma: 0.25
  replay_probability_epsilon: 0.01
  model_replay_mode: probabilistic
ties: {policy: deterministic_jitter, epsilon: 1.0e-6}
p2etg:  {adaptive: true, check_every: 448, max_samples: 200000, constant: 0.1}
preflid: {budget: 100, constant: 0.1, max_lattice_vertices: 5000,
          min_samples_per_pair: 10, min_sample_ratio: 0.5,
          center_policy: random | round_robin, max_iterations: ...}
matching_id: {max_profiles: 200000, timeout_seconds: 2.0, certify_every: 1}
experiment: {seeds: 20, random_baseline_seeds: 200}
output_dir: outputs/<run>
```
