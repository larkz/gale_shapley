# LLM-Task Stable Matching (Offline, LLMRouterBench)

One-to-one stable matching between **LLM models** (men) and **task /
dataset types** (women), learned from **pairwise capability comparisons
only** — the learner never observes the full preference matrices.

Main scientific question:

> Can the correct stable LLM-task assignment be identified using
> substantially fewer pairwise capability comparisons than are required
> to recover the complete two-sided preference profile?

**No LLM inference is performed anywhere.** All model performance comes
from precomputed [LLMRouterBench](https://github.com/ynulihao/LLMRouterBench)
result JSON files (`score` fields).

---

## 1. Research problem

Two-sided stable matching (Gale-Shapley) where:

- **Men = LLM models** (8 lightweight models: general and specialized);
- **Women = task types** (8 datasets spanning math, code, logic, general
  knowledge, finance, medical, affective understanding, reasoning).

Both sides have latent preferences derived from real benchmark scores.
An online learner (P2ETG, and optionally PrefLID) queries a
`SignalProvider` for noisy pairwise feedback
("does agent *a* prefer *b1* over *b2*?") and must stop and commit to a
matching. The hidden oracle matching H* is the model-proposing
Gale-Shapley outcome under the true TRAIN preferences.

## 2. Why models and tasks form the two sides

Each side has a genuine preference over the other:

- Tasks prefer models that solve them better (accuracy).
- Models "prefer" tasks on which they have **comparative advantage**
  relative to the other models — not tasks that are merely easy.

The one-to-one capacity constraint (each model serves exactly one task
type) turns routing into a matching problem rather than per-query
top-1 selection.

## 3. Task-side preference

For task (dataset) `d`, model `m`, and TRAIN instances `x`:

```
U_d(m) = mean_{x in D_d^train} s_{d,x,m}
```

Task `d` prefers `m_i` over `m_j` iff `U_d(m_i) > U_d(m_j)`.

## 4. Model-side preference (comparative advantage)

For model `m` and task `d`:

```
V_m(d) = U_d(m) - (1/(M-1)) * sum_{m' != m} U_d(m')
```

Model `m` prefers tasks where it beats the *other available models* by
the widest margin. This avoids the degenerate "everyone prefers the
easy task" behaviour of raw accuracy.

Exact/numerical utility ties are broken by a deterministic SHA256-based
jitter of size at most `tie_epsilon` (config `ties:`), applied
consistently to oracle rankings, BT probabilities, and replay feedback.

## 5. Oracle stable matching

`H*_train` = model-proposing Gale-Shapley under the strict TRAIN
utilities (task side: `U_d`, model side: `V_m`). The learner never sees
these rankings; they exist only for evaluation. `H*_val` / `H*_test`
are computed for the generalization analysis.

## 6. BT feedback mode (`--feedback bt`)

Bradley-Terry feedback matching P2ETG's statistical assumptions:

```
theta_{d,m} = exp(eta_T * U_d(m))     (task side)
theta_{m,d} = exp(eta_M * V_m(d))     (model side)

P(b1 beats b2 | agent) = theta[b1] / (theta[b1] + theta[b2])
```

`eta_T`, `eta_M` are **calibrated from the data**: the median positive
utility gap induces the configured win probability (default 0.70):

```
eta = log(p / (1 - p)) / median_gap
```

This mode is a controlled experiment derived from real utilities.

## 7. Empirical replay mode (`--feedback replay`)

Robustness mode without the BT assumption, using stored per-instance
outcomes:

- **Task side** (`task d`, models `m1` vs `m2`): sample a TRAIN instance
  `x` of `d`; the model with the higher stored score on `x` wins; exact
  ties flip a seeded fair coin.
- **Model side** (`model m`, tasks `d1` vs `d2`): sample TRAIN instances
  `x1 ~ D_{d1}`, `x2 ~ D_{d2}`; compute per-instance comparative
  advantages `a_i = s_{m,d_i,x_i} - mean_{m' != m} s_{m',d_i,x_i}`; then
  - probabilistic (default): `P(d1 wins) = clip(0.5 + gamma*(a1-a2), eps, 1-eps)`
  - direct: `d1` wins iff `a1 > a2` (ties by coin).

## 8. Data download

The gale_shapley repo expects a sibling LLMRouterBench checkout with
benchmark results:

```
workspace/
├── gale_shapley/          <- run experiments here
└── LLMRouterBench/
    └── results/bench/     <- extracted results
```

Download the official pre-collected results (1.28 GB) from
<https://huggingface.co/datasets/NPULH/LLMRouterBench>
(`bench-release.tar.gz`), extract into `LLMRouterBench/results/`, and
rename `bench-release/` to `bench/`. Or run:

```bash
python scripts/fetch_routerbench_results.py --routerbench-root ../LLMRouterBench
```

If the data are missing, the loader fails with a clear diagnostic.

## 9. Smoke test (4 models x 4 tasks)

```bash
python -m llm_matching.cli \
    --config configs/llm_matching_smoke.yaml \
    --algorithm p2etg \
    --feedback bt
```

## 10. Main 8x8 experiment

Single seed first:

```bash
python -m llm_matching.cli \
    --config configs/llm_matching_8x8.yaml \
    --algorithm p2etg \
    --feedback bt \
    --seed 0
```

Then all 30 seeds (remove `--seed`). For empirical replay, use
`--feedback replay` and a separate `output_dir` (configs can be copied
with a new `output_dir`, e.g. `outputs/llm_matching_8x8_replay`, so the
BT run is not overwritten). `--algorithm preflid` runs PrefLID (only
recommended at 4x4 — lattice enumeration is expensive at 8x8).

CLI arguments override YAML values. Run
`pytest tests/llm_matching -q` for the test suite.

## Phase-2 diagnostic commands

```bash
python -m llm_matching.cli bootstrap --config configs/llm_matching_8x8.yaml --num-bootstrap 500
python -m llm_matching.cli sensitivity --config configs/llm_matching_8x8.yaml
python -m llm_matching.cli diagnostics --config configs/llm_matching_8x8.yaml          # 5 BT + 5 replay seeds
python -m llm_matching.cli matching-id --config configs/llm_matching_smoke.yaml --n-seeds 20
python -m llm_matching.cli market-analysis --config configs/llm_matching_8x8.yaml
```

- `bootstrap` — statistical stability of the oracle matching under
  TRAIN-instance resampling (outputs/<run>/bootstrap/).
- `sensitivity` — per-arm criticality: utility gaps, bootstrap flip
  probabilities, adjacent-reversal / pair-swap GS sensitivity
  (outputs/<run>/sensitivity/, 448-arm table).
- `diagnostics` — 5-seed BT + replay runs with hindsight metrics
  (first-hit, occupancy, streaks) and unresolved critical/non-critical
  arm counts (outputs/<run>/diagnostics_5seed/).
- `matching-id` — exact assignment-aware certificate
  (`llm_matching/matching_id.py`): stop when every preference profile
  consistent with the certified CIs yields the same model-proposing GS
  matching. Wrapper `MatchingIDP2ETG` keeps P2ETG sampling identical
  and changes only the stopping rule.
- `market-analysis` — full 20x15 LLMRouterBench pool diversity
  diagnostics (outputs/<run>/market_analysis/).

See `PHASE2_REPORT.md` at the repository root for measured results.

## 11. Output files

Under `output_dir` (e.g. `outputs/llm_matching_8x8/`):

| File | Meaning |
|---|---|
| `config_resolved.yaml` | Fully resolved configuration |
| `data_alignment.csv` | Per-dataset alignment stats (raw min/max, aligned, missing fraction) |
| `splits.csv` | `dataset, record_index, split` (60/20/20 deterministic) |
| `task_utility_{train,val,test}.csv` | U_d(m) matrices |
| `model_utility_{train,val,test}.csv` | V_m(d) comparative-advantage matrices |
| `tie_report.json` | Utility tie groups and strictified entries |
| `oracle_preferences_train.json` | Hidden strict TRAIN preference lists |
| `oracle_matching_{train,val,test}.json` | H* per split |
| `oracle_generalization.json` | H*_train == H*_val / H*_test flags |
| `bt_calibration.json` | eta_task, eta_model, median gaps, target win prob |
| `hungarian_matching.json` | Max-welfare (train utility) reference |
| `random_baseline.json` | Random-bijection test-welfare stats |
| `traces/seed_XXX.csv` | One row per P2ETG stopping check: t, matching, exact recovery, stability, blocking pairs, resolved fraction, test welfare, stopped |
| `per_seed_summary.csv` | Final metrics per seed |
| `aggregate_summary.csv` | Mean/std/median/q25/q75 of per-seed metrics |
| `run_meta.json` | algorithm/feedback/seeds of the last run |
| `plots/*.png` | accuracy/welfare/resolution vs. queries, stopping time, heatmaps |
| `matching_report.md` | Human-readable oracle/Hungarian/P2ETG report with sanity checks |

## 12. No real LLM inference

The experiment never loads model weights and never calls OpenAI,
Anthropic, Gemini, Hugging Face inference, vLLM, or an LLM-as-judge.
Only the stored benchmark `score` fields are used. This keeps the
experiment fully offline, reproducible, and cheap.

## Scope

Milestone-1 scope is strictly one-to-one 8x8 matching under
performance-only utilities. The architecture keeps room for later
cost-aware utilities `U_d^lambda(m) = Q_d(m) - lambda * C_d(m)` (see
`routerbench_loader.py`, which already extracts `cost`/token columns).
