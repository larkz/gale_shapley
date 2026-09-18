# PHASE 2 REPORT — Matching Identifiability & Exact Matching-ID

Branch `llm-routing-experiment` (Phase-1 base `b0ba0b3` + Phase-2 commit; see git log).
All numbers below are measured from the committed experiment outputs under
`outputs/llm_matching_8x8/` and `outputs/llm_matching_smoke/matching_id/`.

Market (unchanged from Phase 1): 8 models x 8 tasks, TRAIN-split strictified
preferences, model-proposing GS oracle H*_train. CI math unchanged
(`sqrt(c log t / n)`, c = 0.1). No heuristic stopping rule was used anywhere.

---

## Section A — Bootstrap Oracle Stability (8x8, 500 replicates)

Outputs: `outputs/llm_matching_8x8/bootstrap/` (`bootstrap_report.md`,
`assignment_frequencies.csv`, `exact_matching_frequencies.csv`,
`utility_uncertainty_{task,model}.csv`, `preference_flips.csv`, 3 plots).

* **P(H_bootstrap == H_train) = 0.114** (57/500).
* **49 distinct stable matchings** appear across 500 resamples.
* Top-3 matchings: 15.0% (bbh->DS-R1 variant), 15.0% (medqa->Llama-Inst
  variant), 11.4% (= H*_train itself, ranked 3rd).
* Highly stable assignments (freq >= 0.9): **finqa->Fin-R1 (0.984)**;
  near-stable: livecodebench->Qwen3-8B (0.866).
* Ambiguous tasks (max assignment frequency < 0.7): **bbh (0.532),
  korbench (0.550), math500 (0.490), medqa (0.512), meld (0.602),
  mmlupro (0.344)** — 6 of 8 tasks.
* The most unstable tasks by P(assignment differs from H*_train):
  mmlupro (~0.66), bbh (~0.55), medqa (~0.48), math500 (~0.51) —
  driven by the exact ties (math500: DS-R1 == Nemotron = 0.946667;
  medqa: UltraMedical == Llama-3.1-8B-Instruct = 0.684555) and the
  DS-R1/Nemotron near-tie on bbh.
* Preference reversals: median flip probability 0.000; 31/448 arms
  flip with prob > 0.25; only 3 arms > 0.5 (the two exact-tie arms at
  0.89/0.97 plus one meld arm at 0.55).

**Answer to Q1: the empirical oracle matching is NOT statistically
robust.** It is one of ~50 plausible matchings, retained in only 11.4%
of resamples. Only finqa->Fin-R1 (and to a lesser degree
livecodebench->Qwen3-8B) are individually stable assignments.

---

## Section B — Preference Criticality (8x8, 448 arms)

Outputs: `outputs/llm_matching_8x8/sensitivity/` (`arm_sensitivity.csv`
+ by-flip / by-gap sorted copies, `sensitivity_report.md`, 3 plots).

* Adjacent-in-oracle arms: **112** (56 per side).
* **Adjacent matching-critical arms: 7 (6.2% of adjacent; 1.6% of all
  448 arms)** — a minimal single-pair preference reversal changes the
  stable matching.
* Pair-swap sensitivity (all arms, NOT a minimal reversal for
  non-adjacent pairs): 89/448 swaps change the matching.
* Critical vs non-critical adjacent gaps: critical median gap **0.0127**
  (min 0.0000 — the two exact ties, max 0.0494); non-critical median
  gap 0.0334. **All 7 critical arms have gap <= 0.05 and BT true win
  probability <= 0.583**; the two exact-tie arms sit at p = 0.500000.
* The 7 critical arms: (task math500: DS-R1 vs Nemotron [exact tie]),
  (task medqa: UltraMedical vs Llama-Inst [exact tie]),
  (task meld: gemma vs Qwen2.5-Coder, gap 0.0027),
  (model DS-R1: math500 vs bbh, gap 0.0127),
  (model Nemotron: bbh vs korbench, gap 0.0156),
  (task livecodebench: Qwen3-8B vs Nemotron, gap 0.0190),
  (task finqa: Fin-R1 vs Qwen2.5-Coder, gap 0.0494).
* Bootstrap flip probability of critical arms: 0.892, 0.974, 0.418,
  0.258, 0.190, 0.128, 0.002.

**Answer to Q2: only 7 of 448 pairwise preferences (all of them
near-ties or exact ties) actually affect the stable assignment.** The
matching is pinned by a thin set of knife-edge comparisons; every other
preference can be reversed locally without changing H*.

---

## Section C — P2ETG 5-seed BT (8x8, budget 200,000)

Outputs: `outputs/llm_matching_8x8/diagnostics_5seed/bt/` +
`diagnostic_5seed_summary.csv` (10 rows: 5 BT + 5 replay) + 12 plots in
`diagnostics_5seed/plots/`.

* Stop rate: **0/5** (structural: exact-tie arms have BT p = 0.5000002;
  CIs can never exclude 1/2 within budget).
* First oracle hit: 1/5 seeds (t = 7,168); **oracle occupancy mean
  0.007** (after 10k: 0.006; after 100k: 0.000).
* Longest oracle streak: 13 checks (the hitting seed); 0 for the rest.
* Matching changes along a run: 16-60 (mean 32).
* First train-stable time: 7,168 (1/5 seeds ever produce a stable
  matching under true TRAIN preferences).
* Resolved fraction at end: 0.844-0.859.
* **Unresolved arms at end: 4-6 critical + 58-64 non-critical** (of 7
  critical arms in total).
* Final exact match: 0/5; final stable: 0/5; final test welfare mean
  5.159 (best-so-far 5.252 in 4/5 seeds — equal to the best welfare
  any matching achieves on test).

## Section D — P2ETG 5-seed Replay (8x8, budget 200,000)

* Stop rate: **0/5**.
* First oracle hit: 1/5 (t = 1,344); occupancy mean 0.0004.
* Resolved fraction at end: only **0.375-0.408** — empirical
  per-instance comparisons are much noisier than calibrated BT
  feedback (typical model pair on a dataset is close to an
  instance-level coin flip).
* **Unresolved at end: 6-7 critical + 259-273 non-critical.**
* Final exact: 0/5; final stable: 0/5; final welfare mean 5.167.

**Answer to Q3: no — when P2ETG cannot stop, the unresolved arms are
NOT mostly matching-irrelevant.** 4-7 of the 7 matching-critical arms
remain unresolved at budget exhaustion (BT: 5-6 of 7; replay: 6-7 of
7). The arms that block full-preference stopping are precisely the
arms that matter for the assignment, because criticality and
unresolvability have a common cause: tiny utility gaps. Early exact
hits (t ~ 7K BT / 1.3K replay) occur by luck of estimator ordering and
do not persist (occupancy ~ 0).

---

## Section E — Exact Matching-ID vs P2ETG (4x4 smoke market, 20 seeds x 2 feedbacks)

Outputs: `outputs/llm_matching_smoke/matching_id/`
(`per_seed_summary.csv` 80 rows, `aggregate_summary.csv`, `traces/`
80 prefixed trace files, `plots/{t_stop,resolved_fraction_at_stop}_comparison.png`).
Implementation: `llm_matching/matching_id.py` (CI-induced partial
orders -> linear-extension enumeration -> GS uniqueness certificate);
wrapper `MatchingIDP2ETG` changes ONLY the stopping condition.
Tests: `tests/llm_matching/test_matching_id.py` (11 tests incl. the
key "several profiles remain but all give the same GS" case).

| metric (20 seeds) | P2ETG bt | Matching-ID bt | P2ETG replay | Matching-ID replay |
|---|---|---|---|---|
| certified/early stops | 0/20 | **4/20** (t = 816 / 6120 / 6408 / 7032) | 0/20 | 0/20 |
| T_stop mean | 10,000 (budget) | 9,019 | 10,000 | 10,000 |
| exact oracle recovery (final) | 11/20 | 16/20 | 15/20 | 12/20 |
| first-hit rate / median t | 20/20, 180 | 20/20, 120 | 20/20, 252 | 20/20, 408 |
| oracle occupancy | 0.651 | 0.619 | 0.549 | 0.518 |
| resolved fraction at stop | 0.864 | 0.850 | 0.484 | 0.481 |
| **false certifications** | n/a | **1/20 (seed 11)** | n/a | 0/20 |
| certification overhead | — | mean 0.16 s/run (max 1.75 s) | — | mean 1.06 s/run (max 10.4 s, timeout-capped) |

* The single false certification (seed 11, t = 6,120) was traced to a
  genuine CI failure, not a code bug: arm (Fin-R1: finqa vs math500)
  has true BT win probability 0.490, but at n = 128 the sample gave
  p_hat = 0.586 (2.2 sigma) and the Hoeffding interval
  [0.503, 0.669] excluded both 0.5 and the true p. With the
  repository's constant c = 0.1 the per-pair coverage is weak, so the
  certificate inherits occasional unsoundness. A sound certificate
  needs a larger CI constant — deliberately NOT tuned in this phase
  (section 31).
* When certification fires, it stops 1.5-12x earlier than the 10,000
  budget at 0.67-0.88 resolution — demonstrating the intended
  "assignment identified before full preference profile" behaviour on
  the seeds where the market permits it.
* Certification never fires under replay (0/20): with only ~48%
  resolution the remaining ambiguity genuinely spans multiple stable
  matchings — consistent with Section A/D.

**Answer to Q4: partially.** On the 4x4 real-LLM market the exact
assignment-aware certificate stops early in 4/20 BT seeds (never under
replay), so it does NOT yet deliver a reliable across-the-board
reduction; the binding constraint is not the certificate's search cost
(milliseconds) but genuine preference ambiguity near ties, which is
exactly what Sections A-D show.

---

## Section F — Interpretation (only what the data supports)

1. **The 8x8 oracle is fragile** (P_bootstrap = 0.114, 49 matchings):
   "the correct stable assignment" is not a statistically well-defined
   single object on this market at this sample size; six of eight task
   assignments are ambiguous under resampling.
2. **Criticality is concentrated in near-ties**: 7/448 arms (all gaps
   <= 0.05, two exact ties) determine the matching. Everything else is
   matching-irrelevant locally.
3. **The central hypothesis T_assignment-ID << T_full-preference-ID is
   NOT supported in its strong form on these markets**: the unresolved
   arms at budget exhaustion are dominated (in criticality terms) by
   exactly the arms that matter. Early exact hits (t ~ 1-7K) exist but
   are transient (occupancy ~ 0.00-0.65 at 4x4, ~0.01 at 8x8).
4. **Exact certification works mechanically and cheaply** (sub-second
   mean overhead; sound search with caps/timeouts/cycle handling), and
   commits 1.5-12x early on the 20% of BT seeds where ambiguity
   resolves, but its soundness is limited by the CI constant: 1/20
   false certificates with c = 0.1.
5. **BT vs replay**: calibrated BT feedback resolves ~2x more arms at
   equal budget (0.85 vs 0.38-0.41 at 200K on 8x8) — the empirical
   per-instance signal is substantially noisier than the parametric
   one; conclusions drawn only under BT should be checked under replay
   (as done here).

### Consequences for Phase 3 (candidates, not decisions)
* A sound certificate requires a proper confidence level (larger `c`),
  which will increase T_stop — the trade-off must be measured.
* With exact ties present, any full-resolution stopping rule is
  structurally impossible in BT mode; epsilon-stability or
  tie-margin-aware certificates are the principled next step.
* The full 20x15 pool (8 distinct top-1 models, mean task-rank
  correlation 0.403 — see `outputs/llm_matching_8x8/market_analysis/`)
  offers a more specialized market where assignment identification may
  be better posed.

---

## Verification

* `pytest tests/llm_matching -q`: **59 passed, 0 failed** (45 Phase-1 +
  11 Matching-ID + 3 max_samples overshoot regression).
* max_samples overshoot fixed: all runs now end exactly at 200,000
  (previously 200,256).
* `requirements-llm-matching.txt` added (numpy/pandas/scipy/matplotlib/
  PyYAML/pytest only; no ML frameworks).
* Full-market diagnostics: `outputs/llm_matching_8x8/market_analysis/`
  (15 datasets x 20 models, 8 distinct top-1 models).
