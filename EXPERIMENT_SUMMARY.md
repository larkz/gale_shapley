# EXPERIMENT SUMMARY — Offline LLM-Task Stable Matching (as of 2026-09-21)

Branch `llm-routing-experiment` (base `main` @ cb7cfe9), commits
`b0ba0b3 → e893f53` (14 commits), 64 tests passing. All experiments
run under `PYTHONHASHSEED=0` (required: `gs_lib.bt._canonical` uses
Python `hash()`).

**One-line summary**: on one-to-one stable matching between LLM models
and task types, learned only from noisy pairwise capability
comparisons (no LLM inference), we measured how market structure —
not the algorithm — decides whether the correct assignment can be
(a) recovered, (b) *certified*, and (c) certified earlier than the
full preference profile.

---

## 1. How the dataset / market is constructed

### 1.1 Source data (fully offline)

LLMRouterBench pre-collected results (HuggingFace `NPULH/LLMRouterBench`,
`bench-release.tar.gz`, 1.28 GB): per-(dataset, model) JSON files with
per-instance `score` (plus cost/token fields, unused in this phase).
Full pool: **20 models × 15 datasets**. No LLM inference anywhere; only
stored benchmark scores are read.

### 1.2 Alignment (`llm_matching/routerbench_loader.py`)

1. Each (dataset, split, model) JSON → records `{index: {score,...}}`.
2. Per dataset: intersect record indices across all selected models
   (missing entries → hard error; alignment stats per dataset).
3. Quality gate: `min_aligned_records = 100`.
4. Selected markets have **zero missing entries** after alignment.

### 1.3 Split

Deterministic 60/20/20 train/val/test per dataset (seed 3407, no RNG —
index-arithmetic hashing). Same record goes to the same split for all
models.

### 1.4 Latent utilities (TRAIN split)

* **Task side** (task = woman): `U_d(m) = mean_{x∈train} s_{m,d,x}`.
* **Model side** (model = man): comparative advantage
  `V_m(d) = U_d(m) − (1/(M−1)) Σ_{m'≠m} U_d(m')`
  (avoids "everyone prefers the easy task" degeneracy).
* Exact ties → deterministic SHA256-based jitter ≤ `tie_epsilon=1e-6`
  (never Python `hash()`).

### 1.5 Oracle and baselines

* `H*_train` = model-proposing Gale-Shapley on the strictified TRAIN
  preferences (hidden from the learner; evaluation only).
* Hungarian on TRAIN utilities (welfare-optimal train-legal matching);
  random bijection baseline (200 seeds); test-side welfare ceiling.

### 1.6 Feedback providers (`llm_matching/providers.py`)

* **BT** (parametric): `theta = exp(eta·U)` per side; `eta` calibrated
  so the median positive utility gap maps to win prob 0.70
  (e.g. original market: eta_task=6.74, eta_model=8.24).
* **Replay** (empirical, robustness mode): task side samples a TRAIN
  instance, higher stored score wins (ties → coin); model side compares
  per-instance comparative advantages, `P = clip(0.5 + γ·Δa, ε, 1−ε)`
  (γ=0.25). Substantially noisier than BT (≈0.40 vs 0.85 resolution at
  200K on the original 8×8).

### 1.7 The three markets

| | original 8×8 | smoke 4×4 | diverse 8×8 (korbench) |
|---|---|---|---|
| purpose | Phase-1/2 backbone | Matching-ID lab | market-structure test |
| selected by | hand | subset | `scripts/select_diverse_market.py` + preflight |
| datasets | math500, livecodebench, bbh, mmlupro, finqa, medqa, meld, korbench | math500, livecodebench, finqa, medqa | emorynlp, finqa, humaneval, korbench, math500, mbpp, medqa, meld |
| models | DS-R1-Qwen3-8B, Fin-R1, Llama-UltraMedical, Qwen2.5-Coder, Qwen3-8B, Llama-3.1-8B-Instruct, Nemotron-9B-v2, gemma-2-9b | first 4 of the above | DS-R1, Fin-R1, GLM-Z1-9B, Intern-S1-mini, Qwen2.5-Coder, Qwen3-8B, gemma-2-9b, glm-4-9b-chat |
| bootstrap stability P(H_b=H*) | **0.114** | — | **0.538** |
| distinct bootstrap matchings | 49 | — | 30 |
| ambiguous tasks (max freq<0.7) | 6/8 | — | **0/8** |
| matching-critical arms | 7/448 | — | 6/448 |
| min critical gap | **0.000 (2 exact ties)** | — | **0.010 (no tie)** |
| task-rank Spearman (mean pairwise) | 0.55 | — | **−0.062** |

Oracle matchings: original — math500→DS-R1, livecodebench→Qwen3-8B,
bbh→Nemotron, mmlupro→Llama-Inst, finqa→Fin-R1, medqa→UltraMedical,
meld→gemma, korbench→Qwen2.5-Coder. Diverse — meld→glm-4-9b-chat,
medqa→DS-R1, humaneval→Qwen2.5-Coder, math500→GLM-Z1,
korbench→Intern-S1-mini, emorynlp→gemma-2-9b, mbpp→Fin-R1,
finqa→Qwen3-8B (8/8 distinct task-side top-1; H* == Hungarian).

The diverse market was chosen by deterministic-winner greedy (one
dataset per distinct full-pool top-1 model, margin-threshold cascade,
≥300 records) with variant preflight: the greedy kandk pick was
rejected on an exact TRAIN tie (Intern-S1 == Qwen3-8B, P_boot 0.284);
korbench variant won (P_boot 0.538).

---

## 2. Algorithms compared

All learners see ONLY pairwise Bernoulli feedback via the
`SignalProvider` interface ("does agent a prefer b1 over b2?").

| | sampling | stopping certificate |
|---|---|---|
| **P2ETG** | uniform round-robin over all 448 arms; CI `sqrt(c·ln t/n)` per arm | ALL pairwise CIs exclude 0.5 (full-preference ID) |
| **Matching-ID** (ours, Phase-2) | identical to P2ETG (same estimator, same stream) | all preference profiles that are linear extensions of the certified CI edges yield the SAME model-proposing GS matching (exact enumeration, caps/timeout/cycle-safe; incomplete search never certifies) |
| **PrefLID** (upstream `larkin/preflid_tuning`, ported with provider support) | RRT: one center agent round-robins the opposite side per iteration; `min_sample_ratio=0.5` balance gate | single lattice island over the configurations consistent with current partitions (|Ω| ≤ budget=100 enumeration) |

Reference matchings: oracle GS (identification target), Hungarian
(train welfare optimum), random bijection.

---

## 3. Experiment matrix and results

### E1 — Original 8×8, P2ETG, 30 seeds × {BT, replay}, budget 200K
(`FORMAL_8x8_REPORT.md`)

| | BT | replay |
|---|---|---|
| certified stops | 0/30 (structural: 2 exact-tie arms have p=0.5000002) | 0/30 |
| final exact recovery | 5/30 | 2/30 |
| first oracle hit | 14/30, median t=27,552 (earliest 1,792) | 5/30, median t≈103K |
| oracle occupancy | 0.087 (median 0 — transient visits) | 0.014 |
| resolved at exhaustion | 0.851 | 0.401 |
| unresolved critical arms | **5.5/7** | **6.8/7** |
| test welfare (normalized) | 5.129 (0.854) | 5.109 (0.830) |
| cumulative regret | 19,587 ± 12,775 | 35,301 ± 15,046 |

### E2 — Bootstrap & sensitivity diagnostics (500 replicates, 448 arms)
(`PHASE2_REPORT.md` Sections A–B; per-market table in §1.7 above)

Answers: Q1 oracle fragile on the original market (P=0.114, 49
matchings; only finqa→Fin-R1 0.984 stable); Q2 only **7/448** pairwise
preferences determine the matching — all gaps ≤ 0.05, two exact ties;
Q3 unresolved arms at exhaustion are NOT irrelevant (they ARE the
critical ones; criticality and unresolvability share the cause: tiny
gaps).

### E3 — 4×4 smoke, three algorithms, 20 seeds × {BT, replay}
(`outputs/llm_matching_smoke/matching_id/`)

| BT | stops | false-cert | T_stop median | final exact |
|---|---|---|---|---|
| P2ETG @10K | 0/20 | — | — | 13/20 |
| Matching-ID @10K | 4/20 | 1 (traced to a genuine Hoeffding CI failure, arm p=0.490 @n=128) | 6,264 | **16/20** |
| PrefLID @1.2K (200 iters) | 0/20 | — | — | 14/20 |
| PrefLID @10K (aligned, 1667 iters) | **11/20** | **4/11** | **4,368** | 14/20 |

Replay @10K: P2ETG 15/20, Matching-ID 12/20 (0 stops), PrefLID-aligned
13/20 (0 stops). Ordering: stop-speed PrefLID > Matching-ID > P2ETG;
certificate soundness Matching-ID > PrefLID (both limited by CI
coverage); non-certified final quality all comparable.

### E4 — Diverse 8×8 (korbench), three algorithms, 20 seeds × {BT, replay}, 200K
(`DIVERSE_MARKET_REPORT.md`)

| BT | stops | final exact | occupancy | asymp regret | cum regret |
|---|---|---|---|---|---|
| P2ETG | 0/20 | **20/20** | 0.818 | **0.000** | 3,584 |
| Matching-ID | 0/20 | 19/20 | 0.840 | 0.009 | 2,997 |
| PrefLID (200K aligned) | 0/20 | 20/20 | — | 0.000 | — |

Replay: exact 4/20 (P2ETG), 11/20 (Matching-ID), 9/20 (PrefLID).
**Zero certifications despite perfect recovery**: Matching-ID's last
check finds 2 distinct GS matchings still CI-consistent — blocked by
ONE arm (math500: GLM-Z1 vs DS-R1, gap 0.010 → p=0.528, needs ~750K);
PrefLID is blocked upstream by |Ω|>100 enumeration at 8×8
(6,892/7,143 iterations over_budget).

### E5 — Regret analysis family (`llm_matching/regret.py`, CLI `regret`)

Instantaneous regret `W_ref − W_test(H_t)` (refs: H*, Hungarian, test
ceiling, random); cumulative (trapezoid, post-stop frozen at the
committed matching); average-regret convergence CR/T with log-log decay
panel; commitment-freeze figure (certified-correct runs → CR exactly
parallel to the x-axis; verified numerically: seed 8 CR ≡ 35.58 after
t=816). Diverse market: BT asymptotic regret exactly 0; cumulative
regret −82% vs original market with the same algorithm and budget.

### E6 — CI-constant soundness–delay scan (`scripts/run_c_scan.py`)
Matching-ID, 4×4 BT, 20 seeds, budget 100K, **identical sample streams
across c** (controlled: c only changes CI width):

| c | stops | correct | false-cert | T_stop median | per-pair CI fail prob @10K |
|---|---|---|---|---|---|
| 0.1 | 7/20 | 6 | 1 | 7,032 | 0.32 |
| 0.2 | 1/20 | 1 | 0 | 34,344 | 0.05 |
| 0.5 | 0/20 | — | — | — | 2e-4 |
| 1.0 | 0/20 | — | — | — | 2e-8 |

Final exact 17/20 for every c (design check). **The tradeoff is a
cliff, not a dial**: soundness kills certification.

---

## 4. Cross-experiment conclusions

1. **Market structure dominates.** Same algorithm, budget, CIs:
   bootstrap stability 0.114 → 0.538 changes occupancy 0.087 → 0.818,
   final recovery 17% → 100%, asymptotic regret 0.100 → 0.000,
   cumulative regret −82%.
2. **T_assignment-ID ≪ T_full-ID holds behaviourally, not yet
   certificate-wise.** On the diverse market the learner effectively
   knows the assignment from t≈14K (first-hit; regret plateau 0), but
   cannot prove it: one 0.010-gap arm blocks the exact certificate
   until ~750K. Certification budget scales ≈ 1/g_min².
3. **Criticality is sparse and tie-shaped.** 6–7 of 448 arms decide
   the matching; they are exactly the near/exact ties; on fragile
   markets they are also the unresolvable arms (entanglement).
4. **No free soundness in this CI family.** c=0.1 certificates are
   exploration-heuristic (false-cert 1/7–4/11 among stops); c≥0.2
   sound certificates stop firing at practical budgets. Paths forward:
   epsilon-stability certificates (accept orderings within welfare ε)
   or markets with min critical gap ≥ ~0.03.

## 5. Artifact index

| what | where |
|---|---|
| Phase-2 report (bootstrap/sensitivity/diagnostics/4×4) | `PHASE2_REPORT.md` |
| Formal 30-seed original 8×8 | `FORMAL_8x8_REPORT.md`, `outputs/llm_matching_8x8{,_replay}/` |
| Diverse-market selection + comparison | `DIVERSE_MARKET_REPORT.md`, `outputs/llm_matching_8x8_diverse_korbench/` |
| PrefLID budget-aligned rerun | `outputs/llm_matching_smoke_preflid_full/comparison_vs_p2etg_matching_id.md` |
| c-scan | `outputs/llm_matching_smoke/c_scan/` |
| Regret family (per run dir) | `outputs/<run>/regret/` |
| Market analysis (full 20×15 pool) | `outputs/llm_matching_8x8/market_analysis/` |
| Launchers | `scripts/{fetch_routerbench_results,run_formal_30seed,select_diverse_market,run_c_scan}.py` |
| Package | `llm_matching/` (16 modules), tests `tests/llm_matching/` (64) |

Reproduction: every study is one CLI command (`python -m
llm_matching.cli ...`), always with `PYTHONHASHSEED=0`; configs in
`configs/llm_matching_*.yaml`.
