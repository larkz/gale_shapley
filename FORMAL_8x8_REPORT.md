# FORMAL 8×8 EXPERIMENT REPORT — P2ETG, 30 Seeds, BT + Replay

Run: 2026-09-20, branch `llm-routing-experiment`
Setup: 8 models x 8 tasks, deterministic 60/20/20 split (seed 3407),
BT calibrated to eta_task=6.7431 / eta_model=8.2401 (median-gap win
prob 0.70), budget 200,000 pairwise observations, check_every=448,
**PYTHONHASHSEED=0** (cross-process reproducible; `gs_lib.bt._canonical`
uses Python `hash()`), seeds 0–29.

Outputs: `outputs/llm_matching_8x8/` (BT) and
`outputs/llm_matching_8x8_replay/` (replay) — per-seed summaries,
traces, per-check arm-criticality counts, plots, regret analysis.

---

## 1. Stopping and recovery

| metric (30 seeds) | BT | replay |
|---|---|---|
| certified/all-pairs stops | **0/30** | **0/30** |
| final exact recovery | 5/30 (16.7%) | 2/30 (6.7%) |
| first oracle hit | 14/30 (46.7%), median t=27,552 | 5/30 (16.7%), median t=103,040 |
| oracle occupancy (mean) | 0.087 (median 0.000) | 0.014 (median 0.000) |
| longest oracle streak | mean 20.7, max 115 | mean 3.2, max 54 |
| matching changes | mean 37.5 (20–80) | mean 79.6 (43–121) |
| ever train-stable | 14/30 | 5/30 |

**Non-stopping is structural** (Phase-2): the two exact-tie arms
(math500: DS-R1 == Nemotron; medqa: UltraMedical == Llama-Instruct)
have BT p = 0.5000002; CIs can never exclude 1/2 within budget. The
oracle occupancy confirms Phase-2's transient-hit picture at 30 seeds:
early exact hits exist (BT min first-hit t=1,792) but do not persist.

## 2. Resolution and criticality at budget exhaustion

| metric | BT | replay |
|---|---|---|
| resolved fraction (final) | 0.851 ± 0.008 | 0.401 ± 0.015 |
| unresolved CRITICAL arms | **5.5 / 7** (range 4–7) | **6.8 / 7** (range 6–7) |
| unresolved non-critical arms | 61.4 | 261.4 |

**Q3 answer holds at 30 seeds**: when P2ETG cannot stop, the
unresolved arms are NOT mostly irrelevant — 79% (BT) to 97% (replay) of
the 7 matching-critical arms remain unresolved. Criticality and
unresolvability share the same cause (utility gaps <= 0.05).

## 3. Welfare and regret

| metric | BT | replay |
|---|---|---|
| final test welfare | 5.129 ± 0.097 | 5.109 ± 0.088 |
| best welfare along trace | 5.230 (max 5.252 = market best) | 5.214 |
| normalized welfare (vs random/Hungarian) | **0.854** | 0.830 |
| final regret vs H*_train | 0.093 | 0.112 |
| best regret along trace | −0.009 | +0.008 |
| cumulative regret (end) | 19,587 ± 12,775 | 35,301 ± 15,046 |
| average regret CR/T @10K → @end | 0.142 → 0.098 | 0.394 → 0.177 |
| asymptotic regret (tail slope) | 0.100 | 0.123 |

References (test utilities): oracle H*_train 5.2215, Hungarian(train)
5.2438, test ceiling 5.2524, random mean 4.4522.

## 4. Consistency with the 5-seed Phase-2 diagnostics

Every conclusion of the 5-seed diagnostic study replicates at 30 seeds
(and is now hash-seed reproducible): 0 stops, structural non-stopping,
~5.5/7 critical arms unresolved (BT), resolved 0.85/0.40, welfare gap
BT≈replay, transient early hits with near-zero occupancy, cumulative
regret BT < replay at equal budget.

## 5. Answer to the main scientific question

> Can the correct stable LLM-task assignment be identified using
> substantially fewer pairwise capability comparisons than are
> required to recover the complete preference profile?

**On this 8×8 market: no.** The correct matching is visited early in
~half the BT runs (median t≈27K, 14% of budget) but the learner cannot
KNOW this without resolving the near-tie comparisons — and those are
precisely the ones that also decide the assignment (7 critical arms,
all gaps <= 0.05, two exact ties). Full-preference identification and
assignment identification are entangled here: T_assignment-ID is not
measurably smaller than T_full-ID **when the oracle itself is
statistically fragile** (Phase-2 bootstrap: P(H_b == H_train) = 0.114,
49 distinct matchings). The 4×4 studies show the gap CAN open on
better-separated markets (certified stops at 18–70% of budget).

## 6. Reproduction

```bash
PYTHONHASHSEED=0 python scripts/run_formal_30seed.py --feedback bt
PYTHONHASHSEED=0 python scripts/run_formal_30seed.py --feedback replay \
    --output-dir outputs/llm_matching_8x8_replay
PYTHONHASHSEED=0 python -m llm_matching.cli regret --config configs/llm_matching_8x8.yaml
PYTHONHASHSEED=0 python -m llm_matching.cli regret --config configs/llm_matching_8x8.yaml \
    --output-dir outputs/llm_matching_8x8_replay
```

Phase-1 single-seed trace preserved in
`outputs/llm_matching_8x8/phase1_backup/`.
