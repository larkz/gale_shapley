# DIVERSE-MARKET 8×8 REPORT — Three-Algorithm Comparison on a High-Separation Market

Run: 2026-09-20, branch `llm-routing-experiment`, PYTHONHASHSEED=0,
20 seeds x 3 algorithms (P2ETG / Matching-ID / PrefLID) x 2 feedbacks
(BT / replay), budget 200,000 pairwise observations each (PrefLID 7,143
iterations x 28 comparisons).

## 1. Market selection (from the full 20×15 pool)

`scripts/select_diverse_market.py` + variant preflight (see
`outputs/llm_matching_8x8_diverse_korbench/market_selection_note.md`):

| market | P_bootstrap | ambiguous tasks | critical arms | min critical gap |
|---|---|---|---|---|
| original 8×8 | 0.114 | 6/8 | 7 | 0.000 (2 exact ties) |
| diverse (kandk) | 0.284 | 2/8 | 7 | 0.000 (exact tie) |
| **diverse (korbench)** | **0.538** | **0/8** | **6** | **0.010** |

Selected market (task-rank Spearman −0.062 vs original 0.55):
meld→glm-4-9b-chat, medqa→DeepSeek-R1, humaneval→Qwen2.5-Coder,
math500→GLM-Z1, korbench→Intern-S1-mini, emorynlp→gemma-2-9b-it,
mbpp→Fin-R1, finqa→Qwen3-8B (8/8 distinct task-side top-1).
Oracle H*_train == Hungarian(train) here (welfare 5.3826 on test).

## 2. Three-algorithm results (BT, 20 seeds)

| metric | P2ETG | Matching-ID | PrefLID |
|---|---|---|---|
| certified stops | 0/20 | 0/20 | 0/20 |
| final exact recovery | **20/20** | 19/20 | **20/20** |
| oracle occupancy | 0.818 | 0.840 | — (see note) |
| first-hit median t | 14,112 | 5,824 | — |
| resolved at 200K | 0.836 | 0.830 | 0.828 |
| final test welfare | 5.3826 (= oracle) | 5.3775 | 5.3826 |
| final regret vs H* | **0.000** | 0.005 | **0.000** |
| asymptotic regret (tail slope) | **0.000** | 0.009 | **0.000** |
| cumulative regret | 3,584 | 2,997 | — |

Replay (20 seeds): exact 4/20 (p2etg), 11/20 (matching-id), 9/20
(preflid); resolved ~0.37; final regret 0.057/0.044/0.059.

Note: PrefLID's per-check regret/occupancy are undefined on this run —
all 7,143 iterations failed the |Ω| ≤ 100 enumeration budget (6,892
over_budget + 251 insufficient_data, 0 ok rounds), so no per-round
matching was recorded; committed-regret from the summary is exact.

## 3. The certification bottleneck, isolated

Even on this 4.7×-more-stable market with 100% final recovery and 0.82
occupancy, **no algorithm certifies within 200K**:

* **Matching-ID**: the last certification check (search complete, 2
  profiles) finds **2 distinct GS matchings** among the profiles still
  consistent with the CIs. The single blocking arm is
  (math500: GLM-Z1 vs DS-R1, TRAIN gap 0.010 → BT p = 0.528): with the
  CI constant 0.1 it needs n ≈ 1,500–1,700 samples/arm ⇒ ≈ 700–750K
  total budget to resolve — 3.5× the 200K budget.
* **P2ETG**: full-preference stopping additionally needs every
  non-critical near-tie resolved — strictly harder.
* **PrefLID**: at 8×8 the island certificate is blocked upstream by
  the enumeration budget (|Ω| = product of within-block permutations
  over 16 agents > 100 even at 83% resolution), not by sample size.

**Structural law observed across all four markets**: certification
requires resolving ALL matching-critical arms; the binding constraint
is the smallest critical gap g_min, and with width √(c·ln t / n) the
required budget scales as t_cert ≈ 448 · c · ln t / (2·g_min·η)² ~
1/g_min². Empirically: g_min = 0.010 (diverse) ⇒ ~750K; at 4×4 with
favourable gaps ⇒ certified at 8–70% of budget.

## 4. Diverse vs original market (BT)

| | original (30 seeds) | diverse (20 seeds) |
|---|---|---|
| bootstrap stability | 0.114 | 0.538 |
| final exact (P2ETG) | 5/30 (16.7%) | **20/20 (100%)** |
| oracle occupancy | 0.087 | **0.818** |
| first-hit median t | 27,552 | 14,112 |
| final regret (P2ETG) | 0.093 | **0.000** |
| asymptotic regret | 0.100 | **0.000** |
| cumulative regret | 19,587 | **3,584** (−82%) |
| certified stops | 0 | 0 |

## 5. Conclusions

1. **Market structure determines everything measurable**: raising
   bootstrap stability from 0.114 to 0.538 turns the learner's
   behaviour from transient visits (occupancy 0.09) into a stable
   hold (occupancy 0.82), perfect final recovery, and zero regret —
   with the SAME algorithm, budget and CI machinery.
2. **Assignment-identification vs full-preference-identification**:
   on the diverse market the learner effectively KNOWS the assignment
   from t≈14K (first-hit) onward (occupancy 0.82, asymptotic regret
   exactly 0), yet cannot PROVE it: certification is bottlenecked by
   one arm with p = 0.528 and needs ~3.5× the budget. This is the
   cleanest demonstration of T_assignment-ID ≪ T_full-ID to date —
   in behaviour, not yet in certificate.
3. **What separates "knowing" from "proving"**: the CI constant. With
   c = 0.1 the half-width at n = 450/arm is ≈ 0.076; an arm at |p−0.5|
   = 0.028 is invisible. A certificate at the current budget needs a
   larger c (which slows every other resolution) or an
   epsilon-stability formulation that accepts the math500 ambiguity
   (both orderings yield welfare within 0.01).

## Reproduction

```bash
python scripts/select_diverse_market.py          # selection + rationale
PYTHONHASHSEED=0 python -m llm_matching.cli bootstrap \
    --config configs/llm_matching_8x8_diverse_korbench.yaml --num-bootstrap 500
PYTHONHASHSEED=0 python -m llm_matching.cli sensitivity \
    --config configs/llm_matching_8x8_diverse_korbench.yaml
PYTHONHASHSEED=0 python -m llm_matching.cli matching-id \
    --config configs/llm_matching_8x8_diverse_korbench.yaml --n-seeds 20
PYTHONHASHSEED=0 python -m llm_matching.cli regret \
    --config configs/llm_matching_8x8_diverse_korbench.yaml
```
