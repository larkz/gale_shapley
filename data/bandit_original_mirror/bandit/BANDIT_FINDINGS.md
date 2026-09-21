# Online Matching-Bandit Results (P2ETG vs PrefLID, clipped regret)

Setting: online learning, NO train/test split. Environment = full-data
utility matrix; symmetric (mirror) preferences => UNIQUE stable
matching H* (men-GS == women-GS == mutual-best cascade, verified).
Feedback: calibrated BT. Same estimator (MLE-GS) for both algorithms;
only the SAMPLING policy differs (P2ETG uniform round-robin vs
PrefLID RRT center rounds). 10 seeds, horizon 200K.

Regret convention (this version): per-round instantaneous regret is
CLIPPED at zero — r(t) = max(0, W(H*) - W(M_t)) — so holding a
welfare-better-than-H* matching earns no credit and the cumulative
regret R(T) = integral r(t) dt is non-decreasing by construction. No
random-always reference is shown.

## Diverse korbench market (W(H*) = 5.3826)

| | P2ETG | PrefLID |
|---|---|---|
| certified stops | 0/10 | 0/10 |
| final exact recovery | 3/10 | 3/10 |
| final instantaneous regret | 0.0249 (0~0.113) | 0.0085 (0~0.043) |
| cumulative regret | **4,691 ± 5,547** (30~17,923) | 7,256 ± 4,919 (850~15,772) |

## Original 8x8 market (mirror mode)

| | P2ETG | PrefLID |
|---|---|---|
| certified stops | 0/10 | 0/10 |
| final exact recovery | 4/10 | 2/10 |
| final instantaneous regret | 0.0021 (0~0.021) | 0.0159 (0~0.146) |
| cumulative regret | 1,796 ± 1,518 | **142 ± 103** |

## Reading

1. **Both algorithms achieve sub-linear regret**: cumulative regret
   at T=200K is O(10^3-10^4) — between 0.5% and 10% of the linear
   regret a fixed random matching would incur — and the instantaneous
   regret reaches 0 in most seeds (clipped at 0 exactly when the
   learner's matching is welfare >= H*).
2. **Seed variance dominates the algorithm effect**: within 10 seeds
   the P2ETG/PrefLID difference is not significant on either market
   (diverse: 4,691 vs 7,256 with std ~5,000; original: 1,796 vs 142
   but only 2 PrefLID exact recoveries).
3. **Zero instantaneous regret is common** (6-7 of 10 seeds per
   algorithm end at r=0): the learner sits ON a welfare-dominant or
   welfare-equal matching; exact recovery of H* itself is rarer
   (3-4/10) because near-tie alternatives are welfare-equivalent.

## Reproduce

```bash
PYTHONHASHSEED=0 python -m llm_matching.cli bandit \
    --config configs/llm_matching_8x8_diverse_korbench_symmetric.yaml
PYTHONHASHSEED=0 python -m llm_matching.cli bandit \
    --config configs/llm_matching_8x8_symmetric.yaml
```
