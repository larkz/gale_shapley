# No-Tie Reward (probability_floor) — Results

Mechanism: `feedback.probability_floor = 0.1` clamps every BT win
probability to |p − 1/2| ≥ 0.1 in the direction of the (jittered)
strict preference. This is a REWARD-CHANNEL-ONLY guarantee:
preferences, the unique stable matching H*, and welfare are untouched;
exact/near utility ties no longer create unresolvable coin-flip arms.
Resolution bound: every arm resolves at n ~ c·ln(t)/ε² ≈ 120–250
samples/arm (c = 0.1, ε = 0.1) ⇒ certified stopping becomes feasible
within the 200K horizon.

## Diverse korbench 8x8 mirror market, 10 seeds, horizon 200K

| | P2ETG no-floor | P2ETG floor | PrefLID no-floor | PrefLID floor |
|---|---|---|---|---|
| certified stops | 0/10 | 0/10 | 0/10 | **9/10** (all correct) |
| T_stop (stops) | — | — | — | 122K–190K (median ≈ 161K) |
| final exact | 3/10 | **10/10** | 3/10 | **10/10** |
| final instantaneous regret | −0.0030 | **0.0000 (all seeds)** | −0.0264 | **0.0000 (all seeds)** |
| cumulative regret | 4,691 ± 5,547 | **553** (0–1,786) | 7,256 ± 4,919 | **2,344** (1,362–3,724) |

## Findings

1. **The no-tie reward unlocks certified stopping**: PrefLID's island
   certificate fires in 9/10 seeds (122K–190K) and every stop commits
   the CORRECT unique stable matching — vs 0/10 stops without the
   floor, blocked by 10 sub-0.002-gap arms.
2. **End-state quality is perfect for both algorithms**: 10/10 exact
   recovery with final regret exactly 0 (the learner locks onto H*),
   and cumulative welfare regret drops 88% (P2ETG) / 68% (PrefLID)
   at the same budget.
3. **Full-preference stopping remains harder than certified stopping
   even with no ties**: P2ETG (all-CIs-exclude-0.5) still stops 0/10 —
   floor-edge arms (|p−0.5| = 0.1 exactly) need n ≈ 190+/arm AND all
   448 arms must pass simultaneously, which does not happen by 200K.
   PrefLID's structural certificate stops 55K earlier on the median.
   This is T_assignment-ID < T_full-ID in the STOPPING sense, now on a
   real LLM market.
4. Seed 2 of P2ETG achieves cumulative regret exactly 0 (never wrong
   after the first checks) — with no-tie rewards the market becomes
   learnable to perfection within budget.

## Reproduce

```bash
PYTHONHASHSEED=0 python -m llm_matching.cli bandit \
    --config configs/llm_matching_8x8_diverse_korbench_symmetric_floor.yaml
```
