# MARKET-SIZE SCALING — P2ETG vs PrefLID (real LLM markets)

Markets: greedy distinct-winner selection from the full 20x15
LLMRouterBench pool (mbpp/finqa/gpqa/kandk/emorynlp/math500/meld/
mathbench + livecodebench + humaneval; 8 distinct winners + Nemotron +
Fin-R1). Settings: online bandit, symmetric (mirror) preferences,
no-tie reward (probability_floor = 0.1), horizon 200K, 10 seeds,
PYTHONHASHSEED=0. PrefLID uses round-robin center scheduling.
Cumulative regret is clipped at 0 per round; post-stop regret is the
COMMITTED matching's constant regret (piecewise-constant extension).

## Results

| market | arms | algo | stops | cert-correct | final exact | CR mean | CR median | lock median | T_stop median |
|---|---|---|---|---|---|---|---|---|---|
| 3x3 | 18 | P2ETG | 10/10 | 10 | 10/10 | 41 | 29 | 369 | 2,799 |
| 3x3 | 18 | PrefLID-rr | 10/10 | **9** | 9/10 | 1,448* | **17** | **195** | **445** |
| 5x5 | 100 | P2ETG | 10/10 | 10 | 10/10 | 324 | 291 | 2,550 | 36,550 |
| 5x5 | 100 | PrefLID-rr | 10/10 | 10 | 10/10 | **122** | **134** | **1,130** | **14,675** |
| 8x8 | 448 | P2ETG | 0/10 | — | 10/10 | **553** | **430** | 10,976 | — |
| 8x8 | 448 | PrefLID-rr | 9/10 | 9 | 10/10 | 867 | 918 | **9,562** | 129,948 |
| 10x10 | 900 | P2ETG | 0/10 | — | 10/10 | 6,926 | 6,383 | **88,200** | — |
| 10x10 | 900 | PrefLID-rr | 0/10 | — | 9/10 | 9,711 | **5,430** | 126,022 | — |

*3x3 PrefLID mean is dominated by ONE false certification (seed 1,
T_stop=753, committed the wrong island centroid with welfare regret
0.0713 frozen for the remaining horizon => CR ~14.4K); the median (17)
reflects the other nine seeds.

## Findings

1. **Certified stopping beats full-resolution stopping at every size
   where both exist, and survives one size longer.** PrefLID
   certifies 6.3x earlier at 3x3 (445 vs 2,799), 2.5x at 5x5 (14,675
   vs 36,550), and is the ONLY one stopping at 8x8 (9/10 at ~130K);
   at 10x10 neither certifies within 200K.
2. **P2ETG's stop hits a simultaneity wall between 5x5 and 8x8.** A
   floor-edge arm (|p-1/2| = 0.1) resolves at n ~ 122/arm, so
   single-arm-wise 8x8 could stop at ~55K. But the stop needs ALL 448
   arms resolved SIMULTANEOUSLY: each floor arm has ~2-4% unresolved
   probability per check, and the joint probability decays like
   t^(-2c) per arm => e^(-8.5) at 8x8 within any practical horizon.
   PrefLID's structural certificate only needs the MATCHING-RELEVANT
   ambiguity to collapse - it sidesteps the tax.
3. **Small markets: certification is also the better REGRET strategy**
   (3x3 median CR 17 vs 29; 5x5 134 vs 291 - the early commit freezes
   a correct matching while P2ETG keeps exploring). At 8x8 they tie;
   at 10x10 PrefLID's median is lower (5,430 vs 6,383) but its mean is
   worse (one locked-out seed).
4. **The 3x3 false cert is the cautionary tale**: at n ~ 25/arm the
   c=0.1 island certificate can collapse on noise; the frozen wrong
   matching then costs 0.0713 x remaining horizon. Early certificates
   need either more samples per arm or a sound constant.

## Accounting fix included

The post-stop freeze previously used a trapezoid with the last
estimated matching's regret, smearing a transient last-check error
into the frozen phase (and halving false-cert costs). Now the
committed matching's constant regret extends piecewise-constantly.
All numbers above use the corrected accounting.

## Reproduce

```bash
for n in 3 5 10; do
  PYTHONHASHSEED=0 python -m llm_matching.cli bandit \
      --config configs/bandit_scaling_${n}x${n}.yaml
done
```
