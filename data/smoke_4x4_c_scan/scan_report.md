# CI-Constant Soundness-Delay Scan (Matching-ID, 4x4 BT, 20 seeds)

Budget 100,000 (10x the original), PYTHONHASHSEED=0, identical
sample streams across c (RNG seeds fixed per seed index).

| c | stopped | cert_correct | false_cert | T_stop_mean | T_stop_median | T_stop_min | T_stop_max | stopped_before_10k | resolved_at_stop | final_exact | final_welfare | pair_CI_fail_prob@10K |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1000 | 7.0000 | 6.0000 | 1.0000 | 24675.4286 | 7032.0000 | 2856.0000 | 77544.0000 | 4.0000 | 0.9302 | 17.0000 | 2.7121 | 0.3170 |
| 0.2000 | 1.0000 | 1.0000 | 0.0000 | 34344.0000 | 34344.0000 | 34344.0000 | 34344.0000 | 0.0000 | 0.9354 | 17.0000 | 2.7121 | 0.0502 |
| 0.5000 | 0.0000 | 0.0000 | 0.0000 | NaN | NaN | NaN | NaN | 0.0000 | 0.8917 | 17.0000 | 2.7121 | 0.0002 |
| 1.0000 | 0.0000 | 0.0000 | 0.0000 | NaN | NaN | NaN | NaN | 0.0000 | 0.8469 | 17.0000 | 2.7121 | 0.0000 |

Reading: larger c widens the CI sqrt(c ln t / n) — resolving an
arm with |p-1/2| = g requires n ~ c ln t / g^2, so certification
delay scales ~ linearly in c while the per-pair CI failure
probability 2 t^(-2c) collapses exponentially.

Outputs: scan_per_seed.csv, scan_summary.csv, t_stop_vs_c.png
## Findings (measured, 20 seeds x budget 100K)

| c | stops | correct | false-cert | T_stop median | stops within 10K |
|---|---|---|---|---|---|
| 0.1 | 7/20 | 6 | **1** | 7,032 | 4 |
| 0.2 | 1/20 | 1 | 0 | 34,344 | 0 |
| 0.5 | 0/20 | — | — | — | 0 |
| 1.0 | 0/20 | — | — | — | 0 |

1. **Soundness achieved exactly where certification dies.** Doubling c
   to 0.2 removes the false certification and buys a 4.9x later single
   stop (t=34,344, correct); c >= 0.5 produces zero certifications
   within a 10x-extended budget. The tradeoff curve is a cliff, not a
   dial.
2. **Why**: resolution of a critical arm with |p-1/2| = g needs
   n ~ c ln(t) / g^2, so T_cert scales ~linearly in c — but the
   number of stopping OPPORTUNITIES (checks where all remaining
   ambiguity cancels) also shrinks, and the cliff at c=0.2 shows the
   product falls faster than linear on this market.
3. **Implied CI failure probability** 2 t^(-2c) at t=10K: 0.32 (c=0.1),
   0.05 (0.2), 2e-4 (0.5), 2e-8 (1.0). The single c=0.1 false cert
   (1/7 stops) is consistent with the weak-coverage regime.
4. **Controlled design check**: final exact recovery is 17/20 for
   every c (identical sample streams, MLE ranking unaffected by c) —
   all differences across c come from the certificate strictness
   alone.
5. **Implication for the diverse 8x8 market**: its blocking arm has
   gap 0.010 (p=0.528) and needs ~1,500 samples/arm at c=0.1 (~750K
   total). At c=0.2 the same arm needs ~2x that; making the 8x8
   certificate sound while certifying within budget requires either
   an epsilon-stability formulation or a market whose minimum
   critical gap exceeds ~0.03.

Bottom line: with this CI family there is no c that is simultaneously
sound and certifying on these markets. The c=0.1 certificate is best
read as an EXPLORATION-heuristic stop (great early hits, ~14% false
rate among stops), not a sound guarantee.
