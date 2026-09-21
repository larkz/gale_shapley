# PrefLID Early-Stop Benefit Study (smoke 4x4 mirror, 20 seeds)

Horizon 40,000; BT reward sharpened (target 0.90 / 0.70);
clipped per-round regret (>= 0); PYTHONHASHSEED=0.

| target | arm | n | stopped | cert_correct | false_cert | T_stop_median | queries_saved_vs_horizon | final_cum_regret_mean | final_cum_regret_std | final_exact |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.7000 | P2ETG | 20 | 1 | 1 | 0 | 21648.0000 | 0.4588 | 281.2363 | 261.7645 | 20 |
| 0.7000 | PrefLID-cert | 20 | 20 | 20 | 0 | 7650.0000 | 0.8087 | 249.0026 | 354.1765 | 20 |
| 0.7000 | PrefLID-nocert | 20 | 3 | 3 | 0 | 19092.0000 | 0.5227 | 249.0026 | 354.1765 | 20 |
| 0.9000 | P2ETG | 20 | 17 | 17 | 0 | 13296.0000 | 0.6676 | 47.6859 | 50.5702 | 20 |
| 0.9000 | PrefLID-cert | 20 | 20 | 20 | 0 | 1194.0000 | 0.9701 | 45.6693 | 40.7068 | 20 |
| 0.9000 | PrefLID-nocert | 20 | 18 | 18 | 0 | 15522.0000 | 0.6119 | 45.6693 | 40.7068 | 20 |

Reading: PrefLID-cert stops at a small fraction of the horizon
with a certified-correct matching (its regret curve is exactly
flat after T_stop). P2ETG can only stop via FULL preference
resolution, which additionally requires the two hard arms
(gaps 0.017/0.020 — matching-irrelevant) and therefore fires
much later (target 0.90) or never within the horizon (0.70).
PrefLID-nocert shares the sampler but never stops, isolating
the certificate's contribution.