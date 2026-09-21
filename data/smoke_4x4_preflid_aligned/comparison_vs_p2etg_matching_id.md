# PrefLID Budget-Aligned Rerun (4x4, 20 seeds, PYTHONHASHSEED=0)

PrefLID iteration cap raised 200 -> 1667 so the sampling budget matches the
P2ETG/Matching-ID 10K budget (each iteration = C(4,2) = 6 comparisons).

| run | n | stopped | cert_correct | false_cert | T_stop_med_stopped | exact_final | resolved | welfare | stable |
|---|---|---|---|---|---|---|---|---|---|
| p2etg/bt@10K | 20 | 0 | 0 | 0 | NaN | 13 | 0.8700 | 2.6902 | 13 |
| p2etg/replay@10K | 20 | 0 | 0 | 0 | NaN | 15 | 0.4840 | 2.7011 | 15 |
| matching_id/bt@10K | 20 | 4 | 3 | 1 | 6264 | 16 | 0.8610 | 2.7066 | 16 |
| matching_id/replay@10K | 20 | 0 | 0 | 0 | NaN | 12 | 0.4810 | 2.6781 | 12 |
| preflid/bt@10K | 20 | 0 | 0 | 0 | NaN | 14 | 0.5710 | 2.6957 | 14 |
| preflid/replay@10K | 20 | 0 | 0 | 0 | NaN | 4 | 0.2190 | 2.4868 | 4 |
| preflid/bt@10K-ALIGNED | 20 | 11 | 7 | 4 | 4368 | 14 | 0.8530 | 2.6957 | 14 |
| preflid/replay@10K-ALIGNED | 20 | 0 | 0 | 0 | NaN | 13 | 0.4690 | 2.6869 | 13 |

## Key findings

1. **Island certification now fires**: 11/20 BT seeds certify (median T_stop
   4,368; earliest 1,854) vs 0/20 at 200 iterations, and earlier than
   Matching-ID's GS-uniqueness certificate (4/20 stops, median 6,264).
2. **But false certifications are frequent**: 4/11 PrefLID stops commit a
   wrong matching (vs 1/4 for Matching-ID). The island certificate
   (lattices intersecting into one island, centroidal H*) is easier to
   trigger and more fragile under the weak constant=0.1 CIs than the
   exact all-profiles-agree certificate.
3. **Replay**: still 0/20 certified (resolution 0.47), but final quality
   recovers strongly vs the truncated 200-iteration baseline: exact 13/20
   (was 4/20), welfare 2.687 (was 2.487).
4. Exact recovery is NOT monotone in budget for the certificate: more
   sampling gives more chances to certify, including wrongly.

Outputs: per_seed_summary.csv in both run dirs; comparison_table.csv;
preflid_aligned_per_seed.csv (combined).