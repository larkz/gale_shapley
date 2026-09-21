# UPSTREAM REPLICATION REPORT — larkz/gale_shapley experiments on a real LLM market

Replicates origin/main `experiment.py` (P2ETG) and `experiment_preflid.py`
(PrefLID) with ONLY the market and reward sourced from real
LLMRouterBench data (diverse korbench 8×8, full-data utilities,
eta-calibrated BT at target 0.70). Everything else follows upstream:
dense per-round timeline, post-stop tail fill (200 rounds), cumulative
**0/1 identification regret** (rounds where the current matching ≠ H*),
upstream summary schema (correct_at_stop / stable_under_truth /
stable_under_hat), per-run dirs with rounds.csv + summary.json +
config.json, all_rounds/all_summaries CSVs, upstream-style plots.
`PYTHONHASHSEED=0`. Harness: `llm_matching/upstream_rep.py`, CLI
`upstream-rep`.

## Upstream configurations found (hardcoded in main(), no YAML files)

| | experiment.py (P2ETG) | experiment_preflid.py |
|---|---|---|
| market | N=10, K=10, Dirichlet(alpha=2.0) per side | N=3, K=3 |
| seeds | 50 | 100 |
| learner params | adaptive=True, check_every=25, max_samples=100,000, constant=0.25 | budget=300,000, constant=0.005, max_iterations=10,000, horizon=10,000 |
| metrics | dense 0/1 regret, T_stop, correct_at_stop, stability under truth/hat | same + horizon fill |

## Replication results on the REAL diverse-korbench 8×8 market

### P2ETG (50 seeds, upstream params)

* stopped (all CIs exclude 0.5): **0/50** — the real market's near-tie
  arms (min critical gap 0.010, BT p=0.528) cannot resolve within
  100K, exactly as in our Phase-2/formal findings. Upstream's synthetic
  Dirichlet market has no such ties, so upstream experiments stop.
* final matching == H*: **29/50 (58%)**; stable under TRUE
  preferences: 29/50; stable under estimated: 50/50.
* cumulative 0/1 regret: mean 54,524 / 100,000 rounds (±31,701);
  checkpoint means 984 @1K → 4,604 @5K → 8,651 @10K → 33,301 @50K →
  54,524 @100K. Early growth ~linear (98% of rounds wrong before t≈1K),
  then slows (learner sits right ~50% of the time in the tail).

### PrefLID (5 seeds, upstream params: constant=0.005, budget=300K)

* stopped: **5/5** (certificates fire at T_stop = 8,092–82,012 — the
  huge enumeration budget lets the island machinery run at 8×8).
* **correct: 1/5.** Four of five certifications commit a WRONG
  matching: constant=0.005 gives CI half-width √(0.005·ln t/n) with
  per-pair failure prob 2t^(−2·0.005) ≈ 2 — essentially no coverage
  guarantee. On upstream's 3×3 synthetic market the same parameter
  happened to work; on the real market's near-ties it false-certifies
  80% of the time.

## Reading

1. **The upstream pipeline transfers cleanly**: every stage (market →
   oracle GS → BT provider → learner → dense timeline → 0/1 regret →
   sweep harness → plots) runs on real data with only the
   market/reward swapped, as intended.
2. **What does NOT transfer is the upstream's implicit assumption of a
   tie-free, well-separated market.** With real utilities:
   P2ETG never certifies (near-tie arms), and PrefLID's upstream
   constant (0.005) false-certifies 4/5. The real-market-safe settings
   are exactly the ones our phases identified: c=0.1–0.25 for
   P2ETG-style resolution, c≥0.2 (and gaps ≥0.03 or epsilon-stability)
   for sound certification.
3. **0/1 identification regret is much harsher than welfare regret**:
  the learner is wrong-matching ~55% of rounds at 100K on the diverse
  market even though its welfare regret is ~0 there (E4) — near-tie
  alternatives are welfare-equivalent but not identical.

## Deviations from upstream (all documented in-code)

* market 8×8 real (upstream: 10×10 / 3×3 synthetic); eta-calibrated BT
  instead of raw Dirichlet theta; exact ties jittered;
* dense rows drop the matching_str column (disk: ~1.7 GB → ~25 MB);
  oracle/committed strings kept in summary.json;
* PrefLID horizon = actual end time (upstream fixed 10,000 for 3×3).

## Reproduce

```bash
PYTHONHASHSEED=0 python -m llm_matching.cli upstream-rep \
    --config configs/llm_matching_upstream_rep_diverse.yaml --n-seeds 50
PYTHONHASHSEED=0 python -m llm_matching.cli upstream-rep --preflid \
    --config configs/llm_matching_upstream_rep_diverse.yaml --seed 0 1 2 3 4
```

Outputs: `outputs/llm_matching_upstream_rep_diverse/runs_upstream/{p2etg,preflid}/`
(per-run dirs, all_rounds.csv, all_summaries.csv, plots/).
