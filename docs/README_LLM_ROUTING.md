# gale_shapley_llm_routing

Offline LLM-task **stable matching** experiments on real benchmark  
data: can a one-to-one assignment between LLM models and task types be  
learned — and *certified* — from noisy pairwise capability  
comparisons, without ever running an LLM?

Built on the [larkz/gale\_shapley](https://github.com/larkz/gale_shapley)  
matching-bandit codebase (P2ETG / PrefLID learners, GS/BT/island  
libraries), replacing the synthetic Dirichlet market with markets  
constructed from **LLMRouterBench** (20 models x 15 datasets,  
per-instance scores only — no inference in the loop).

## What is here

```
gs_lib/          Gale-Shapley, Bradley-Terry MLE/CIs, lattice islands
p2etg.py         P2ETG learner (+ SignalProvider interface)
preflid.py       PrefLID learner (RRT sampling; random/round_robin centers)
llm_matching/    the LLM-routing experiment package:
                   routerbench_loader  alignment of benchmark results
                   utilities           U/V utilities, tie strictification,
                                       symmetric (mirror) matrices
                   providers           BT (eta-calibrated, optional
                                       probability_floor) + replay feedback
                   oracle / metrics / runner / diagnostics
                   matching_id         exact GS-uniqueness certificate
                   bandit              online matching-bandit studies
                   upstream_rep        replication of the upstream
                                       experiment.py / experiment_preflid.py
                   bootstrap / sensitivity / market_analysis / regret / plots
tests/           83 tests (pytest)
configs/         one YAML per study (markets, floors, scaling grid)
scripts/         data fetch, market selection, formal-run launchers
docs/            ENVIRONMENT.md (full environment construction),
                 EXPERIMENT_SUMMARY.md, and per-phase reports
data/            curated experiment outputs: per-seed summaries,
                 aggregate tables, plots, reports, oracle/baseline
                 JSONs, market diagnostics (traces excluded — every
                 study is one CLI command to regenerate)
```

Read **docs/ENVIRONMENT.md first** — it specifies the full pipeline:  
LLMRouterBench -> alignment -> (split) -> utilities U/V ->  
tie-strictified preferences -> GS oracle -> calibrated BT / replay /  
floored reward providers, for every configuration used.

## Headline results (all real markets, PYTHONHASHSEED=0)

| finding                                                                                                                                                                                                                                     | where                                                   |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| Market structure dominates: bootstrap stability 0.114 -> 0.538 turns 17% final recovery into 100% and asymptotic welfare regret 0.100 into exactly 0, same algorithm & budget                                                               | docs/DIVERSE_MARKET_REPORT.md                           |
| T_assignment-ID < T_full-ID in the STOPPING sense: with no-tie rewards (floor 0.1), PrefLID's structural certificate stops 6.3x/2.5x earlier than P2ETG's full-resolution stop at 3x3/5x5, and is the only one stopping at 8x8 (9/10 seeds) | data/bandit_scaling_summary.csv, docs/SCALING_REPORT.md |
| P2ETG's full-resolution stop pays a "simultaneity tax" (ALL arms resolved at once), which phase-changes between 5x5 and 8x8                                                                                                                 | docs/SCALING_REPORT.md                                  |
| Certified early stopping is also the better regret strategy on small markets (median CR 17 vs 29 at 3x3) — with one 3x3 false-cert cautionary tale                                                                                          | data/bandit_scaling_3x3, docs/SCALING_REPORT.md         |
| Round-robin center scheduling fixes PrefLID's multinomial-straggler problem (CR 2,344 -> 867; lock 42K -> 9.6K)                                                                                                                             | data/bandit_diverse_mirror_floor_rr                     |
| CI-constant soundness-delay is a cliff, not a dial: c=0.1 false-certifies, c>=0.2 stops firing within 10x budget                                                                                                                            | data/smoke_4x4_c_scan                                   |
| Upstream synthetic-market assumptions (tie-free, small) do NOT transfer to real markets: upstream PrefLID params false-certify 4/5                                                                                                          | docs/REPLICATION_REPORT.md                              |

## Quickstart

```bash
pip install -r requirements-llm-matching.txt
python scripts/fetch_routerbench_results.py   # once; expects ../LLMRouterBench

# online matching bandit on the diverse mirror market with no-tie rewards
PYTHONHASHSEED=0 python -m llm_matching.cli bandit \
    --config configs/llm_matching_8x8_diverse_korbench_symmetric_floor.yaml

# three-algorithm comparison study (P2ETG / Matching-ID / PrefLID)
PYTHONHASHSEED=0 python -m llm_matching.cli matching-id \
    --config configs/llm_matching_smoke.yaml --n-seeds 20

# market diagnostics: bootstrap stability + arm criticality
PYTHONHASHSEED=0 python -m llm_matching.cli bootstrap \
    --config configs/llm_matching_8x8_diverse_korbench.yaml --num-bootstrap 500
PYTHONHASHSEED=0 python -m llm_matching.cli sensitivity \
    --config configs/llm_matching_8x8_diverse_korbench.yaml

pytest tests/llm_matching -q                    # 83 tests
```

`PYTHONHASHSEED=0` is required for reproducibility (see  
docs/ENVIRONMENT.md §9).

## Provenance

Forked from `larkz/gale_shapley` @ `cb7cfe9` (origin/main), branch  
`zilong/LLM_routing`; the llm_matching package and all experiments  
were developed on `llm-routing-experiment` (commit history preserved  
there). Upstream learner semantics are preserved — PrefLID keeps the  
upstream `random` center policy as its default; `round_robin` and the  
`probability_floor` reward guarantee are opt-in extensions documented  
in docs/ENVIRONMENT.md.
