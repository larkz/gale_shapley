# Changes from the first upload

## Fixes

| File | Change |
|---|---|
| `p2etg.py`, `preflid.py` | Replaced by one consistent version of both learners. The first upload combined files from two branches: `preflid.py` imported `SignalProvider` from a `p2etg.py` that did not define it, and assigned a `provider` argument that its constructor did not take (`ImportError`, then `NameError`). `P2ETG` again accepts `true_theta_men` / `true_theta_women` as used by `experiment.py`. On synthetic markets both learners produce `rounds.csv` files byte-identical to those of the learner versions used for the synthetic experiments. |
| `providers.py` | Added. The README referred to it, but it was missing. |
| `simulate_preflid.py` | Fixed a format-string crash in the progress printout. |

## Plotting

| File | Change |
|---|---|
| `plotting.py` | New helper `plot_sweep_figure3_style`, which calls the existing `plot_mean_regret_with_std` with the Figure 3 settings. No existing function changed. |
| `experiment.py` | `main()` draws `plots/mean_cum_regret.pdf/.png` with that helper instead of the diagnostic `plot_regret_with_uncertainty`. The diagnostic averages only the runs still recorded at each round (each run's timeline ends 200 rounds after it stops), so its curve has steps where runs drop out, and its band is a 95% CI. The function itself is unchanged. |
| `experiment_preflid.py` | `main()` draws `plots_preflid/mean_cum_regret.pdf/.png` with the same helper. |
| `make_figure3.py` | Added. Draws the three panels of Figure 3 from run directories. |
| `results/llm_routing/` | Added. The 40 runs plotted in Figure 3c. |

No learner, market generator, regret definition or experiment setting
changed. Running `make_figure3.py` on the included runs gives the
published Figure 3c.

Tests: `python -m pytest tests` gives 81 passed, 2 skipped (the 2 need
the LLMRouterBench raw data).
