# First Execution Report — Offline LLM-Task Stable Matching (Milestone 4)

Date: 2026-09-18 · Market: 8 models x 8 tasks · Feedback: BT (calibrated) · Algorithm: P2ETG

## 1. Git commit / branch

- Repository: `larkz/gale_shapley` clone at `/Users/russelzwang/WorkBuddy/Matching_Experiment/gale_shapley`
- Branch: `llm-routing-experiment` (base: `main` @ `cb7cfe9`)
- Commit: `b0ba0b3` "Add offline LLM-task stable matching experiment (llm_matching)"

## 2. Files added / modified

Added: `llm_matching/` (12 files: loader, splits, utilities, oracle, providers, baselines,
metrics, runner, plots, cli, README, `__init__`), `configs/llm_matching_8x8.yaml`,
`configs/llm_matching_smoke.yaml`, `scripts/fetch_routerbench_results.py`,
`tests/llm_matching/` (8 files, 45 tests).
Modified: `preflid.py` (optional `SignalProvider` param, backward compatible),
`.gitignore` (`outputs/`, `.pytest_tmp/`). `gs_lib/` and `p2etg.py` untouched.

## 3. Data coverage of the 8x8 matrix

Complete. Every selected (dataset, model) pair exists; every dataset resolves to a
split containing all 8 models (mmlupro -> `test_1000`). No missing entries.

## 4. Aligned records per dataset (intersection over all 8 models)

| dataset | on-disk id | split | raw_min | raw_max | aligned | missing_frac |
|---|---|---|---|---|---|---|
| math500 | math500 | test | 500 | 500 | 500 | 0.0000 |
| livecodebench | livecodebench | test | 1055 | 1055 | 1055 | 0.0000 |
| bbh | bbh | test | 1080 | 1080 | 1080 | 0.0000 |
| mmlupro | mmlupro | test_1000 | 1001 | 1001 | 1001 | 0.0000 |
| finqa | finqa | test | 1147 | 1147 | 1147 | 0.0000 |
| medqa | medqa | test | 1273 | 1273 | 1273 | 0.0000 |
| meld | meld | test | 1232 | 1232 | 1232 | 0.0000 |
| korbench | korbench | test | 1250 | 1250 | 1250 | 0.0000 |

## 5. Train / val / test sizes (60/20/20, seed 3407, per dataset train/val/test)

math500 300/100/100 · livecodebench 633/211/211 · bbh 648/216/216 · mmlupro 601/200/200 ·
finqa 688/229/230 · medqa 764/254/255 · meld 739/247/246 · korbench 750/250/250.
Total: 5323 train / 1707 val / 1708 test.

## 6. Task utility matrix U_d(m), TRAIN (percent)

| dataset | DS-R1-Qwen3-8B | Fin-R1 | UltraMedical | Qwen2.5-Coder | Qwen3-8B | Llama-3.1-8B-Inst | Nemotron-9B-v2 | gemma-2-9b |
|---|---|---|---|---|---|---|---|---|
| math500 | **94.7** | 76.3 | 46.0 | 67.3 | 94.0 | 51.7 | **94.7** | 50.0 |
| livecodebench | 62.9 | 7.1 | 14.8 | 28.0 | **66.7** | 18.0 | 64.8 | 17.7 |
| bbh | **88.1** | 59.4 | 37.7 | 56.3 | 83.2 | 59.7 | 86.3 | 60.5 |
| mmlupro | **70.7** | 48.9 | 38.4 | 45.4 | 67.2 | 45.6 | 69.1 | 53.4 |
| finqa | 71.1 | 69.2 | 53.8 | 64.2 | **73.5** | 52.5 | 71.2 | 64.1 |
| medqa | **81.0** | 61.1 | 68.5 | 47.6 | 78.0 | 68.5 | 72.3 | 63.5 |
| meld | 52.0 | 51.3 | 43.4 | 53.3 | 54.0 | 48.0 | 49.5 | **53.6** |
| korbench | 55.5 | 32.3 | 11.9 | 33.5 | 53.2 | 21.6 | **55.6** | 33.3 |

Exact ties (strictified by deterministic SHA256 jitter): math500 DS-R1 == Nemotron
(0.946667), medqa UltraMedical == Llama-3.1-8B-Instruct (0.684555).

## 7. Model comparative-advantage matrix V_m(d), TRAIN (x100)

| model | math500 | livecode | bbh | mmlupro | finqa | medqa | meld | korbench |
|---|---|---|---|---|---|---|---|---|
| DS-R1-Qwen3-8B | +26.1 | +31.9 | +24.8 | +18.1 | +7.0 | +15.4 | +1.5 | +21.0 |
| Fin-R1 | +5.1 | -31.9 | -8.0 | -6.8 | +4.8 | -7.3 | +0.7 | -5.5 |
| UltraMedical | -29.5 | -23.0 | -32.8 | -18.8 | -12.8 | **+1.0** | -8.2 | -28.8 |
| Qwen2.5-Coder | -5.1 | -8.0 | -11.5 | -10.8 | -0.8 | -22.8 | +3.1 | -4.2 |
| Qwen3-8B | +25.3 | **+36.2** | +19.2 | +14.1 | +9.8 | +11.9 | +3.8 | +18.4 |
| Llama-3.1-8B-Inst | -23.0 | -19.4 | -7.6 | -10.6 | -14.3 | +1.0 | -3.0 | -17.7 |
| Nemotron-9B-v2 | +26.1 | +34.0 | +22.7 | +16.2 | +7.2 | +5.4 | -1.3 | **+21.1** |
| gemma-2-9b | -25.0 | -19.8 | -6.7 | -1.6 | -1.0 | -4.7 | **+3.4** | -4.3 |

## 8. Oracle model-proposing stable matching H*_train

| task | model |
|---|---|
| math500 | DeepSeek-R1-0528-Qwen3-8B |
| livecodebench | Qwen3-8B |
| bbh | NVIDIA-Nemotron-Nano-9B-v2 |
| mmlupro | Llama-3.1-8B-Instruct |
| finqa | Fin-R1 |
| medqa | Llama-3.1-8B-UltraMedical |
| meld | gemma-2-9b-it |
| korbench | Qwen2.5-Coder-7B-Instruct |

H*_train == H*_val: **False**; H*_train == H*_test: **False** (test swaps bbh/korbench
partners). Spearman(train, test): task side mean 0.924 (min 0.810), model side 0.902.

## 9. Hungarian matching (max sum U_train)

Same as oracle on math500? No — differs on 5/8 tasks:
math500->Nemotron (vs DS-R1, tie), mmlupro->gemma (vs Llama-Inst), meld->Llama-Inst
(vs gemma), bbh->DS-R1 (vs Nemotron), korbench->Qwen2.5-Coder (same).
Test welfare: Hungarian 5.2438 vs oracle 5.2215 — stability costs 0.022 test welfare.

## 10. Pairwise utility gap statistics (TRAIN, positive gaps only)

- Task side: 222 gaps; min 0.00133, p25 0.0400, **median 0.12565**, max 0.5956; 14 gaps < 0.01, 29 < 0.02.
- Model side: 224 gaps; min 0.00305, p25 0.0511, **median 0.10283**, max 0.3701; 9 gaps < 0.01, 16 < 0.02.

## 11. BT calibration (target median win prob 0.70)

- eta_task = **6.7431** (median task gap 0.125654)
- eta_model = **8.2401** (median model gap 0.102826)

## 12. One-seed P2ETG result (seed 0, BT, adaptive, check_every=448)

- **Exact oracle matching recovery at t = 12,096** (2.7% of the 200K sample budget;
  27 sweeps over all 448 arms) — but the estimator then drifted off the oracle as
  near-tie pairs flipped; exact match held at only 1/447 checks.
- Best test welfare along the trace: 5.2524 at t = 12,544 (above oracle 5.2215 and
  Hungarian 5.2438 on test).
- Final (t = 200,256): matching differs from H*_train on 4 tasks
  (medqa/mmlupro/meld/korbench rotate); 2 blocking pairs under true TRAIN prefs;
  test welfare 4.9856 (mean score 0.6232), normalized welfare 0.674.

## 13. T_stop

Not stopped: 200,256 samples (= max_samples + 2 partial-check overshoot), 447 checks.

## 14. Exact oracle matching recovery

At commit time: **False** (final matching != H*_train). Hit True once at t = 12,096.

## 15. Resolved preference fraction at stop

**0.8527** (382/448 arms with CI excluding 1/2). The unresolved tail is structural:
the two exact-tie arms have BT p = 0.5 ± 2e-6 (CI can never exclude 1/2), and
~23 arms with gaps < 0.01 have p within [0.49, 0.52].

## 16. Test welfare

P2ETG final 4.9856 · oracle H*_train 5.2215 · Hungarian(train) 5.2438 ·
random 4.4522 (std 0.2790, 200 seeds).

## 17. Tests

`pytest tests/llm_matching -q`: **45 passed, 0 failed** (incl. 4x4 end-to-end BT and
replay, provider canonicalisation, BT calibration, oracle stability, split
reproducibility, alignment). Note: run with `--basetemp` inside the repo under the
WorkBuddy sandbox (system tmp dir is restricted); on a normal machine plain
`pytest tests/llm_matching -q` works.

## 18. Remaining issues

1. **Non-stopping is structural in BT mode**: exact train ties (math500, medqa) and
   ~14 sub-0.01 gaps make some arms' win probability indistinguishable from 0.5.
   P2ETG's all-pairs-CI stopping rule can never fire. Options for the 30-seed run:
   (a) accept and report trace-based metrics; (b) stop on "matching-stable for K
   consecutive checks" instead; (c) restrict the market to models with a minimum
   utility gap. Decision needed before Milestone 4 full run.
2. **Late drift**: the matching is correct early (t~12K) then degrades as
   near-tie estimates flip — GS amplifies tiny preference perturbations into
   different stable matchings. The committed matching should arguably be the
   modal/earliest-stable one rather than the last; worth a follow-up rule.
3. PrefLID validated at 4x4 only (t=1200, no island certification within 200
   iterations — lattice enumeration is the bottleneck, as expected).
4. `python -m llm_matching.cli` requires the venv with pandas/numpy/scipy/pyyaml/
   matplotlib; requirements not yet pinned in `requirements.txt`.
