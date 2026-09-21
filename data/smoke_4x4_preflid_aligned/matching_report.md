# LLM-Task Stable Matching Report

* algorithm: `preflid`, feedback mode: `bt`
* datasets: 4, models: 4
* BT calibration: target median win prob = 0.7, median task gap = 0.178149, median model gap = 0.183058
* eta_task = 4.7561, eta_model = 4.6286
* ties strictified (train): task side 0 entries, model side 0 entries

## Oracle model-proposing stable matching (H*_train)

| Task | Assigned model | task-rank of model | model-rank of task | train utility U | test utility U | comparative adv. V (train) |
|---|---|---|---|---|---|---|
| finqa | Qwen2.5-Coder-7B-Instruct | 3 | 2 | 0.6424 | 0.6130 | -0.0044 |
| livecodebench | DeepSeek-R1-0528-Qwen3-8B | 1 | 1 | 0.6288 | 0.6825 | +0.4623 |
| math500 | Fin-R1 | 2 | 1 | 0.7633 | 0.7400 | +0.0700 |
| medqa | Llama-3.1-8B-UltraMedical | 2 | 1 | 0.6846 | 0.6929 | +0.0519 |

## Hungarian maximum-welfare matching (train utility)

| Task | Hungarian model | Oracle model | same? |
|---|---|---|---|
| finqa | Qwen2.5-Coder-7B-Instruct | Qwen2.5-Coder-7B-Instruct | yes |
| livecodebench | DeepSeek-R1-0528-Qwen3-8B | DeepSeek-R1-0528-Qwen3-8B | yes |
| math500 | Fin-R1 | Fin-R1 | yes |
| medqa | Llama-3.1-8B-UltraMedical | Llama-3.1-8B-UltraMedical | yes |

## Welfare comparison (test utilities)

| Matching | Test welfare |
|---|---|
| Oracle GS (H*_train) | 2.7284 |
| Hungarian (train util) | 2.7284 |
| Random (mean over seeds) | 2.1791 (std 0.2869) |

## Preference generalization across splits

* H*_train == H*_val: True
* H*_train == H*_test: True
* mean Spearman rank correlation (task side, train vs test): 0.9500
* mean Spearman rank correlation (model side, train vs test): 1.0000

## Sanity checks

* Dataset-wise best model ignoring capacity: 1 distinct model(s) chosen across 4 tasks -> weak specialization (a single model dominates every task).
| Task | Best model (ignoring capacity) | Oracle model |
|---|---|---|
| finqa | DeepSeek-R1-0528-Qwen3-8B | Qwen2.5-Coder-7B-Instruct |
| livecodebench | DeepSeek-R1-0528-Qwen3-8B | DeepSeek-R1-0528-Qwen3-8B |
| math500 | DeepSeek-R1-0528-Qwen3-8B | Fin-R1 |
| medqa | DeepSeek-R1-0528-Qwen3-8B | Llama-3.1-8B-UltraMedical |

* Oracle stable matching equals the Hungarian welfare-optimal matching.

## preflid (bt) per-seed results

| seed | stopped | T_stop | exact_oracle_match | stable_train | blocking_pairs_train | resolved_fraction_at_stop | test_welfare | normalized_test_welfare |
|---|---|---|---|---|---|---|---|---|
| 0 | False | 10002 | True | True | 0 | 0.9167 | 2.7284 | 1.0000 |
| 1 | True | 6066 | False | False | 1 | 0.8542 | 2.6193 | 0.8013 |
| 2 | True | 7614 | True | True | 0 | 0.8333 | 2.7284 | 1.0000 |
| 3 | False | 10002 | True | True | 0 | 0.8750 | 2.7284 | 1.0000 |
| 4 | False | 10002 | False | False | 1 | 0.8958 | 2.6193 | 0.8013 |
| 5 | True | 4692 | True | True | 0 | 0.8750 | 2.7284 | 1.0000 |
| 6 | True | 2994 | True | True | 0 | 0.8333 | 2.7284 | 1.0000 |
| 7 | False | 10002 | True | True | 0 | 0.8542 | 2.7284 | 1.0000 |
| 8 | True | 3270 | False | False | 1 | 0.8125 | 2.6193 | 0.8013 |
| 9 | True | 4368 | False | False | 1 | 0.8542 | 2.6193 | 0.8013 |
| 10 | False | 10002 | True | True | 0 | 0.8125 | 2.7284 | 1.0000 |
| 11 | True | 3528 | True | True | 0 | 0.8542 | 2.7284 | 1.0000 |
| 12 | False | 10002 | True | True | 0 | 0.8750 | 2.7284 | 1.0000 |
| 13 | False | 10002 | True | True | 0 | 0.8958 | 2.7284 | 1.0000 |
| 14 | False | 10002 | True | True | 0 | 0.8542 | 2.7284 | 1.0000 |
| 15 | True | 3120 | True | True | 0 | 0.8542 | 2.7284 | 1.0000 |
| 16 | False | 10002 | False | False | 1 | 0.8333 | 2.6193 | 0.8013 |
| 17 | True | 9948 | False | False | 1 | 0.8542 | 2.6193 | 0.8013 |
| 18 | True | 4404 | True | True | 0 | 0.8333 | 2.7284 | 1.0000 |
| 19 | True | 1854 | True | True | 0 | 0.7917 | 2.7284 | 1.0000 |
