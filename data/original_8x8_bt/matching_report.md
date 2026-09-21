# LLM-Task Stable Matching Report

* algorithm: `p2etg`, feedback mode: `bt`
* datasets: 8, models: 8
* BT calibration: target median win prob = 0.7, median task gap = 0.125654, median model gap = 0.102826
* eta_task = 6.7431, eta_model = 8.2401
* ties strictified (train): task side 4 entries, model side 0 entries

## Oracle model-proposing stable matching (H*_train)

| Task | Assigned model | task-rank of model | model-rank of task | train utility U | test utility U | comparative adv. V (train) |
|---|---|---|---|---|---|---|
| mmlupro | Llama-3.1-8B-Instruct | 6 | 4 | 0.4559 | 0.4900 | -0.1058 |
| livecodebench | Qwen3-8B | 1 | 1 | 0.6667 | 0.7014 | +0.3620 |
| medqa | Llama-3.1-8B-UltraMedical | 4 | 1 | 0.6846 | 0.6929 | +0.0103 |
| meld | gemma-2-9b-it | 2 | 1 | 0.5359 | 0.5425 | +0.0336 |
| math500 | DeepSeek-R1-0528-Qwen3-8B | 1 | 2 | 0.9467 | 0.9200 | +0.2610 |
| bbh | NVIDIA-Nemotron-Nano-9B-v2 | 2 | 3 | 0.8627 | 0.8287 | +0.2271 |
| finqa | Fin-R1 | 4 | 2 | 0.6919 | 0.6739 | +0.0484 |
| korbench | Qwen2.5-Coder-7B-Instruct | 4 | 3 | 0.3347 | 0.3720 | -0.0415 |

## Hungarian maximum-welfare matching (train utility)

| Task | Hungarian model | Oracle model | same? |
|---|---|---|---|
| mmlupro | gemma-2-9b-it | Llama-3.1-8B-Instruct | no |
| livecodebench | Qwen3-8B | Qwen3-8B | yes |
| medqa | Llama-3.1-8B-UltraMedical | Llama-3.1-8B-UltraMedical | yes |
| meld | Llama-3.1-8B-Instruct | gemma-2-9b-it | no |
| math500 | NVIDIA-Nemotron-Nano-9B-v2 | DeepSeek-R1-0528-Qwen3-8B | no |
| bbh | DeepSeek-R1-0528-Qwen3-8B | NVIDIA-Nemotron-Nano-9B-v2 | no |
| finqa | Fin-R1 | Fin-R1 | yes |
| korbench | Qwen2.5-Coder-7B-Instruct | Qwen2.5-Coder-7B-Instruct | yes |

## Welfare comparison (test utilities)

| Matching | Test welfare |
|---|---|
| Oracle GS (H*_train) | 5.2215 |
| Hungarian (train util) | 5.2438 |
| Random (mean over seeds) | 4.4522 (std 0.2790) |

## Preference generalization across splits

* H*_train == H*_val: False
* H*_train == H*_test: False
* mean Spearman rank correlation (task side, train vs test): 0.9239
* mean Spearman rank correlation (model side, train vs test): 0.9018

## Sanity checks

* Dataset-wise best model ignoring capacity: 3 distinct model(s) chosen across 8 tasks -> specialization exists (multiple distinct best models).
| Task | Best model (ignoring capacity) | Oracle model |
|---|---|---|
| mmlupro | DeepSeek-R1-0528-Qwen3-8B | Llama-3.1-8B-Instruct |
| livecodebench | Qwen3-8B | Qwen3-8B |
| medqa | DeepSeek-R1-0528-Qwen3-8B | Llama-3.1-8B-UltraMedical |
| meld | Qwen3-8B | gemma-2-9b-it |
| math500 | DeepSeek-R1-0528-Qwen3-8B | DeepSeek-R1-0528-Qwen3-8B |
| bbh | DeepSeek-R1-0528-Qwen3-8B | NVIDIA-Nemotron-Nano-9B-v2 |
| finqa | Qwen3-8B | Fin-R1 |
| korbench | NVIDIA-Nemotron-Nano-9B-v2 | Qwen2.5-Coder-7B-Instruct |

* Oracle stable matching differs from the Hungarian welfare-optimal matching.

## p2etg (bt) per-seed results

| seed | stopped | T_stop | exact_oracle_match | stable_train | blocking_pairs_train | resolved_fraction_at_stop | test_welfare | normalized_test_welfare |
|---|---|---|---|---|---|---|---|---|
| 0 | False | 200000 | False | False | 2 | 0.8527 | 5.1639 | 0.8991 |
| 1 | False | 200000 | True | True | 0 | 0.8415 | 5.2215 | 0.9718 |
| 2 | False | 200000 | False | False | 3 | 0.8438 | 5.0165 | 0.7129 |
| 3 | False | 200000 | False | False | 3 | 0.8460 | 5.0165 | 0.7129 |
| 4 | False | 200000 | True | True | 0 | 0.8438 | 5.2215 | 0.9718 |
| 5 | False | 200000 | False | False | 3 | 0.8393 | 5.0165 | 0.7129 |
| 6 | False | 200000 | True | True | 0 | 0.8348 | 5.2215 | 0.9718 |
| 7 | False | 200000 | False | False | 3 | 0.8571 | 5.0165 | 0.7129 |
| 8 | False | 200000 | False | False | 1 | 0.8504 | 5.2524 | 1.0108 |
| 9 | False | 200000 | False | False | 1 | 0.8616 | 5.1021 | 0.8210 |
| 10 | False | 200000 | False | False | 1 | 0.8571 | 5.1330 | 0.8600 |
| 11 | False | 200000 | False | False | 2 | 0.8438 | 4.9856 | 0.6738 |
| 12 | False | 200000 | False | False | 1 | 0.8594 | 5.2524 | 1.0108 |
| 13 | False | 200000 | False | False | 2 | 0.8482 | 4.9856 | 0.6738 |
| 14 | False | 200000 | False | False | 2 | 0.8616 | 5.1639 | 0.8991 |
| 15 | False | 200000 | False | False | 1 | 0.8549 | 5.1021 | 0.8210 |
| 16 | False | 200000 | False | False | 1 | 0.8527 | 5.2524 | 1.0108 |
| 17 | False | 200000 | False | False | 1 | 0.8460 | 5.2524 | 1.0108 |
| 18 | False | 200000 | False | False | 2 | 0.8438 | 4.9856 | 0.6738 |
| 19 | False | 200000 | False | False | 1 | 0.8504 | 5.1021 | 0.8210 |
| 20 | False | 200000 | False | False | 2 | 0.8571 | 5.1330 | 0.8600 |
| 21 | False | 200000 | False | False | 1 | 0.8415 | 5.1330 | 0.8600 |
| 22 | False | 200000 | True | True | 0 | 0.8638 | 5.2215 | 0.9718 |
| 23 | False | 200000 | False | False | 2 | 0.8504 | 4.9856 | 0.6738 |
| 24 | False | 200000 | False | False | 2 | 0.8571 | 5.1639 | 0.8991 |
| 25 | False | 200000 | True | True | 0 | 0.8348 | 5.2215 | 0.9718 |
| 26 | False | 200000 | False | False | 1 | 0.8594 | 5.2524 | 1.0108 |
| 27 | False | 200000 | False | False | 2 | 0.8549 | 5.1330 | 0.8600 |
| 28 | False | 200000 | False | False | 2 | 0.8638 | 5.1330 | 0.8600 |
| 29 | False | 200000 | False | False | 3 | 0.8482 | 5.0165 | 0.7129 |
