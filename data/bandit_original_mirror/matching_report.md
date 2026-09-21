# LLM-Task Stable Matching Report

* algorithm: `p2etg`, feedback mode: `bt`
* datasets: 8, models: 8
* BT calibration: target median win prob = 0.7, median task gap = 0.125654, median model gap = 0.166592
* eta_task = 6.7431, eta_model = 5.0861
* ties strictified (train): task side 64 entries, model side 0 entries

## Oracle model-proposing stable matching (H*_train)

| Task | Assigned model | task-rank of model | model-rank of task | train utility U | test utility U | comparative adv. V (train) |
|---|---|---|---|---|---|---|
| mmlupro | Llama-3.1-8B-Instruct | 6 | 6 | 0.4559 | 0.4900 | +0.4559 |
| medqa | Qwen3-8B | 2 | 3 | 0.7801 | 0.7795 | +0.7801 |
| livecodebench | Llama-3.1-8B-UltraMedical | 7 | 7 | 0.1485 | 0.0900 | +0.1485 |
| meld | gemma-2-9b-it | 2 | 4 | 0.5359 | 0.5425 | +0.5359 |
| math500 | DeepSeek-R1-0528-Qwen3-8B | 1 | 1 | 0.9467 | 0.9200 | +0.9467 |
| bbh | NVIDIA-Nemotron-Nano-9B-v2 | 2 | 2 | 0.8627 | 0.8287 | +0.8627 |
| finqa | Fin-R1 | 4 | 2 | 0.6919 | 0.6739 | +0.6919 |
| korbench | Qwen2.5-Coder-7B-Instruct | 4 | 7 | 0.3347 | 0.3720 | +0.3347 |

## Hungarian maximum-welfare matching (train utility)

| Task | Hungarian model | Oracle model | same? |
|---|---|---|---|
| mmlupro | gemma-2-9b-it | Llama-3.1-8B-Instruct | no |
| medqa | Llama-3.1-8B-UltraMedical | Qwen3-8B | no |
| livecodebench | Qwen3-8B | Llama-3.1-8B-UltraMedical | no |
| meld | Llama-3.1-8B-Instruct | gemma-2-9b-it | no |
| math500 | NVIDIA-Nemotron-Nano-9B-v2 | DeepSeek-R1-0528-Qwen3-8B | no |
| bbh | DeepSeek-R1-0528-Qwen3-8B | NVIDIA-Nemotron-Nano-9B-v2 | no |
| finqa | Fin-R1 | Fin-R1 | yes |
| korbench | Qwen2.5-Coder-7B-Instruct | Qwen2.5-Coder-7B-Instruct | yes |

## Welfare comparison (test utilities)

| Matching | Test welfare |
|---|---|
| Oracle GS (H*_train) | 4.6967 |
| Hungarian (train util) | 5.2438 |
| Random (mean over seeds) | 4.4522 (std 0.2790) |

## Preference generalization across splits

* H*_train == H*_val: False
* H*_train == H*_test: True
* mean Spearman rank correlation (task side, train vs test): 0.9239
* mean Spearman rank correlation (model side, train vs test): 0.9435

## Sanity checks

* Dataset-wise best model ignoring capacity: 3 distinct model(s) chosen across 8 tasks -> specialization exists (multiple distinct best models).
| Task | Best model (ignoring capacity) | Oracle model |
|---|---|---|
| mmlupro | DeepSeek-R1-0528-Qwen3-8B | Llama-3.1-8B-Instruct |
| medqa | DeepSeek-R1-0528-Qwen3-8B | Qwen3-8B |
| livecodebench | Qwen3-8B | Llama-3.1-8B-UltraMedical |
| meld | Qwen3-8B | gemma-2-9b-it |
| math500 | DeepSeek-R1-0528-Qwen3-8B | DeepSeek-R1-0528-Qwen3-8B |
| bbh | DeepSeek-R1-0528-Qwen3-8B | NVIDIA-Nemotron-Nano-9B-v2 |
| finqa | Qwen3-8B | Fin-R1 |
| korbench | NVIDIA-Nemotron-Nano-9B-v2 | Qwen2.5-Coder-7B-Instruct |

* Oracle stable matching differs from the Hungarian welfare-optimal matching.

## p2etg (bt) per-seed results

| seed | stopped | T_stop | exact_oracle_match | stable_train | blocking_pairs_train | resolved_fraction_at_stop | test_welfare | normalized_test_welfare |
|---|---|---|---|---|---|---|---|---|
| 0 | False | 200000 | False | False | 2 | 0.8438 | 4.6391 | 0.2361 |
| 1 | False | 200000 | True | True | 0 | 0.8482 | 4.6967 | 0.3089 |
| 2 | False | 200000 | False | False | 2 | 0.8415 | 4.6391 | 0.2361 |
| 3 | False | 200000 | False | False | 2 | 0.8527 | 4.6391 | 0.2361 |
| 4 | False | 200000 | False | False | 1 | 0.8460 | 4.6082 | 0.1971 |
