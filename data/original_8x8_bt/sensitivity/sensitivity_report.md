# Preference-Arm Sensitivity Report

* total arms: 448
* adjacent-in-oracle arms: 112 (task side 56, model side 56)
* adjacent arms whose minimal reversal changes the stable matching: **7** (6.2% of adjacent)
* adjacent non-critical: 105

## Critical vs non-critical adjacent-arm gap statistics

| group | n | min | median | mean | max |
|---|---|---|---|---|---|
| critical | 7 | 0.00000 | 0.01272 | 0.01421 | 0.04942 |
| non-critical | 105 | 0.00133 | 0.03343 | 0.05113 | 0.34913 |

## Pair-swap sensitivity (all arms, not minimal for non-adjacent)

* arms whose position-swap changes the matching: 89 / 448

## Critical arms (adjacent, minimal reversal)

| side | agent | partner_1 | partner_2 | gap | flip prob | bt_true_prob | rank_1 | rank_2 | changes |
|---|---|---|---|---|---|---|---|---|---|
| task | math500 | DeepSeek-R1-0528-Qwen3-8B | NVIDIA-Nemotron-Nano-9B-v2 | 0.00000 | 0.892 | 0.500 | 1 | 2 | 2 |
| task | medqa | Llama-3.1-8B-UltraMedical | Llama-3.1-8B-Instruct | 0.00000 | 0.974 | 0.500 | 4 | 5 | 2 |
| task | meld | gemma-2-9b-it | Qwen2.5-Coder-7B-Instruct | 0.00271 | 0.418 | 0.505 | 2 | 3 | 3 |
| model | DeepSeek-R1-0528-Qwen3-8B | math500 | bbh | 0.01272 | 0.258 | 0.526 | 2 | 3 | 2 |
| model | NVIDIA-Nemotron-Nano-9B-v2 | bbh | korbench | 0.01564 | 0.190 | 0.532 | 3 | 4 | 3 |
| task | livecodebench | Qwen3-8B | NVIDIA-Nemotron-Nano-9B-v2 | 0.01896 | 0.128 | 0.532 | 1 | 2 | 2 |
| task | finqa | Fin-R1 | Qwen2.5-Coder-7B-Instruct | 0.04942 | 0.002 | 0.583 | 4 | 5 | 2 |
