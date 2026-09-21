# Preference-Arm Sensitivity Report

* total arms: 448
* adjacent-in-oracle arms: 112 (task side 56, model side 56)
* adjacent arms whose minimal reversal changes the stable matching: **11** (9.8% of adjacent)
* adjacent non-critical: 101

## Critical vs non-critical adjacent-arm gap statistics

| group | n | min | median | mean | max |
|---|---|---|---|---|---|
| critical | 11 | 0.00000 | 0.04464 | 0.05742 | 0.23991 |
| non-critical | 101 | 0.00000 | 0.03629 | 0.06210 | 0.34913 |

## Pair-swap sensitivity (all arms, not minimal for non-adjacent)

* arms whose position-swap changes the matching: 134 / 448

## Critical arms (adjacent, minimal reversal)

| side | agent | partner_1 | partner_2 | gap | flip prob | bt_true_prob | rank_1 | rank_2 | changes |
|---|---|---|---|---|---|---|---|---|---|
| task | math500 | DeepSeek-R1-0528-Qwen3-8B | NVIDIA-Nemotron-Nano-9B-v2 | 0.00000 | 0.892 | 0.500 | 1 | 2 | 2 |
| task | mmlupro | Llama-3.1-8B-Instruct | Qwen2.5-Coder-7B-Instruct | 0.00166 | 0.476 | 0.503 | 6 | 7 | 2 |
| model | gemma-2-9b-it | meld | mmlupro | 0.00175 | 0.500 | 0.502 | 4 | 5 | 3 |
| task | meld | gemma-2-9b-it | Qwen2.5-Coder-7B-Instruct | 0.00271 | 0.418 | 0.505 | 2 | 3 | 3 |
| task | bbh | NVIDIA-Nemotron-Nano-9B-v2 | Qwen3-8B | 0.03086 | 0.024 | 0.552 | 2 | 3 | 2 |
| model | Qwen3-8B | medqa | finqa | 0.04464 | 0.024 | 0.557 | 3 | 4 | 4 |
| task | finqa | Fin-R1 | Qwen2.5-Coder-7B-Instruct | 0.04942 | 0.002 | 0.583 | 4 | 5 | 3 |
| model | Qwen2.5-Coder-7B-Instruct | korbench | livecodebench | 0.05505 | 0.022 | 0.570 | 7 | 8 | 2 |
| model | DeepSeek-R1-0528-Qwen3-8B | math500 | bbh | 0.06549 | 0.000 | 0.583 | 1 | 2 | 2 |
| model | NVIDIA-Nemotron-Nano-9B-v2 | bbh | medqa | 0.14014 | 0.000 | 0.671 | 2 | 3 | 2 |
| model | Llama-3.1-8B-Instruct | mmlupro | korbench | 0.23991 | 0.000 | 0.772 | 6 | 7 | 2 |
