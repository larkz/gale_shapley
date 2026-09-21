# Preference-Arm Sensitivity Report

* total arms: 448
* adjacent-in-oracle arms: 112 (task side 56, model side 56)
* adjacent arms whose minimal reversal changes the stable matching: **5** (4.5% of adjacent)
* adjacent non-critical: 107

## Critical vs non-critical adjacent-arm gap statistics

| group | n | min | median | mean | max |
|---|---|---|---|---|---|
| critical | 5 | 0.01000 | 0.07646 | 0.05953 | 0.09945 |
| non-critical | 107 | 0.00000 | 0.03505 | 0.05792 | 0.25117 |

## Pair-swap sensitivity (all arms, not minimal for non-adjacent)

* arms whose position-swap changes the matching: 75 / 448

## Critical arms (adjacent, minimal reversal)

| side | agent | partner_1 | partner_2 | gap | flip prob | bt_true_prob | rank_1 | rank_2 | changes |
|---|---|---|---|---|---|---|---|---|---|
| task | math500 | GLM-Z1-9B-0414 | DeepSeek-R1-0528-Qwen3-8B | 0.01000 | 0.222 | 0.528 | 1 | 2 | 4 |
| task | medqa | DeepSeek-R1-0528-Qwen3-8B | Qwen3-8B | 0.03010 | 0.028 | 0.585 | 1 | 2 | 2 |
| model | Qwen2.5-Coder-7B-Instruct | humaneval | mbpp | 0.07646 | 0.036 | 0.586 | 1 | 2 | 4 |
| task | humaneval | Qwen2.5-Coder-7B-Instruct | glm-4-9b-chat | 0.08163 | 0.050 | 0.716 | 1 | 2 | 4 |
| model | DeepSeek-R1-0528-Qwen3-8B | medqa | finqa | 0.09945 | 0.000 | 0.611 | 2 | 3 | 2 |
