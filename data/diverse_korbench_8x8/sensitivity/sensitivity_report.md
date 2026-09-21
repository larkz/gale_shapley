# Preference-Arm Sensitivity Report

* total arms: 448
* adjacent-in-oracle arms: 112 (task side 56, model side 56)
* adjacent arms whose minimal reversal changes the stable matching: **6** (5.4% of adjacent)
* adjacent non-critical: 106

## Critical vs non-critical adjacent-arm gap statistics

| group | n | min | median | mean | max |
|---|---|---|---|---|---|
| critical | 6 | 0.01000 | 0.04425 | 0.05344 | 0.11533 |
| non-critical | 106 | 0.00000 | 0.02456 | 0.04766 | 0.26207 |

## Pair-swap sensitivity (all arms, not minimal for non-adjacent)

* arms whose position-swap changes the matching: 82 / 448

## Critical arms (adjacent, minimal reversal)

| side | agent | partner_1 | partner_2 | gap | flip prob | bt_true_prob | rank_1 | rank_2 | changes |
|---|---|---|---|---|---|---|---|---|---|
| task | math500 | GLM-Z1-9B-0414 | DeepSeek-R1-0528-Qwen3-8B | 0.01000 | 0.222 | 0.528 | 1 | 2 | 3 |
| model | glm-4-9b-chat | meld | emorynlp | 0.02515 | 0.112 | 0.544 | 3 | 4 | 2 |
| task | medqa | DeepSeek-R1-0528-Qwen3-8B | Qwen3-8B | 0.03010 | 0.028 | 0.585 | 1 | 2 | 2 |
| model | Qwen2.5-Coder-7B-Instruct | humaneval | mbpp | 0.05839 | 0.060 | 0.601 | 1 | 2 | 4 |
| task | humaneval | Qwen2.5-Coder-7B-Instruct | glm-4-9b-chat | 0.08163 | 0.050 | 0.716 | 1 | 2 | 4 |
| model | Intern-S1-mini | korbench | finqa | 0.11533 | 0.000 | 0.692 | 2 | 3 | 2 |
