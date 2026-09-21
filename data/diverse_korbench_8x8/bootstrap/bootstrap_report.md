# Bootstrap Oracle Stability Report

* replicates: 500, seed: 20260918, elapsed: 1.0s

## A. Overall matching stability

* **P(H_bootstrap == H_train) = 0.538**
* distinct bootstrap stable matchings: 30

Top-5 most frequent bootstrap matchings:

| rank | matching | count | frequency | is train oracle |
|---|---|---|---|---|
| 1 | emorynlp=gemma-2-9b-it|finqa=Qwen3-8B|humaneval=Qwen2.5-Coder-7B-Instruct|korbench=Intern-S1-mini|math500=GLM-Z1-9B-0414|mbpp=Fin-R1|medqa=DeepSeek-R1-0528-Qwen3-8B|meld=glm-4-9b-chat | 269 | 0.538 | True |
| 2 | emorynlp=gemma-2-9b-it|finqa=GLM-Z1-9B-0414|humaneval=Qwen2.5-Coder-7B-Instruct|korbench=Intern-S1-mini|math500=DeepSeek-R1-0528-Qwen3-8B|mbpp=Fin-R1|medqa=Qwen3-8B|meld=glm-4-9b-chat | 65 | 0.130 | False |
| 3 | emorynlp=glm-4-9b-chat|finqa=Qwen3-8B|humaneval=Qwen2.5-Coder-7B-Instruct|korbench=Intern-S1-mini|math500=GLM-Z1-9B-0414|mbpp=Fin-R1|medqa=DeepSeek-R1-0528-Qwen3-8B|meld=gemma-2-9b-it | 56 | 0.112 | False |
| 4 | emorynlp=glm-4-9b-chat|finqa=GLM-Z1-9B-0414|humaneval=Qwen2.5-Coder-7B-Instruct|korbench=Intern-S1-mini|math500=DeepSeek-R1-0528-Qwen3-8B|mbpp=Fin-R1|medqa=Qwen3-8B|meld=gemma-2-9b-it | 23 | 0.046 | False |
| 5 | emorynlp=gemma-2-9b-it|finqa=GLM-Z1-9B-0414|humaneval=Qwen2.5-Coder-7B-Instruct|korbench=Intern-S1-mini|math500=Qwen3-8B|mbpp=Fin-R1|medqa=DeepSeek-R1-0528-Qwen3-8B|meld=glm-4-9b-chat | 17 | 0.034 | False |

## B/C. Per-task assignment stability

| task | most frequent model | frequency | oracle assignment | max frequency |
|---|---|---|---|---|
| emorynlp | gemma-2-9b-it | 0.784 | gemma-2-9b-it | 0.784 |
| finqa | Qwen3-8B | 0.722 | Qwen3-8B | 0.722 |
| humaneval | Qwen2.5-Coder-7B-Instruct | 0.910 | Qwen2.5-Coder-7B-Instruct | 0.910 |
| korbench | Intern-S1-mini | 0.966 | Intern-S1-mini | 0.966 |
| math500 | GLM-Z1-9B-0414 | 0.746 | GLM-Z1-9B-0414 | 0.746 |
| mbpp | Fin-R1 | 0.910 | Fin-R1 | 0.910 |
| medqa | DeepSeek-R1-0528-Qwen3-8B | 0.776 | DeepSeek-R1-0528-Qwen3-8B | 0.776 |
| meld | glm-4-9b-chat | 0.764 | glm-4-9b-chat | 0.764 |

* assignments with frequency >= 0.9: 3 (of 8 tasks)
* ambiguous tasks (max assignment frequency < 0.7): none

## D. Which tasks drive full-matching instability?

| task | P_b(assignment differs from H_train) |
|---|---|
| finqa | 0.278 |
| math500 | 0.254 |
| meld | 0.236 |
| medqa | 0.224 |
| emorynlp | 0.216 |
| humaneval | 0.090 |
| mbpp | 0.090 |
| korbench | 0.034 |

## E. Pairwise preference reversal rates

* median flip probability: 0.000
* arms with flip probability > 0.1: 98 / 448
* arms with flip probability > 0.25: 54 / 448
* arms with flip probability > 0.5: 5 / 448

Top-10 most fragile preference pairs:

| side | agent | partner_1 | partner_2 | base_gap | flip prob |
|---|---|---|---|---|---|
| task | emorynlp | DeepSeek-R1-0528-Qwen3-8B | glm-4-9b-chat | 0.0000 | 0.972 |
| task | emorynlp | Intern-S1-mini | Qwen2.5-Coder-7B-Instruct | 0.0024 | 0.536 |
| model | Qwen3-8B | emorynlp | meld | 0.0001 | 0.528 |
| task | emorynlp | gemma-2-9b-it | glm-4-9b-chat | 0.0024 | 0.516 |
| task | finqa | Qwen2.5-Coder-7B-Instruct | gemma-2-9b-it | 0.0015 | 0.504 |
| task | korbench | Qwen2.5-Coder-7B-Instruct | gemma-2-9b-it | 0.0013 | 0.492 |
| model | Intern-S1-mini | emorynlp | meld | 0.0042 | 0.478 |
| task | emorynlp | DeepSeek-R1-0528-Qwen3-8B | gemma-2-9b-it | 0.0024 | 0.462 |
| model | Qwen2.5-Coder-7B-Instruct | korbench | math500 | 0.0029 | 0.446 |
| model | glm-4-9b-chat | mbpp | meld | 0.0030 | 0.440 |
