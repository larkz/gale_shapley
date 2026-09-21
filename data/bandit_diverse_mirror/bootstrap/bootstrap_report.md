# Bootstrap Oracle Stability Report

* replicates: 500, seed: 20260918, elapsed: 1.1s

## A. Overall matching stability

* **P(H_bootstrap == H_train) = 0.574**
* distinct bootstrap stable matchings: 31

Top-5 most frequent bootstrap matchings:

| rank | matching | count | frequency | is train oracle |
|---|---|---|---|---|
| 1 | emorynlp=gemma-2-9b-it|finqa=Qwen3-8B|humaneval=Qwen2.5-Coder-7B-Instruct|korbench=Intern-S1-mini|math500=GLM-Z1-9B-0414|mbpp=Fin-R1|medqa=DeepSeek-R1-0528-Qwen3-8B|meld=glm-4-9b-chat | 287 | 0.574 | True |
| 2 | emorynlp=gemma-2-9b-it|finqa=Intern-S1-mini|humaneval=Qwen2.5-Coder-7B-Instruct|korbench=GLM-Z1-9B-0414|math500=DeepSeek-R1-0528-Qwen3-8B|mbpp=Fin-R1|medqa=Qwen3-8B|meld=glm-4-9b-chat | 60 | 0.120 | False |
| 3 | emorynlp=glm-4-9b-chat|finqa=Qwen3-8B|humaneval=Qwen2.5-Coder-7B-Instruct|korbench=Intern-S1-mini|math500=GLM-Z1-9B-0414|mbpp=Fin-R1|medqa=DeepSeek-R1-0528-Qwen3-8B|meld=gemma-2-9b-it | 47 | 0.094 | False |
| 4 | emorynlp=gemma-2-9b-it|finqa=Intern-S1-mini|humaneval=Qwen2.5-Coder-7B-Instruct|korbench=GLM-Z1-9B-0414|math500=Qwen3-8B|mbpp=Fin-R1|medqa=DeepSeek-R1-0528-Qwen3-8B|meld=glm-4-9b-chat | 15 | 0.030 | False |
| 5 | emorynlp=Fin-R1|finqa=Qwen3-8B|humaneval=glm-4-9b-chat|korbench=Intern-S1-mini|math500=GLM-Z1-9B-0414|mbpp=Qwen2.5-Coder-7B-Instruct|medqa=DeepSeek-R1-0528-Qwen3-8B|meld=gemma-2-9b-it | 15 | 0.030 | False |

## B/C. Per-task assignment stability

| task | most frequent model | frequency | oracle assignment | max frequency |
|---|---|---|---|---|
| emorynlp | gemma-2-9b-it | 0.808 | gemma-2-9b-it | 0.808 |
| finqa | Qwen3-8B | 0.738 | Qwen3-8B | 0.738 |
| humaneval | Qwen2.5-Coder-7B-Instruct | 0.922 | Qwen2.5-Coder-7B-Instruct | 0.922 |
| korbench | Intern-S1-mini | 0.786 | Intern-S1-mini | 0.786 |
| math500 | GLM-Z1-9B-0414 | 0.744 | GLM-Z1-9B-0414 | 0.744 |
| mbpp | Fin-R1 | 0.896 | Fin-R1 | 0.896 |
| medqa | DeepSeek-R1-0528-Qwen3-8B | 0.782 | DeepSeek-R1-0528-Qwen3-8B | 0.782 |
| meld | glm-4-9b-chat | 0.806 | glm-4-9b-chat | 0.806 |

* assignments with frequency >= 0.9: 1 (of 8 tasks)
* ambiguous tasks (max assignment frequency < 0.7): none

## D. Which tasks drive full-matching instability?

| task | P_b(assignment differs from H_train) |
|---|---|
| finqa | 0.262 |
| math500 | 0.256 |
| medqa | 0.218 |
| korbench | 0.214 |
| meld | 0.194 |
| emorynlp | 0.192 |
| mbpp | 0.104 |
| humaneval | 0.078 |

## E. Pairwise preference reversal rates

* median flip probability: 0.000
* arms with flip probability > 0.1: 92 / 448
* arms with flip probability > 0.25: 52 / 448
* arms with flip probability > 0.5: 5 / 448

Top-10 most fragile preference pairs:

| side | agent | partner_1 | partner_2 | base_gap | flip prob |
|---|---|---|---|---|---|
| task | emorynlp | DeepSeek-R1-0528-Qwen3-8B | glm-4-9b-chat | 0.0000 | 0.972 |
| task | emorynlp | Intern-S1-mini | Qwen2.5-Coder-7B-Instruct | 0.0024 | 0.536 |
| model | Intern-S1-mini | humaneval | mbpp | 0.0038 | 0.532 |
| task | emorynlp | gemma-2-9b-it | glm-4-9b-chat | 0.0024 | 0.516 |
| task | finqa | Qwen2.5-Coder-7B-Instruct | gemma-2-9b-it | 0.0015 | 0.504 |
| task | korbench | Qwen2.5-Coder-7B-Instruct | gemma-2-9b-it | 0.0013 | 0.492 |
| model | gemma-2-9b-it | humaneval | mbpp | 0.0008 | 0.490 |
| model | Fin-R1 | finqa | mbpp | 0.0001 | 0.488 |
| task | emorynlp | DeepSeek-R1-0528-Qwen3-8B | gemma-2-9b-it | 0.0024 | 0.462 |
| task | emorynlp | Qwen3-8B | gemma-2-9b-it | 0.0048 | 0.436 |
