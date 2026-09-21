# Bootstrap Oracle Stability Report

* replicates: 500, seed: 20260918, elapsed: 1.1s

## A. Overall matching stability

* **P(H_bootstrap == H_train) = 0.114**
* distinct bootstrap stable matchings: 49

Top-5 most frequent bootstrap matchings:

| rank | matching | count | frequency | is train oracle |
|---|---|---|---|---|
| 1 | bbh=DeepSeek-R1-0528-Qwen3-8B|finqa=Fin-R1|korbench=Qwen2.5-Coder-7B-Instruct|livecodebench=Qwen3-8B|math500=NVIDIA-Nemotron-Nano-9B-v2|medqa=Llama-3.1-8B-UltraMedical|meld=gemma-2-9b-it|mmlupro=Llama-3.1-8B-Instruct | 75 | 0.150 | False |
| 2 | bbh=DeepSeek-R1-0528-Qwen3-8B|finqa=Fin-R1|korbench=Qwen2.5-Coder-7B-Instruct|livecodebench=Qwen3-8B|math500=NVIDIA-Nemotron-Nano-9B-v2|medqa=Llama-3.1-8B-Instruct|meld=gemma-2-9b-it|mmlupro=Llama-3.1-8B-UltraMedical | 75 | 0.150 | False |
| 3 | bbh=NVIDIA-Nemotron-Nano-9B-v2|finqa=Fin-R1|korbench=Qwen2.5-Coder-7B-Instruct|livecodebench=Qwen3-8B|math500=DeepSeek-R1-0528-Qwen3-8B|medqa=Llama-3.1-8B-UltraMedical|meld=gemma-2-9b-it|mmlupro=Llama-3.1-8B-Instruct | 57 | 0.114 | True |
| 4 | bbh=NVIDIA-Nemotron-Nano-9B-v2|finqa=Fin-R1|korbench=Qwen2.5-Coder-7B-Instruct|livecodebench=Qwen3-8B|math500=DeepSeek-R1-0528-Qwen3-8B|medqa=Llama-3.1-8B-Instruct|meld=gemma-2-9b-it|mmlupro=Llama-3.1-8B-UltraMedical | 38 | 0.076 | False |
| 5 | bbh=DeepSeek-R1-0528-Qwen3-8B|finqa=Fin-R1|korbench=Llama-3.1-8B-Instruct|livecodebench=Qwen3-8B|math500=NVIDIA-Nemotron-Nano-9B-v2|medqa=Llama-3.1-8B-UltraMedical|meld=Qwen2.5-Coder-7B-Instruct|mmlupro=gemma-2-9b-it | 38 | 0.076 | False |

## B/C. Per-task assignment stability

| task | most frequent model | frequency | oracle assignment | max frequency |
|---|---|---|---|---|
| bbh | DeepSeek-R1-0528-Qwen3-8B | 0.532 | NVIDIA-Nemotron-Nano-9B-v2 | 0.532 |
| finqa | Fin-R1 | 0.984 | Fin-R1 | 0.984 |
| korbench | Qwen2.5-Coder-7B-Instruct | 0.550 | Qwen2.5-Coder-7B-Instruct | 0.550 |
| livecodebench | Qwen3-8B | 0.866 | Qwen3-8B | 0.866 |
| math500 | NVIDIA-Nemotron-Nano-9B-v2 | 0.490 | DeepSeek-R1-0528-Qwen3-8B | 0.490 |
| medqa | Llama-3.1-8B-UltraMedical | 0.512 | Llama-3.1-8B-UltraMedical | 0.512 |
| meld | gemma-2-9b-it | 0.602 | gemma-2-9b-it | 0.602 |
| mmlupro | gemma-2-9b-it | 0.344 | Llama-3.1-8B-Instruct | 0.344 |

* assignments with frequency >= 0.9: 1 (of 8 tasks)
* ambiguous tasks (max assignment frequency < 0.7): ['bbh', 'korbench', 'math500', 'medqa', 'meld', 'mmlupro']

## D. Which tasks drive full-matching instability?

| task | P_b(assignment differs from H_train) |
|---|---|
| bbh | 0.678 |
| mmlupro | 0.674 |
| math500 | 0.550 |
| medqa | 0.488 |
| korbench | 0.450 |
| meld | 0.398 |
| livecodebench | 0.134 |
| finqa | 0.016 |

## E. Pairwise preference reversal rates

* median flip probability: 0.000
* arms with flip probability > 0.1: 57 / 448
* arms with flip probability > 0.25: 31 / 448
* arms with flip probability > 0.5: 3 / 448

Top-10 most fragile preference pairs:

| side | agent | partner_1 | partner_2 | base_gap | flip prob |
|---|---|---|---|---|---|
| task | medqa | Llama-3.1-8B-UltraMedical | Llama-3.1-8B-Instruct | 0.0000 | 0.974 |
| task | math500 | DeepSeek-R1-0528-Qwen3-8B | NVIDIA-Nemotron-Nano-9B-v2 | 0.0000 | 0.892 |
| task | finqa | Qwen2.5-Coder-7B-Instruct | gemma-2-9b-it | 0.0015 | 0.504 |
| task | korbench | Qwen2.5-Coder-7B-Instruct | gemma-2-9b-it | 0.0013 | 0.492 |
| task | korbench | DeepSeek-R1-0528-Qwen3-8B | NVIDIA-Nemotron-Nano-9B-v2 | 0.0013 | 0.480 |
| task | mmlupro | Qwen2.5-Coder-7B-Instruct | Llama-3.1-8B-Instruct | 0.0017 | 0.476 |
| task | finqa | DeepSeek-R1-0528-Qwen3-8B | NVIDIA-Nemotron-Nano-9B-v2 | 0.0015 | 0.462 |
| model | gemma-2-9b-it | medqa | korbench | 0.0035 | 0.448 |
| task | livecodebench | Llama-3.1-8B-Instruct | gemma-2-9b-it | 0.0032 | 0.436 |
| task | meld | Qwen3-8B | gemma-2-9b-it | 0.0041 | 0.432 |
