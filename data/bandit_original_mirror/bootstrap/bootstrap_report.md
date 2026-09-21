# Bootstrap Oracle Stability Report

* replicates: 500, seed: 20260918, elapsed: 1.1s

## A. Overall matching stability

* **P(H_bootstrap == H_train) = 0.090**
* distinct bootstrap stable matchings: 40

Top-5 most frequent bootstrap matchings:

| rank | matching | count | frequency | is train oracle |
|---|---|---|---|---|
| 1 | bbh=NVIDIA-Nemotron-Nano-9B-v2|finqa=Fin-R1|korbench=Llama-3.1-8B-Instruct|livecodebench=Llama-3.1-8B-UltraMedical|math500=DeepSeek-R1-0528-Qwen3-8B|medqa=Qwen3-8B|meld=Qwen2.5-Coder-7B-Instruct|mmlupro=gemma-2-9b-it | 151 | 0.302 | False |
| 2 | bbh=DeepSeek-R1-0528-Qwen3-8B|finqa=Fin-R1|korbench=Llama-3.1-8B-Instruct|livecodebench=Llama-3.1-8B-UltraMedical|math500=NVIDIA-Nemotron-Nano-9B-v2|medqa=Qwen3-8B|meld=Qwen2.5-Coder-7B-Instruct|mmlupro=gemma-2-9b-it | 104 | 0.208 | False |
| 3 | bbh=NVIDIA-Nemotron-Nano-9B-v2|finqa=Fin-R1|korbench=Qwen2.5-Coder-7B-Instruct|livecodebench=Llama-3.1-8B-UltraMedical|math500=DeepSeek-R1-0528-Qwen3-8B|medqa=Qwen3-8B|meld=gemma-2-9b-it|mmlupro=Llama-3.1-8B-Instruct | 45 | 0.090 | True |
| 4 | bbh=NVIDIA-Nemotron-Nano-9B-v2|finqa=Fin-R1|korbench=Llama-3.1-8B-Instruct|livecodebench=Llama-3.1-8B-UltraMedical|math500=DeepSeek-R1-0528-Qwen3-8B|medqa=Qwen3-8B|meld=gemma-2-9b-it|mmlupro=Qwen2.5-Coder-7B-Instruct | 39 | 0.078 | False |
| 5 | bbh=DeepSeek-R1-0528-Qwen3-8B|finqa=Fin-R1|korbench=Qwen2.5-Coder-7B-Instruct|livecodebench=Llama-3.1-8B-UltraMedical|math500=NVIDIA-Nemotron-Nano-9B-v2|medqa=Qwen3-8B|meld=gemma-2-9b-it|mmlupro=Llama-3.1-8B-Instruct | 33 | 0.066 | False |

## B/C. Per-task assignment stability

| task | most frequent model | frequency | oracle assignment | max frequency |
|---|---|---|---|---|
| bbh | NVIDIA-Nemotron-Nano-9B-v2 | 0.528 | NVIDIA-Nemotron-Nano-9B-v2 | 0.528 |
| finqa | Fin-R1 | 0.946 | Fin-R1 | 0.946 |
| korbench | Llama-3.1-8B-Instruct | 0.708 | Qwen2.5-Coder-7B-Instruct | 0.708 |
| livecodebench | Llama-3.1-8B-UltraMedical | 0.914 | Llama-3.1-8B-UltraMedical | 0.914 |
| math500 | DeepSeek-R1-0528-Qwen3-8B | 0.524 | DeepSeek-R1-0528-Qwen3-8B | 0.524 |
| medqa | Qwen3-8B | 0.862 | Qwen3-8B | 0.862 |
| meld | Qwen2.5-Coder-7B-Instruct | 0.632 | gemma-2-9b-it | 0.632 |
| mmlupro | gemma-2-9b-it | 0.638 | Llama-3.1-8B-Instruct | 0.638 |

* assignments with frequency >= 0.9: 2 (of 8 tasks)
* ambiguous tasks (max assignment frequency < 0.7): ['bbh', 'math500', 'meld', 'mmlupro']

## D. Which tasks drive full-matching instability?

| task | P_b(assignment differs from H_train) |
|---|---|
| mmlupro | 0.820 |
| korbench | 0.794 |
| meld | 0.646 |
| math500 | 0.476 |
| bbh | 0.472 |
| medqa | 0.138 |
| livecodebench | 0.086 |
| finqa | 0.054 |

## E. Pairwise preference reversal rates

* median flip probability: 0.000
* arms with flip probability > 0.1: 54 / 448
* arms with flip probability > 0.25: 28 / 448
* arms with flip probability > 0.5: 3 / 448

Top-10 most fragile preference pairs:

| side | agent | partner_1 | partner_2 | base_gap | flip prob |
|---|---|---|---|---|---|
| task | medqa | Llama-3.1-8B-UltraMedical | Llama-3.1-8B-Instruct | 0.0000 | 0.974 |
| task | math500 | DeepSeek-R1-0528-Qwen3-8B | NVIDIA-Nemotron-Nano-9B-v2 | 0.0000 | 0.892 |
| task | finqa | Qwen2.5-Coder-7B-Instruct | gemma-2-9b-it | 0.0015 | 0.504 |
| model | gemma-2-9b-it | mmlupro | meld | 0.0017 | 0.500 |
| task | korbench | Qwen2.5-Coder-7B-Instruct | gemma-2-9b-it | 0.0013 | 0.492 |
| task | korbench | DeepSeek-R1-0528-Qwen3-8B | NVIDIA-Nemotron-Nano-9B-v2 | 0.0013 | 0.480 |
| task | mmlupro | Qwen2.5-Coder-7B-Instruct | Llama-3.1-8B-Instruct | 0.0017 | 0.476 |
| task | finqa | DeepSeek-R1-0528-Qwen3-8B | NVIDIA-Nemotron-Nano-9B-v2 | 0.0015 | 0.462 |
| model | DeepSeek-R1-0528-Qwen3-8B | mmlupro | finqa | 0.0036 | 0.438 |
| task | livecodebench | Llama-3.1-8B-Instruct | gemma-2-9b-it | 0.0032 | 0.436 |
