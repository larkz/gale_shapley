# Variant decision: kandk -> korbench

The greedy selection (scripts/select_diverse_market.py) picked kandk
for Intern-S1-mini (full-pool margin 0.0243). Preflight on the TRAIN
split revealed an EXACT utility tie (Intern-S1-mini == Qwen3-8B,
gap 0.000000) that is matching-critical — the same structural blocker
as the original market (2 exact ties, P_bootstrap = 0.114).

Variant preflight (500 bootstrap replicates, sensitivity):

| market | P(H_b == H_train) | distinct matchings | ambiguous tasks | critical arms | min critical gap |
|---|---|---|---|---|---|
| original 8x8 | 0.114 | 49 | 6/8 | 7 | 0.000 (2 exact ties) |
| diverse (kandk) | 0.284 | 33 | 2/8 (finqa, kandk) | 7 | 0.000 (exact tie) |
| **diverse (korbench)** | **0.538** | 30 | **0/8** | **6** | **0.010 (no tie)** |

korbench per-task assignment stability: korbench 0.966, humaneval 0.910,
mbpp 0.910, medqa 0.776, emorynlp 0.784, math500 0.746, meld 0.764,
finqa 0.722. The remaining fragile pair is math500 (GLM-Z1 vs DS-R1,
TRAIN gap 0.010 -> BT p = 0.528, bootstrap flip prob 0.222) — it is
matching-critical, so exact certification still requires resolving it
(~1,500+ samples/arm; ~750K total budget), but unlike the original
market there is no arm with p == 0.5 exactly.

Selected: the korbench variant (this directory's config:
configs/llm_matching_8x8_diverse_korbench.yaml).
