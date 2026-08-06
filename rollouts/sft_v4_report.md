# SFT-v4 corpus report

seed=42 max_seq=4096 max_think=2000c tokenizer=v6_tokenizer_export

**52847 records** (51847 train + 1000 held-out val), 26.0M assembled tokens

math/science:  19905 (38%) — v3 was 64%
code:          6500 (12%) — v3 had ZERO
tool-calling:  3500 (7%) — new in v4
other spec.:   4500 (9%) (tables / multilingual / safety)

## By mode
- chat: 3883 (7.3%)
- off: 29405 (55.6%)
- on: 19559 (37.0%)

## By source
- gsm8k-train: 6500
- orca-math: 6000
- evol-code: 3000
- smoltalk2:smoltalk_smollm3_smol_magpie_ultra: 2500
- smoltalk2:smoltalk_systemchats_Qwen3_32B_think: 2000
- smoltalk2:tulu_3_sft_personas_instruction_following: 2000
- smoltalk2:smoltalk_smollm3_smol_rewrite: 2000
- smoltalk2:smoltalk_smollm3_smol_summarize: 2000
- smoltalk2:Mixture_of_Thoughts_science: 2000
- smoltalk2:smoltalk_smollm3_systemchats_30k: 2000
- nemotron:math: 2000
- nemotron:code: 2000
- smoltalk2:smoltalk_smollm3_explore_instruct_rewriting: 1500
- smoltalk2:xlam_traces: 1500
- smoltalk2:table_gpt: 1500
- smoltalk2:smoltalk_multilingual_8languages_lang_5: 1500
- smoltalk2:OpenThoughts3_1.2M: 1500
- nemotron:science: 1500
- nemotron:safety: 1500
- v2:openr1: 1500
- v2:chat: 1500
- smoltalk2:aya_dataset_Qwen3_32B_think: 1200
- smoltalk2:multi_turn_reasoning_if_think: 1200
- smoltalk2:hermes_function_calling_v1: 1000
- smoltalk2:smolagents_toolcalling_traces_think: 1000
- v2:mopd-verified: 405
- smoltalk2:smoltalk_smollm3_everyday_conversations: 383
- smoltalk2:smoltalk_everyday_convs_reasoning_Qwen3_32B_think: 159

## THINKING length — the v4 headline (model emits 250-440c; v3 median was ~4,500c)
- rows with thinking: 19559
- p10=135 p50=501 p90=1655 max=2000

### Histogram (250-char buckets)
-    0-250 :   5684 #################
-  250-500 :   4069 ############
-  500-750 :   2231 #######
-  750-1000:   1700 #####
- 1000-1250:   1492 #####
- 1250-1500:   1493 #####
- 1500-1750:   1489 #####
- 1750-2000:   1396 ####

## Assembled length (tokens)
- p50=385 p90=950 p99=2030 max=4043

## Drops
- contaminated (8-gram/prefix vs GSM8K-test + MATH-500): 584
- duplicate problem (cross-source hash): 18563
- **narration-rejected (ON, think < answer)**: 3097  <- the v3 defect, now auto-caught
- **think > 2000c (impossible-length trace)**: 16300
- assembled > 4096 tokens: 20
- parse-fail / too-short / no extractable answer: 4311

## Slice fill vs target
- gsm8k_train_on: 6500 / 6500
- orca_math_on: 6000 / 6000
- smol_everyday_think: 159 / 2000
- smol_systemchats_think: 2000 / 2000
- smol_aya_think: 1200 / 1200
- smol_multiturn_if_think: 1200 / 1200
- nemotron_science_short_on: 1500 / 1500
- nemotron_math_off: 2000 / 2000
- openr1_off: 1500 / 1500
- mopd_off: 405 / ALL
- smol_magpie_off: 2500 / 2500
- smol_tulu_if_off: 2000 / 2000
- smol_rewrite_off: 2000 / 2000
- smol_summarize_off: 2000 / 2000
- smol_explore_rewrite_off: 1500 / 1500
- smol_xlam_tool: 1500 / 1500
- smol_hermes_tool: 1000 / 1000
- smol_agentic_tool_on: 1000 / 1000
- evol_code_off: 3000 / 3000
- nemotron_code_off: 2000 / 2000
- nemotron_safety_off: 1500 / 1500
- smol_science_short_off: 2000 / 2000
- smol_tables_off: 1500 / 1500
- smol_multilingual_off: 1500 / 1500
- smol_openthoughts_off: 1500 / 1500
- smol_s1k_on: 0 / 400
- smol_everyday_chat: 383 / 2000
- smol_systemchats_chat: 2000 / 2000
- v2_chat: 1500 / 1500
