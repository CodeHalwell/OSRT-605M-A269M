# SFT-v3 corpus report

seed=42 max_seq=4096 tokenizer=v6_tokenizer_export

**42215 records, 56.9M assembled tokens** (~13,882 packed rows at 4096)

## By mode
- chat: 6500 (15.4%)
- off: 12773 (30.3%)
- on: 22942 (54.3%)

## By source
- nemotron:math: 11000
- v2:openr1: 6000
- nemotron:science: 5000
- nemotron:chat: 4000
- v2:mopd-verified: 3215
- v2:chat: 3000
- smoltalk2:smoltalk_smollm3_smol_magpie_ultra: 2500
- v2:stratos: 2000
- smoltalk2:tulu_3_sft_personas_instruction_following: 2000
- smoltalk2:smoltalk_smollm3_systemchats_30k: 2000
- smoltalk2:smoltalk_smollm3_everyday_conversations: 1500

## Assembled length (tokens)
- p10=205 p25=416 p50=925 p75=2128 p90=3205 p99=4004 max=4096

### Histogram (512-token buckets)
-    0-512 :  13507 ###################
-  512-1024:   8713 ############
- 1024-1536:   4933 #######
- 1536-2048:   3956 ######
- 2048-2560:   3146 ####
- 2560-3072:   2973 ####
- 3072-3584:   2627 ####
- 3584-4096:   2355 ###

## Drops
- contaminated (8-gram/prefix vs GSM8K-test + MATH-500): 1715
- duplicate problem (cross-source hash): 16356
- assembled > 4096 tokens: 14951
- parse-fail / multi-turn / too-short: 6558

## Slice fill vs target
- anchor_mopd: 3215 / ALL
- anchor_openr1: 6000 / 6000
- anchor_stratos: 2000 / 2000
- anchor_chat: 3000 / 3000
- nemotron_math_on: 9000 / 9000
- nemotron_science_on: 5000 / 5000
- nemotron_math_off: 2000 / 2000
- nemotron_chat_off: 4000 / 4000
- smol_magpie_off: 2500 / 2500
- smol_tulu_if_off: 2000 / 2000
- smol_everyday_chat: 1500 / 1500
- smol_systemchats_chat: 2000 / 2000
