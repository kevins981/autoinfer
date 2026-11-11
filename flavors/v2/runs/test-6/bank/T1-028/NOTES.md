# leader-2.w17 R9 mixed-step leftovers (ws leader-2.mixleft = T1-026 + this)
Change: (a) csrc gdn_replay_flush_kernel modes 0/1/3: grid (reqs, HV, 1), loop over layers (was gridZ=layers); mode 2 unchanged.
(b) replay.py _GroupClear + qwen_gdn_linear_attn._gdn_replay_flush_layer: mode-3 clear once per KV group per forward (first GDN layer
of the group, all group layers), dedup on (forward_context, attn_metadata) identity; int32 cast behind dedup.
Profile basis (T1-023 B sqlite): flush calls/GPU: mode3 ~2400 (48/mixed step, 1.5us), mode0/1 ~3300 (6/step every step, ~7.5us @16 reqs).
Item 2: replay kernel decode node trace (agents/leader-2.w7/prof_stack/node) 40.7us @16 reqs vs 46.7 @15 mixed; roof ~50us -> no cheap fix.
Unit: job 104 unit_test_layers.py 0/900 (neg 28); job 103 unit_test_mixed.py 0 (neg 6860).
ABBA: abba_job.sh -> job_abba/{1A,2B,3B,4A}
ABBA job 105 (12.6 GPU-min): tok/s 1A 838.2, 2B 844.6, 3B 836.3, 4A 839.2 -> A 838.7, B 840.4 (+0.2%, within noise; B spread 8.3).
TPOT p50 A 16.64/16.58, B 16.81/16.54; acceptance 2.345/2.346 vs 2.345/2.34; gsm8k50 (2B) 88%.
