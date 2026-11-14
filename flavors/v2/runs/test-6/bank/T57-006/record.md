team:         T57
builds_on:    T57-005 (T57-004 + HOSTEMB) + T58-001's default (diff includes it)
performance:  "screen null: ABBAAB job 510 +0.01% (mixed); step probe -0.1 ms/step"
summary:
  hypothesis: drafter v2 plus target-side glue (with HOSTEMB) removes ~0.1 ms/step, worth ~+0.4%.
  change:     T57_HOSTEMB (T57-005); T57_FLUSH1: new CUDA op gdn_replay_flush_multi, 1 launch per site instead of 3 per-group gdn_replay_flush; T57_RADIX: 4-pass radix top-64 select replacing the 32-step bisection in _chunk_topk/_merge_exact (draft screen). All default ON. Diff also carries T58-001's VLLM_HOSTAR_L2PF_MSG_MAX default.
  results:    Observed: job 507 radix select bitwise equal to the bisection (random/ties/const/negative); _chunk_topk 12.3 -> 7.7 us, _merge_exact 17.6 -> 15.5 us. Job 508 probe (200 prompts): decode median GPU span 24.4 vs 24.5 ms; greedy 5 prompts identical; acceptance 2.708 vs 2.671.
              ABBAAB job 510 (400 prompts, A = three flags off = T57-004 behaviour, B = on): B 1231.5/1224.4/1226.3, A 1225.8/1225.3/1230.6 tok/s: +0.01%, mixed; acceptance 2.665-2.674 both arms.
              Job 513 flush bitwise test (task7/flush_test.py): 16 cases (modes 0/1 incl. page-boundary crossing, red8 on/off, two-slot on/off, 2 seeds): state, flags and counters identical; negative control differs. (job 511's mismatch was a slot race in the test itself.)
  conclusion: Interpretation: ~0.1 ms/step (0.4% of decode) is below what a 6-start e2e A/B resolves here; no official eval. The exact parts (HOSTEMB frees 1.17 GiB/GPU, FLUSH1, RADIX) are available to stack with a future memory-blocked change.
validity:     FLUSH1 bitwise on the target state (job 513); RADIX bitwise on the drafter screen; HOSTEMB bitwise (T57-005). Not submitted.
