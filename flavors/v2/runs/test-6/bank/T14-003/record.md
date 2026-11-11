team:         T14
builds_on:    T14-002
performance:  "unofficial same-job ABA(B), 400 prompts c=16: A 949.2 (1 start), B 951.1 / 953.6 (2 starts); B +0.2..+0.5% over the single A; INCOMPLETE (4th start killed by job timeout)"
summary:
  hypothesis: the Marlin slab policy (T14-001/002) gives a measurable end-to-end gain over the exact champion T9-005.
  change:     same change.diff as T14-002 (ws leader-14.m1; policy default-on, AI_POLICY=0 = stock). A = ws leader-14.champ (= T9-005, own build, no T6 code at all); B = leader-14.m1, AI_POLICY unset. serve_args.txt = T9-005's, both arms. Job 119 (14.0 GPU-min, timed out), order A B B A, each a fresh start via lib.sh start_server_bench 400 16; log shows each arm's _C from its own workspace (job119.log "ARM A/B _C =").
  results:    Observed (tok/s / p99 TTFT ms / p99 TPOT ms / median TTFT ms / MTP acceptance):
    1A  949.2 / 1105 / 37.6 / 169.4 / 2.329
    2B  951.1 / 1148 / 29.4 / 171.7 / 2.324
    3B  953.6 / 1066 / 33.2 / 160.9 / 2.322
    4A  NOT COMPLETED: the job hit its 14-min timeout while this start was running (starts took 309 s, 178 s, 197 s incl. 15 s gsm8k on 3B; the first start paid cold compile/cache).
    Every B (951.1, 953.6) is above the only A (949.2), by +1.9 and +4.4 tok/s (+0.2%, +0.5%). With one A there is no A spread; earlier ABBAs on this stack show per-start ranges of 2-4 tok/s (T9-005: A 940.2-942.9, B 945.1-950.5) and a ~±1.3% single-trial CV per KB36 before NUMA binding, ~0.1-0.4% since.
    gsm8k 50 on 3B (policy on): 88.0% (44/50), no missing answers (BASE 88.0%).
  conclusion: Interpretation: the sign agrees with T14-002 (in-situ Marlin cp -0.6 s/trial = ~-0.4% of a ~100 s trial... i.e. an expected +0.3..0.6%) but the evidence is one A against two B within about one start-to-start spread; it does not establish a gain. A 4th start (A) would be needed to say whether every B > every A.
validity:     AI_POLICY defaults ON in the code (t6_knobs: unset/1 = measured policy, 0 = stock, "manual" = raw knobs, not for serving). M<=48 bitwise identical to the champion (T14-001: bitwise at every M<=48 on 6 shapes); M>48 differs only by fp32 reduction order, max abs 4.0 = 1 bf16 ulp at |y|~512-1024, no non-finite, no writes past row M (NaN-guard test). Tail rows masked in-kernel. T6's min_workspace_size change (sms -> 2*sms in the C++ check) only loosens a lower-bound check on the workspace tensor; the Python side allocates sms*MARLIN_MAX_BLOCKS_PER_SM int32 locks (a few KB per device, unchanged), so GPU memory is unchanged. gsm8k50 88.0% on the policy server. Nothing benchmark-specific. Leader review pending.
