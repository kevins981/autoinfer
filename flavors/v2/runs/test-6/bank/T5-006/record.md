team:         T5
builds_on:    T5-005 (t5-rb two-slot) + T6-004 (Marlin per-shape policy) ; T3-005..T3-008 (budgeted-verification design and offline data)
performance:  unofficial: budgeted K_MAX=3 (B=2/req, M=48, real draft confidence) 825.7 vs k=2 867.2 tok/s (-4.8%, single runs, same job);
              timer model predicts +1.7%.
summary:
  hypothesis: confidence-budgeted verification (draft up to 3, verify the batch's best 2/req by cumulative draft confidence, M=48)
              gains ~+4% decode-step (sim, agents/leader-5.w1/sim_m64.txt; T3-008 sim 1.046); T6's policy shrinks the M=64 cliff in situ.
  change:     t5-rb6 = t5-rb + T6-004 marlin.cu/marlin_template.h. P2: fused greedy_argmax_with_sumexp (draft confidence from the sharded
              98K draft vocab, (max, idx, sumexp) in the existing all-gather; 4 us/call in graph vs 16 us old path; max abs err 6e-7).
              Upstream adaptive_verification enabled for MTP + local argmax + GDN replay + two-slot + TRITON_ATTN with a fixed budget
              (AUTOINFER_BUDGET_PER_REQ), test-only forced patterns and budget log; rejection-sampler greedy-ids exclusion lifted under
              a fixed budget; cost profiling skipped. Notes: agents/leader-5.w1/budget_design.md, p23/, ragged/, t8/.
  results:    Observed:
              - Ragged exactness (jobs 48, 51, 53, 55 run0): 4 concurrent greedy requests with forced per-request counts (32 n_i patterns,
                M=12): independent per-row check (argmax of that forward's logits vs emitted tokens, drafts fed at row+1, device
                cu_num_logits) 1853 rows, 0 mismatches -- on the GENERIC rejection path (147 steps). The greedy-ids fast path was not
                exercised (client used logprobs, then min_tokens/ignore_eos): UNTESTED. Divergences vs c=1 reference occur at bf16 ties
                at the same rate as the champion's own c=4-vs-c=1 control (3/4 requests, gap 0.0 ties).
              - Job 55 (11.6 GPU-min), 100 prompts c=16 each, single runs, medians at 16 req (ms):
                | run    | M  | fwd   | samp | prop | gap  | step  | accept | tok/s | TTFT mean | ITL mean/median |
                | k2_on  | 48 | 22.36 | 1.12 | 1.68 | 0.09 | 25.25 | 2.357  | 867.2 | 298 | 36.7/25.3 |
                | bud3   | 48 | 22.48 | 1.12 | 2.48 | 0.13 | 26.21 | 2.489  | 825.7 | 349 | 40.2/26.2 |
                | k3_on  | 64 | 25.75 | 1.15 | 2.51 | 0.10 | 29.51 | 2.691  | 853.8 | 338 | 42.3/29.5 |
                | k3_off | 64 | 26.05 | 1.15 | 2.53 | 0.10 | 29.83 | 2.729  | 863.8 | 330 | 42.6/29.8 |
                bud3 admitted 2.0 slots/req; n_i 0/1/2/3 = 9.8/25.0/20.5/44.6%; per-position accepted .725/.461/.303.
  conclusion: Interpretation:
              - In-situ, T6-004's policy takes only 0.30 ms off the M=64 forward; the M=48->64 cliff stays +3.4 ms. Plain k=3 and any
                M=64 budget remain negative (sim at +3.0 ms cliff: K_MAX=4 B=48 1.020).
              - Budgeted K_MAX=3 at M=48 raises E only +5.6% (2.357 -> 2.489; offline sim +8.1%) for +3.8% step (the 3rd draft step is
                +0.80 ms) => +1.7% by the timer model. Measured -4.8%: decode median ITL +3.6% but mean ITL +9.5% and TTFT +17%, so
                the non-decode (mixed/prefill) steps get slower under the adaptive path (not isolated: varlen/mixed graph handling or
                host bookkeeping suspected).
              - Even with that artifact fixed, the ceiling is ~+1.7% decode (~+1.3% e2e): far below the 5% bar. The direction's premise
                (more accepted tokens per weight pass at affordable cost) fails on this stack: the drafter step (0.8 ms, at its byte roof)
                buys too little E, and M>48 costs +3.4 ms of Marlin in situ.
validity:     test-only env switches; two-slot bitwise (T5-005); policy is T6-004's (<=1 ulp reorder). Budget path exact on the generic
              sampler by the per-row check; greedy-ids ragged path untested. Not an official candidate.
