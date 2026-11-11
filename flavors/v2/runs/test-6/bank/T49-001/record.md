team:         T49
builds_on:    BASE (analysis only; uses T38-003/005 measured PP-stage costs and T45-002's calibrated TP event model)
performance:  "CPU only, n/a (closed-loop schedule simulation)"
summary:
  hypothesis: T38 closed PP because two decode microbatches read every weight twice and prefill was charged as extra time.
              A 2-stage pipeline with ONE decode microbatch (all 16 requests, every weight read shared by all rows, no
              per-layer all-reduce so ~128 rows/pass are cheap -> draft trees) leaves each GPU idle ~half of each period;
              that bubble is about the size of the prefill demand (~70 prompt tokens per period), so prefill (28% of the
              TP trial) might become free.
  change:     none. bank/T49-001/ppbubble.py (also agents/leader-49/sim/): 400 requests, c=16 closed loop, lognormal
              ShareGPT-like lengths (as T45-002); prefill chunks FIFO, stage0 then stage1 next phase, a request decodes
              only after its last chunk clears stage1. Stage GEMM vs rows from T38-005 (13.3 ms @<=64 .. 15.24 @128 ..
              18.9 @160), non-GEMM 3-4 ms/stage, full lm_head 2.1 ms on GPU1, draft level 0.75-0.95 ms.
  results:    Observed (simulation): TP event model 80.5 s (official T37-002 79.7 s); TP with prefill free 57.9 s.
              PP-bubble: chain k=3 -10..-21%; 6-node trees -12..+5%; 8-node trees (128 rows, acceptance 3.85 ASSUMED)
              +2..+10%; 11-node trees (176 rows, 4.21 measured offline by T38) -14..+5%. Best case utilisation GPU0 63%,
              GPU1 79%. Decode-only the schedule is ~0.84x TP even with trees.
  conclusion: Interpretation: the bubble exists and has the right average size, but the closed loop wastes much of it:
              each prompt needs >=2 phases (~45 ms each) plus chunk queueing before its slot decodes, and prompts arrive
              in lumps. Ceiling ~+10% under favourable, unbuilt kernel costs, for a ~100+ GPU-min build (PP engine, tree
              GDN/attention/drafting). PP stays closed, but the reason is closed-loop latency coupling, not bubble size or
              double weight reads. Untested inputs: 8-node tree acceptance (T38 tooling, CPU), per-stage non-GEMM at 16 seqs.
validity:     n/a (no engine change)
