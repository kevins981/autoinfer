[manager, manager of run test-4] Round 1 task for leader-2 (team T2)

## Direction: R1-B, the vocab-parallel head diet, then re-tune draft depth
Cut the per-decode-step cost of the lm_heads and the full-vocab logits all-gathers, then
re-measure num_speculative_tokens with the cheaper drafter.
 (a) existing spec-config flags: "use_local_argmax_reduction": true and
     "draft_lm_head_quantization": "fp8". Check that they're supported on this MTP path (V2 model
     runner) and compose, and that the FP8 head doesn't fall back to a slow kernel on sm_89.
 (b) greedy target verification without the logits all-gather: when every request in the batch
     is greedy and needs no logprobs/penalties/bias/structured output, take a per-rank argmax with
     an exact (value, index) reduction and the lowest index on ties, so the result matches the full
     argmax. Otherwise fall back to the full path. The condition must be generic, never keyed to
     the benchmark.
 (c) with (a)+(b) in place: num_speculative_tokens 3 (maybe 4) vs 2. This changes memory, so it
     needs the 128K test and a run under benchmark load. TTFT/TPOT p99 gates apply.

## Rationale (both brainstormers, measured: brainstorm/r1/systematic_engineer.md #1/#3, adjacent_field.md #2/#3)
- Per decode step (~35-40 ms): drafter lm_head 2 x 1.37 ms (bf16 1.27 GB/GPU, at DRAM roofline),
  target lm_head 1.59 ms, logits all-gathers 0.96 ms (target) + 2 x 0.33 ms (drafter).
  Together ~10% of a decode step; decode steps are ~64% of wall.
- Estimates (assumed): flags ~+4%, target argmax ~+2-3%, depth 3 +4-10% (position-3 acceptance
  unknown; BASE is 0.78/0.56 at positions 1/2).
- Allowed explicitly by task_definition.md: an FP8 draft lm_head inside the drafter only;
  verification stays full-vocab, exact, bf16 target. Exactness of (b) matters: test it
  numerically, including ties (bf16 logits do tie).

## Budget: 55 GPU-minutes for T2 this round (gpu/budget.json); keep ~11 per official eval.
The flag A/B in (a) is cheap. Batch tests per server start.

## Champion to beat: BASE, 635.9 tok/s official (bank/BASE/official/1/results.json).
TTFT p99 1341, TPOT p99 46.1 ms, GSM8K 94.4%, MTP acceptance 2.345.

## Scope notes
- Read task_definition.md (global).
- Team T1 is replacing the all-reduce backend (R1-A) in parallel. Don't touch the communication
  layer.
- Report to manager when done: records, findings, proposed SHARED_KB entries, what's left open,
  and your single "next" for round 2.
