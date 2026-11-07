# Round 2 task for leader-1 (team T1)

## Direction: R2-A, the "logits diet" (on top of the champion T1-005)
Cut the lm_head and logits traffic that never changes the output. In order:
1. Flags, in the speculative config:
   - `"use_local_argmax_reduction": true`: removes the 2 x ~0.36 ms draft-logits all-gathers.
   - `"draft_lm_head_quantization": "fp8"`: halves the 2 x 1.37 ms draft heads. Allowed by human
     ruling 1, inside the drafter only. Watch the acceptance length (2.34 now).
2. Code: a greedy fast path for the target's verification, saving ~0.95 ms/step. When every
   request in the batch is greedy, with no logprobs, penalties or logit processors, gather a
   (max, argmax) per row per rank instead of the full logits. Fall back to the full path otherwise.
   - Tie-break must equal torch.argmax over the full row: the lowest global id wins.
   - Prove the output is identical: compare greedy outputs on a set of prompts. Use a non-benchmark
     prompt set, or a mix, not a check tuned to the 400.
3. If 1-2 work and budget is left:
   - re-check num_speculative_tokens=3 (a memory change: needs the 128K test);
   - optionally a reduced-vocab FP8 draft head (ruling 2: ids/frequencies from non-benchmark text
     only).

### Rationale
Measured on the T1-005 nsys:
- a 38.5 ms decode step spends 6.0 ms in lm_heads + logits all-gathers;
- ~3 ms of that is removable without changing outputs.

Both round-2 brainstormers put it top-2, est. +5-6%. Details: brainstorm/r2/systematic_engineer.md
(Idea 1) and brainstorm/r2/adjacent_field.md (Idea 2), with code pointers:
- vllm/config/speculative.py:534,540;
- qwen3_5_mtp.py:310 (get_top_tokens);
- logits_processor.py:~225;
- spec_decode/speculator.py:181.

Note: the target logits all-gather may now go through your host-staged all_gather (T1-004).

## Budget
T1: 80 GPU-minutes this round. Up to 2 workers. Keep ~15 for official evals.

## Reference to beat
CHAMPION = T1-005: 699.2 tok/s at c=16 (TTFT p99 1309 ms, TPOT p99 50.6 ms, GSM8K 93.2%,
acceptance 2.34). Gates vs BASE: TTFT p99 <= 1784 ms, TPOT p99 <= 58.1 ms, GSM8K >= 89.6%. A win
must beat 699.2 by more than ~3%.

## Notes
- New KB8 in shared_kb.md: GPU 0 is clock-locked at 2235 MHz, ~15% slower on compute-bound work.
  That is the prefill skew; don't chase it.
- Other round-2 ideas (GDN state, L2 prefetch, Marlin) are in tree.md: not this round. Mention
  anything you notice in your report.
