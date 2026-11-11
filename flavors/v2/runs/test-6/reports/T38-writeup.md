# T38 write-up (leader-38, moonshot). Closing: three directions measured, none clears +5%

Records: bank/T38-001..005. Proposals: T38-1, T38-2, T38-3. Notes: local_kb/leader-38.md. Worker: leader-38.w1.
GPU used: see `gpu history T38` (jobs 382-393). Sparring: T38-1, T38-2 (sparring/).

## Directions and why each closed
1. **T38-1, all-reduce streamed into the consumer GEMM (K-chunk flags).** Closed before building. T4-001's measured
   ceiling (5-8 us per AR; the AR slows ~2x beside a GEMM) caps it far below +5% (T38-001).
2. **T38-2, self-distilled recursive MTP head.** Closed: T3-001..004 already measured flat results (T38-001 addendum).
3. **T38-3, pipeline-parallel engine (no all-reduce) + dynamic draft trees.** Funded; stages 1-2 measured; closed at stage 2.
   - Tree acceptance is NOT the limit (T38-002/003, exact offline, T3's data and MTP):
     - best-first trees: N=11 4.21, N=19 4.74 tokens/step;
     - level-wise K=4: N=15 D=6 4.09, D=8 4.28;
     - today's chain k=3: 2.73.
   - PP prefill is ~0.51x TP's per prompt token (T38-004, job 392: 105 us of GPU time per GPU vs 205 us of wall), if the
     pipeline stays full (stock vLLM PP: 12% bubble).
   - PP decode can't keep up (T38-003/005). Two microbatches mean every weight byte (layers, lm_head, MTP, draft head per
     level) is read twice per round. Tree rows leave the weight floor beyond M~96-128 per microbatch (1.15x at 128, 1.43x
     at 160). Six draft levels cost ~4.4-5 ms per microbatch.
     - Best measured configuration: r = 0.91-0.92 of the champion's decode (pass rule 0.93).
     - Overall +3.7% before bubble, tree-kernel and scheduling losses; ~+6% ceiling with a hypothetical better kernel.
     - The build would be ~100+ GPU-min: PP port, tree GDN/attention/conv, tree drafting, new pipeline scheduler.
   - Hybrid "PP prefill phases + TP decode" (1.5x weights): fits in memory only with cuts (graphs to ~0.8 GiB, embedding
     on host). But it saves ~0 on ~300-token prompts: 3 stage-times x ~21 ms vs a 62.6 ms TP prefill extra, because the
     phase pays its own weight reads and fill/drain. Batching prompts to amortise fill/drain costs more than it saves in the
     closed loop: ~1.1 prompts/s, and a waiting prompt idles its slot. Not measured end-to-end.

## What others can reuse
- **Exact offline tree-acceptance tooling:** agents/leader-38.w1/tree/. Chain/static/dynamic/level-wise trees and
  draft-vocab subsets, CIs over sequences, CPU only (~7 min).
- **GDN exact tree verification:** the chunked delta rule's UT/WY form with the ANCESTOR mask in place of the
  strict-lower mask gives each node its own path's recursion. No per-node states; commit = replay of the accepted path.
  Algebraic; not implemented.
- **Full-width (PP-shape) GEMM table and split-K sweep** of the champion's kernels: agents/leader-38.w1/pp_gemm/,
  s2/splitk.json.
  - At M=128 on N=5120 shapes, split-K 3 with 128x64 tiles beats the champion's table pick: gdn_out 45.4 vs 56.5 us,
    down 117.5 vs 155.8 us. TP shapes are different; not checked.
- **t27 draft-head screen at M=32:** its scores kernel uses BM=16, so it reads U twice (323 us vs a ~109 us U-read
  floor). BM=32 would likely save ~150-190 us per draft level at 32 rows. In today's TP engine, M per draft step is
  16 seqs: check whether the same applies there. An exploit-sized item, untested.
- **PP=2 runs with the champion code** once the gdn/replay.py min(default=0) guard is in (T38-004).
- **Mixed-serving measurement:** TP prefill costs 205 us/prompt token of wall vs ~105 us of GPU time per GPU in PP
  (spec off, nsys, 64 prompts).

## Opinion
Decode on two 4090s is pinned by bytes: TP reads weights once but pays the AR; PP avoids the AR but reads twice. Trees
convert extra rows into tokens, but the rows aren't free beyond ~100 per weight pass, and deep drafting re-reads the
drafter. The one clearly large lever I found is PP's prefill (~2x cheaper per prompt token), which can only be harvested
in a design whose decode is no worse than TP's. I didn't find one that fits on these GPUs.
