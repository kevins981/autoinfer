# T4 map update 2 (closing)

## Tested since update-1
| candidate | result | evidence |
|---|---|---|
| HMAR all-reduce fused with add-RMSNorm + dyn128 quant (decode) | **screen negative**: 852.1 vs 850.7 tok/s (1 start each), decode D=16 step +0.37 ms; per-pair graph microbench fused = unfused | bank/T4-002, review T4-3 |
| recurrent GDN prefill restricted to <= 64-row prompts, on T3-007 | **stopped after 1 start**: 851.4 vs 853.6 (-0.26%); expected ~+0.6%, below one start's resolution | bank/T4-003 (job 88) |

## Verdicts changed
- The decode all-reduce + norm pair is dominated by HMAR (+ prefetch join): ~44 of ~47 us at M=48; the separate norm costs ~2-3 us
  inside a CUDA graph, so "launch saving" from fusion inside graphs is ~0. Framework note: inside CUDA graphs, small-kernel fusion
  buys only the kernel's own bytes, not a launch.
- Mixed steps: the recurrent-prefill dual form (03 §4.2) is a real but small, length-dependent trade (wins below ~64-128 rows,
  loses above ~256); the large mixed-step lever I ranked #1 was mostly an nsys artifact (T4-001).

## Final ranking (what I would do next, vs champion T3-007 853.6)
1. GDN DeltaLog with the log kept OUTSIDE the spec state slots (T5 is on DeltaLog): besides ~1-2.6% decode bandwidth it would free 2
   of 3 GDN state slots per request (~2.4 GB/GPU of the 4.4 GiB KV budget), which unblocks k=3 drafts (~+3.5% before memory
   effects, review T4-1) and more KV headroom.
2. k=3 drafts once 1. frees the slots.
3. Mixed-step GPU work (~34% of time unprofiled): GEMMs at M 300-900 run ~210 TFLOP/s on Triton; a better FP8 block-scaled GEMM
   for large M and the 132 NCCL all-reduces are the remaining big items; no cheap lever found.
4. Small: recurrent prefill <= 64 rows (T4-003, ~+0.6% napkin, needs many starts to resolve).

## Addendum: closing review (reviews/T4-5) and T4-004
The closing review challenged the close and proposed one cheap lever: a finer mixed-step graph grid (every 8 tokens to 1024) to cut
padded rows (~+1% est.). Tried as T4-004 (flag only): server start OOM at graph instantiation (job 94). Memory headroom is again the
binding dimension (T1's lesson 2); freeing GDN spec state slots would help this too.
