# T4 write-up (leader-4, explore). Closed voluntarily after the seed

GPU: ~1.5 of 20 used (jobs 24-27, worker leader-4.w1). Record: bank/T4-001. KB: KB62. Ruling obtained: deferred RMSNorm scaling allowed (task_definition.md, end).

## Direction tried: hide each TP all-reduce under its consumer GEMM
- Premise: in the champion's 23.8 ms decode step, 5.9 ms is all-reduce wait (128 ARs x ~46 us at M=48, T2-006). Moving 1/rms into the consumer GEMM's epilogue (allowed) would let that GEMM take K-chunks of h as they arrive.
- Go bar, set before measuring: at least 15 us saved per AR. Estimate: 30-40 us.
- Measured (T4-001; 2 processes, real FP8 Marlin shapes, M=48, 16 layers = 32 ARs, CUDA graphs, 6 interleaved reps):
  - Champion path (host AR + compression + L2 prefetch): 5.02 ms/chain.
  - Without prefetch: 5.60 ms. No AR at all: 4.11-4.24 ms.
  - Ceiling: removing an AR is worth 24-28 us. Prefetch already recovers ~18 us of the ~46.
  - AR on a side stream beside the consumer (grid 128-f; consumer doesn't wait, so this is an upper bound): saves -1 us (f=8) up to 5-8 us (f=32, or the copy-engine AR).
  - Cause: the AR takes 51 us alone and ~100 us beside Marlin (PCIe round trips stuck behind HBM traffic). The GEMM slows too: gate_up 115 -> 139 us at f=8.
  - Only gate_up (~105 us) is long enough to cover the slowed AR. qkvz and down (~50 us) are not. Hiding just the gate_up AR is worth <= ~1.5% of a decode step.
  - A smaller grid by itself is cheap: +1-5 us per GEMM.
- First-chunk latency of a host AR (48 x cols): 1/8 width 12 us, 1/4 16 us, full 40 us (compressed). Each extra chunk costs ~10-12 us (cf. KB24).
- Interpretation: communication and memory-bound GEMMs are zero-sum on this machine (KB62), like compute vs compute in KB59.

## Considered and rejected (napkin math, no GPU)
- **Fewer ARs through replicated weights:** no memory.
  - Per rank: model ~15.0 GiB, ~7.9 GiB left (KB16), of which KV 5 GiB and CUDA graphs ~1.3-1.8 GiB.
  - KV needed for one 128K request: 16 attention layers x 4 KV heads x 256 dim x 2 (K, V) x 1 B / 2 ranks = 16 KB/token/rank, so ~2 GiB plus GDN states. At most ~2.5-3 GiB could be freed.
  - Replicating all GDN mixer weights (qkvz + out_proj, 115 MB/layer x 48 layers) costs +2.75 GB/rank. That roughly fits, but it doesn't pay: with mixers data-parallel by request, each rank streams +2.76 GB per step (~+3.2 ms) to save 64 ARs x ~25 us (~1.6 ms).
  - The MLP is worse: 133 MB/layer/rank.
- **PP=2 instead of TP=2:** no AR, but each GPU streams its 12 GB of weights once per 8-sequence microbatch. That is ~2 x 15.7 ms per step for 16 sequences vs 23.8 ms. Worse (agrees with test-5).
- **Faster prefill compute:** fp16-accumulate MMAs (2x rate on GeForce) lower the intermediate precision, so they're not a reordering. An exact split of the bf16 activation into 2 FP8 pieces needs 2 passes, which at best breaks even with bf16 on this card, and the e4m3 exponent range makes it inexact for small values. Uneven TP for GPU 0's locked clock was already ruled ~<=3% (KB8).
- **Cost model for the next team:** a trial is ~97 s ≈ processed rows x (~0.27 ms bf16 tensor work + ~0.1 ms AR) + a fixed weight-stream floor per step. Prompt rows (~104K) are fixed. The levers left are fewer verify rows per output token (acceptance, T3) or removing communication outright (not hiding it).
