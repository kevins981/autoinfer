# T2 write-up (leader-2, worker t2-w1)

Qwen3.8-27B-FP8, 2x RTX 4090 (TP=2, no P2P), MTP k=2. BASE 635.9 tok/s official. HEAD at close = T2-018,
**838.0 tok/s official (+31.8%)**. Map: [T2-sweep.md](T2-sweep.md), files 1-6 and update-1..5.
GPU: 140 granted, ~123 used.

## Wins (all confirmed in the official chain)
| record | what | unofficial | how found |
|---|---|---|---|
| T2-004 | prefill CUDA graphs up to 512 tokens (`max_prefill_cudagraph_capture_tokens`) | +4.49% | calibration: mixed steps were CPU-dispatch-bound |
| T2-011 | drafter greedy argmax over token ids [0, 98304) (tokenizer BPE order, ruling 2) | +0.52% | sweep SM4 |
| T2-012 | Marlin tries the 128x64 tile first for M 17-48 | +0.85% | config sweep at the real decode M |
| T2-016 | GDN single-write SSM state: replay accepted rows next step (prefix caching on) | +3.53% | 2-graph: the kernel wrote 3 states/layer |
| T2-018 | lazy state commit every other replay step | +1.60% | profile + T1's in-situ drain measurement |
Also enabled T1-015 (+1.29%): our profile (job 101) found that T1-010 had switched off split-KV attention.

## Reverted / negative / null (all in the bank)
- T2-015: KR1 v2, reverted. Greedy output differed at max_model_len. Root cause: the runner sets
  num_draft_tokens_per_req=None when no row has drafts, which skipped the replay decision. Fixed as T2-016.
- PL1 L2 prefetch during the AR (T2-005..007): +2.3% on NCCL, negative on the host AR (link and HBM are coupled).
- 3 drafts (T2-008), drafter TRITON_ATTN (T2-010, OOM), FLA fast launch (T2-014, CPU -1/3 but no end-to-end gain),
  Humming kernel (T2-013, equal to Marlin), new decode GEMM (T2-017, Marlin near the read ceiling),
  lazy K=3 (T2-019, -0.59%).

## What worked
- **The graph step found the biggest single item.** "Which tensors are written but never read?" pointed at the 3 GDN
  state writes per layer, which became KR1 plus lazy commit, ~+5% together.
- **Calibration against a measured profile**, not intuition, found the mixed-step dispatch bottleneck (W1).
- **Gated, cheap-first experiments**: a microbench go/no-go before any build (PL1, KE1b, Humming), and GPU-free code reading before traces.
- **Re-profiling HEAD against an older profile** finds other teams' regressions (T1-010 split-KV).

## What I'd do differently
- Check the bank before testing: I duplicated T1's 3-draft test.
- Make exactness gates part of the build, not an afterthought. T2-015 was promoted with an untested path. The c=1
  OFF/ON/OFF logprob test at a length cap would have caught the bug before promotion; it is now our standard gate.
- Don't trust nsys gaps as recoverable time (T2-014).
- Napkins for removed writes should be priced where the write-back is paid (the next GEMMs), not at the writing kernel.

## Framework findings
See update-5.md: host dispatch is a resource; deferred writes are billed to later kernels; Link and HBM are coupled;
Speculate x Rewrite need an acceptance term; how to test exactness.

Worker t2-w1's own summary (builds, bugs found and how, lessons): [agents/t2-w1/summary.md](../agents/t2-w1/summary.md).
