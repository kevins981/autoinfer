# Final report: run sweep-unofficial-eval (qwen38-27b-fp8-2x4090)

## Result
**CONFIRMED champion: T3-004 at 908.5 tok/s official** (bank/T3-004/official/1/), **+42.9% over BASE (635.9)**.
All gates pass: GSM8K 93.2%, latency (TTFT p50 174 ms, TPOT p50 16.2 ms), 128K request. HEAD = CONFIRMED (nothing unconfirmed).
Path: bank/T3-004/ (change.diff + serve_args.txt); it contains every promoted change below.

GPU: 462.5 of 1000 GPU-min spent (teams 403.5, manager evals/reference 59.0); 537.5 left unspent.

## Official confirmation chain
| Confirmed record | Official tok/s | vs BASE | Unofficial chain expectation |
|---|---|---|---|
| BASE | 635.9 | — | — |
| T2-004 (T1-001, T1-004, T2-004) | 728.8 | +14.6% | 697.2 (+4.5%) |
| T1-005 | 760.8 | +19.6% | 765.7 (-0.6%) |
| T1-012 (T2-011, T2-012, T1-010, T1-012) | 785.3 | +23.5% | 789.9 (-0.6%) |
| T2-018 (T1-015, T2-016, T2-018) | 838.0 | +31.8% | 836.7 (+0.2%) |
| T1-019 (T1-016, T1-017, T1-019) | 897.5 | +41.1% | 884.5 (+1.5%) |
| T3-004 | 908.5 | +42.9% | 916.4 (-0.9%) |

One revert: T2-015 (GDN single-write state), exactness bug at max_model_len after a replay step, found by T2's own post-promotion check (job 104); fixed and re-promoted as T2-016.

## Promoted changes (all in the champion)
| Record | Team | Change | Unofficial gain |
|---|---|---|---|
| T1-001 | T1 | FP8 MTP draft lm_head + local argmax reduction (serve args; ruling 1) | +3.26% |
| T1-004 | T1 | batch-sharded sampling (serve args) | +1.61% |
| T2-004 | T2 | prefill CUDA graphs up to 512 tokens (serve args) | +4.49% |
| T1-005 | T1 | host-pinned one-shot all-reduce for <=128 rows (no P2P), bitwise == NCCL | +5.07% |
| T2-011 | T2 | reduced-vocab draft argmax (ids < 98304; ruling 2) | +0.52% |
| T2-012 | T2 | Marlin tile heuristic for M 17-48 | +0.85% |
| T1-010 | T1 | no-sync FlashInfer decode metadata | +1.89% |
| T1-012 | T1 | chunked GEMM + side-stream AR overlap in eager prefill | +0.52% |
| T1-015 | T1 | exact split-KV under the no-sync plan (problem found by T2's profile) | +1.29% |
| T2-016 | T2 | GDN single-write SSM state with exact replay (prefix caching on) | +3.53% |
| T2-018 | T2 | GDN lazy commit (every other replay step) | +1.60% |
| T1-016 | T1 | Triton W8A16 for out_proj/o_proj on checkpoint layout | +0.34% |
| T1-017 | T1 | checkpoint-layout MLP/out_proj; W8A8 (checkpoint's own scheme) for large M | +3.01% |
| T1-019 | T1 | no Marlin left; tuned 4090 W8A8 configs; full Triton warm-up | +2.12% |
| T3-004 | T3 | target-model local argmax for all-greedy batches (removes per-step logits all-to-all) | +2.11% |

## Teams
- **T1** (leader-1; 210 granted, 193.6 used; closed on own recommendation): 9 wins (above). Mapped the decode step down to measured floors: host AR at the PCIe floor; decode GEMMs near HBM floor; host syncs worth ~2% (lesson: put the CPU in the graph). Negatives: 3 drafts (x3: -4.9%, -6.3%, -10.2%), AR+norm fusion, max-num-batched-tokens 512, lm_head via addmm, W8A16 config tuning, runtime-JIT stalls (T1-018, fixed in T1-019). Last map left open: a new W8A16 decode GEMM >0.9 TB/s (~3-4%, high risk); lm_head and drafter GEMMs (~0.3 ms each). reports/T1-writeup.md, reports/T1-sweep/update-6.md.
- **T2** (leader-2; 140 granted, 128.7 used; closed on own recommendation): 5 wins (T2-004, T2-011, T2-012, T2-016, T2-018), 1 self-found revert (T2-015). Closed by measurement: decode GEMM at 0.93-0.96 of read ceiling, GDN headroom, lazy commit K=3 (-0.59%), tree speculation (T2-020: top-2 coverage 0.815 vs ~0.93 needed), L2 prefetch, Humming (= Marlin). reports/T2-writeup.md, reports/T2-sweep/update-5.md.
- **T3** (leader-3; 80 granted, 57.9 used; closed on own recommendation): 1 win (T3-004). Withdrew T3-002 (admission batching) after its benchmark-picked value failed on generic data. Nulls: copy-engine duplex AR (1.8x worse), GDN ba GEMM (already off the critical path), T3-003 (-0.72%). Left open: SM-driven duplex prefill communication; the mixed-step fixed cost (~1,700 eager launches; full graph for mixed steps); principled admission batching. reports/T3-writeup.md, reports/T3-sweep/update-3.md.
- **T4** (leader-4; 80 granted, 13.0 used; stopped at wrap-up): no win. T4-001: two-stream split of mixed steps, each half with its own host AR (microbench job 204: ~4% of wall, bitwise equal), built and 128K-passing, but correctness unresolved (hidden-state diffs at 531/1159 rows). reports/T4-writeup.md, reports/T4-sweep/update-1.md.
- **T5** (leader-5; 80 granted, 10.3 used; stopped at wrap-up): no win. C1, lossless block-coded AR payload: offline gate passed (0.77 of raw bytes, bitwise on 160 files, T5-001); kernel not built; stacks with T4-001. reports/T5-writeup.md, reports/T5-sweep/update-1.md.

## Open for a next run (from the last map updates)
1. Mixed (prefill) steps: ~27% of wall. T4-001 two-stream split with per-half host AR (~4%; fix the correctness diff first) + T5's lossless AR encoding (~4% napkin, needs a bitwise stress test) + a full CUDA graph for mixed steps (T3).
2. A W8A16 decode GEMM above 0.9 TB/s (T1, ~3-4%).

## Rulings made in this run (task_definition.md)
Prefix caching may not be turned off (human); scheduler knob values may not be picked on benchmark prompts (manager); fp16 accumulator inside W8A8 blocks not allowed (human, final); lossless communication encoding allowed (manager). Humming was banned while it wrote to $HOME (task env fixed by the human), then re-allowed.
