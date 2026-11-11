# T9 write-up (leader-9, explore). Voluntary close after resync #1

GPU: ~70 of 112 granted used (seed 20 + start 60 + ABBA grant 12 + extension 20). Records: bank/T9-001..T9-007.
Worker: leader-9.w1. KB: KB67 (manager). Notes: local_kb/leader-9.md.

## Result
- **T9-005 is the CHAMPION: 948.9 tok/s official** (trials 947.9 / 949.8), +0.7% over T1-042's 942.2. All gates pass:
  GSM8K 92.8%, acceptance 2.327, p99 TTFT 1035 ms, p99 TPOT 35.2 ms, 128K pass.
- Two same-job ABBAs, every B above every A:
  - job 100 (100 prompts): +1.2%;
  - job 104 (400 prompts): +0.7%.
- What changed (S3c, `VLLM_T9_PINGPONG=1`, default on). In mixed steps with R >= 256, the champion's two-chunk token split now:
  1. sends chunk 1's ARs over a second copy-engine lane as well, instead of the SM-driven hostar;
  2. carries each chunk on through the next layer's add+norm and qkvz/ba (or qkv) projection before the join.
- Cores stay full-row.
- Correctness:
  - the residual stream and qkvz/qkv are bitwise identical to the full-row path;
  - ba differs at ULP level because its GEMM runs on chunk rows (the same class as the champion's own split).
- Memory: +~130 MB/rank, so the prefill-graph capture cap goes from 2048 to 1536 (T5-004: neutral). KV is unchanged.

## Forecast vs outcome
| | forecast | outcome |
|---|---|---|
| proposal T9-1 payoff if it works | +5.6% | |
| self-corrected before spending (KB66: comm-only is 15-23% of split-step span, not 30%) | +4%, central +2% | |
| delivered | | +0.7% official |

Where the gap went, by measurement:
1. **Real GDN core.** The real GDN prefill and FA2 core overlap far less than my sleep-kernel stand-in (T9-002): 39-58% of the AR hidden beyond the champion split, vs 86-96% with the stand-in. The two chunks mostly share SMs, so only the copy engines give a real overlap resource. Inside the ping-pong the copies run at 8-14 GB/s (T9-003). The B-waits-A dependency is NOT the cause, and uneven splits are worse.
2. **Breakable CUDA graphs.** Mixed steps replay as breakable graphs: segments are cut at the eager GDN/attention cores, and streams must join inside a segment. True cross-layer ping-pong would need per-chunk capture lists. The in-segment variant S3c gets ~45-60% of it (T9-004).
3. **The 1536 cap.** Needed for memory; neutral per T5-004.

## Measurements worth keeping
- **NUMA (KB67, T9-001).** With pinned slots verified on node 0, copy-engine copies run 20-24 GB/s even with both GPUs copying. T7-001's ~14 GB/s didn't reproduce. T4-001's and T7-001's benches never NUMA-bound their processes, so KB65 was likely a far-node artifact (not proven).
  - Rule: every 2-process bench must bind both processes to node 0 and log numa_maps.
- **The champion split (T1-042) hid only 13-22% of the prefill AR.** S3c roughly doubles that.
- **Small steps.** R=128-192 ping-pong is negative: the split costs +48..84 us/layer with 64-row chunks.
- **Decode copy-engine AR (T9-006): dead.** Even with a separate L2-prefetch kernel it is 23 us/AR slower than hostar plus prefetch, because of the uncompressed DMA per-op latency. 4 pieces is worse than 2.
- **CE payload codec (T9-007): dead.** Encode/decode SM time exceeds the 22% byte saving: +108..399 us/layer, the same as T1-031. A lossless static CE copy also cannot carry the sparse overflow region.
- **GDN two-call state handoff.** Bitwise at 64-aligned boundaries (relative to the sequence start); ULP-level otherwise (T9-002).
- **Graph pool pitfall.** Tensors allocated during capture and freed while a later eager op or segment still needs them get recycled, which gave garbage output (T9-005 smoke6). Persistent buffers allocated outside the pool fix it.

## Left open (none clears 5% alone)
| item | estimate | note |
|---|---|---|
| Per-chunk cores (option B: metadata slicing + GDN handoff + separately captured chunk-B tail graph, flag-wait join in the eager callable) | <= ~1% | ~3-4 days; design in agents/leader-9.w1/m2_design.md |
| Rank rebalance toward GPU1 (GPU0 clock lock) | ~+2% payoff, ~+1% central | ~600 MB/rank of duplicated MLP slice on GPU1 (only 360 MiB free → KV cut, KB13 risk); a Marlin N-sub-range call so decode stays one GEMM |
| Regain the 2048 graph grid | ~0 | T5-004 |

## Opinion
- Mixed-step communication is now mostly hidden for large steps. What remains in mixed steps is compute (at the bf16 roof) and small-step overheads (T7's list).
- The champion is close to the sum of its measured floors. A next large gain needs fewer verify rows per output token, or fewer bytes per row crossing PCIe, not better scheduling.

## How to reproduce T9-005
- Apply bank/T9-005/change.diff to the task baseline (T1-042 + S3c), then `ws build` (csrc changes).
- Serve with the task flags plus bank/T9-005/serve_args.txt. This is T1-042's args with the prefill-graph cap at 1536 and the grid truncated to 1536.
- Switches:
  - `VLLM_T9_PINGPONG=1` (default; 0 = T1-042 behaviour);
  - `VLLM_T9_PP_CHECK=0` (test-only full-row recheck);
  - more listed in bank/T9-005/record.md, "How to reproduce / switches".
- The second DMA lane uses the same JIT ext (dma_staged_ar, NL=2), built at first server start.
