# T1 write-up (leader-1, worker t1-w1)

## Wins (all on the official chain or HEAD)
| record | what | unofficial | official |
|---|---|---|---|
| T1-001 | MTP draft lm_head fp8 + local argmax reduction (serve args) | +3.26% vs BASE | confirmed |
| T1-004 | batch-sharded sampling (serve args) | +1.61% | confirmed |
| T1-005 | host-pinned one-shot all-reduce for M<=128 (no P2P; /dev/shm mapped buffer, fixed 4-block grid, bitwise == NCCL) | +5.07% | 760.8 |
| T1-010 | no-sync FlashInfer decode metadata (plan from host upper bound, exact device lengths) | +1.89% | in chain (785.3) |
| T1-012 | chunked RowParallel GEMM + side-stream AR in eager prefill steps | +0.52% | 785.3 |
| T1-015 | exact split-KV under the no-sync plan (device rewrite of o_indptr/merge_indptr/block mask) | +1.29% | in chain (838.0) |
| T1-016 | Triton W8A16 on the checkpoint layout for out_proj/o_proj | +0.34% | in chain (897.5) |
| T1-017 | gate_up/down/out on the checkpoint layout; W8A16 small M, block-fp8 W8A8 (checkpoint scheme) large M; FP8 GSM8K A/B 93.2->93.6 | +3.01% | in chain (897.5) |
| T1-019 | tuned 4090 W8A8 configs, W8A8 from M=80, qkvz/qkv moved too (no Marlin left), complete Triton warm-up; FP8 A/B 93.2->93.6 | +2.12% | 897.5 |

## Negatives (records filed)
k=3 drafts (T1-003, T1-008: acceptance +15% but step +13-25%, TTFT p99 fails); chunked AR on T1-005 code (T1-006, 1 noisy start);
AR+add+RMSNorm fusion (T1-007 grid 4: -1.6%; T1-009 grid 16: within noise); max_num_batched_tokens 512 (T1-011, -1.05%);
lm_head via addmm (T1-014, -1.29% despite -150 us/call in microbench).

## What we learned
- The host-pinned AR is at the PCIe floor (22 us write / 22 us read at M=48); NCCL tuning <=10%; FlashInfer IPC AR needs P2P.
- Marlin at M=48 is 1.07-1.14x the cold-HBM floor in isolation but 1.46x in situ. The GDN-layer gate_up excess
  (138.6 vs 117 vs 109 us isolated) is not L2, preceding kernels, clocks, smem carveout or TLB (jobs 89-91). Unsolved.
- The shipped baseline profile had GPU0 clocked at 2235 MHz; today both GPUs run at ~2670. The asymmetric-split idea depended on that and was dropped.
- Host-side syncs are invisible to FLOP/byte napkin math, but they were worth ~2% (T1-010). The framework's graph should
  include the CPU (scheduler, metadata) as a device with sync edges.
- nsys inflates CPU launch gaps ~10x vs real runs; microbench gains under ~0.5% don't reliably carry over. Use same-job ABBA, then unofficial eval.
- Record the cost side of every trade as a new candidate (the split-KV cost of T1-010 was found by T2, then recovered by T1-015).

## Late negatives
lm_head addmm (T1-014, -1.29%); tuned W8A8 with incomplete warm-up (T1-018, -6.1%: runtime JIT); W8A16 config tuning (T1-020, noise);
3 drafts on current code (T1-021, -10.2%: acceptance +15% but step +28%).

## Late lessons
- The biggest late win came from a sparring question: "W8A8 prefill needs a second weight copy" was an assumption. Dropping the
  Marlin repack let one checkpoint-layout copy serve W8A16 (decode) and W8A8 (prefill).
- Triton specialises integer args on %16 and pointers on alignment: warm every config through the real call path, at M and
  M+1, or the official eval (cold cache) pays multi-second stalls.
- Microbenches must be graph-timed: Python-loop timing added ~30 us of launch overhead and mis-ranked small kernels.

## Open (superseded list below kept for history)
In-situ GEMM excess in GDN layers (~1.5-2 ms/decode step): needs a real-model 2-layer experiment. Drafter ~0.3 ms, lm_head ~0.3 ms (kernel work).
Map: reports/T1-sweep/ (1-6, update-1..4).
