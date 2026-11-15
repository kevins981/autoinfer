# T1 map update 5 (resync 3; HEAD = T1-017; CONFIRMED = T2-018 at 838.0)

## Tested since update-4
| candidate | record / job | result | verdict |
|---|---|---|---|
| GDN-layer GEMM excess (in-situ gate_up 138.6 vs 108 us isolated) | jobs 119, 127 | real weights isolated: no difference (weights not the cause); no concurrent kernels; clocks constant; on T2-016 the excess fell to +13 us; attributed to SSM-state write-back draining from L2 during the next GEMMs | explained; T2 (T2-018 lazy commit) acted on it |
| T1-012 deferral side effect (leader-2) | job 127 | fused AR+norm never runs in this stack; no difference | refuted |
| Small-prefill mixed steps (~100 ms) | GPU-free on job 127 | GPU 98% busy; real work: Marlin re-reads weights per 64-row block, host AR at 1 MB payloads | -> Marlin M>64 |
| Marlin M>64 | jobs 130, task 25 | step at M=65 (126 -> 191 us gate_up); par blocks already concurrent; 65-128 is dequant/compute bound | no cheap Marlin fix |
| W8A8 needs a 2nd weight copy (sparring T1-2) | jobs 137, 138, 141, 147, 149 | assumption, not measurement: a checkpoint-layout Triton W8A16 matches Marlin in decode (graph-timed); W8A8 0.67-0.79x Marlin at M>=384 | **opened a new dimension: weight layout as a knob** |
| out_proj on Triton W8A16 | T1-016 | PASS +0.34% | **won** |
| gate_up/down/out on checkpoint layout + W8A8 above per-shape thresholds | T1-017; FP8 GSM8K A/B job 157 (93.2 -> 93.6) | PASS +3.01% | **won (HEAD)** |
| lm_head via addmm (on T2-015/T1-012) | T1-013/014 | -1.29% | dropped |
| W8A8 configs tuned for the 4090 | job 156 (t1-w13) | 13-26% faster than vLLM's default config at every shape/M | next record (T1-018) |

## Framework notes
- The framework's Rewrite knob had "kernel choice" per node, but my map tied the weight-layout choice to the kernel (Marlin repack)
  and priced kernels one at a time. The **layout of a data node is a shared choice across all compute nodes that read it**; pricing
  must consider the set of kernels that can share one layout. My phase-1 map missed that; a sparring question found it.
- Profiles mislead in both directions (nsys inflation; untuned defaults). Graph-timed microbenches with cold rotated weights have been
  reliable predictors here (T1-017 predicted ~+4%, got +3%).

## Ranking now (against T1-017)
1. T1-018: tuned W8A8 configs + thresholds re-checked below M=128 (job 156: -13-26% on the W8A8 GEMMs). Needs unofficial + FP8 GSM8K A/B.
2. qkv/in_proj on the same path (checkpoint layout, W8A16 small M, W8A8 large M): these are the remaining Marlin GEMMs (~1/4 of
   GEMM time). Needs W8A16 configs for their shapes and a decode parity check.
3. Drafter MLP/attn GEMMs on the same path (if shapes match, they already are).
