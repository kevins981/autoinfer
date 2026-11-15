# T1 sweep, step 5: my map against the other maps and the run's history

State of the run when written (2026-11-15, right after T1's phase 1): no records besides BASE, empty shared_kb.md,
no reports or sparring notes, no resyncs. HEAD = CONFIRMED = BASE. The manager's BASE reference unofficial eval: 625.1 / 639.9 tok/s.
The only other map is T2's, still in phase 1 (only `reports/T2-sweep/1-inputs.md` and `2-graph.md` exist; no knob sweep yet).
So every candidate of mine is **untried** (nobody has built or tested anything on any code).

## My candidates: status
| id | candidate | status |
|---|---|---|
| X1 | faster all-reduce (NCCL tuning / custom host-memory one-shot AR) | untried; T1 microbenchmark running (worker t1-w1) |
| W1/W3 | overlap prefill ARs with GEMMs (half-batch stagger, or column-chunked GEMM->AR) | untried; T2 lists "overlap AR (link) with GEMM by splitting out cols or r" as a site |
| R2 | W8A8 FP8 prefill GEMMs with fp16 in-block accumulation | untried |
| R3 | better decode GEMM (Marlin at 0.67 of BW overall) | untried |
| R6 | draft lm_head FP8 (flag) / reduced vocab | untried; T2 lists `draft_lm_head_quantization` as present-but-off |
| R1 | local argmax instead of logits all-gather (draft: flag; target: code) | untried; T2 has the same site (argmax_v combine) and notes `enable_batch_sharded_sampling` / `compute_logits_local` |
| S1 | 3 MTP drafts | untried |
| R5 | torch.compile / prefill graphs | untried |
| W4 | async scheduling | untried |
| K2 | ReplaySSM GDN state | untried; T2 lists "per-row checkpoints for MTP rollback" at the S site |
| R4 | AR+RMSnorm fusion | untried; T2 lists it |

## What T2's (partial) map has that mine missed
1. **Marlin re-reads weights per 64-row block** (thread_m_blocks <= 4). I priced the prefill GEMMs as compute-bound (66.7 ms at
   165 TF vs 91.9 measured, "73% efficiency"). With re-reads, 437 rows = 7 blocks x 12.2 GB = 85 GB -> ~85 ms at 1 TB/s, which
   explains the 91.9 ms better than my compute napkin. Correction to my file 3: the prefill GEMM time is set by weight re-reads
   (HBM), not by tensor throughput. Consequence: a GEMM that reads each weight once at M=437 (large-M tiles, e.g. a dequantize-once
   or FP8 W8A8 kernel with big M tiles) is bounded by compute at ~67 ms (bf16/fp32-acc FP8): **~25 ms per prefill step = 7.8 s (~5%)
   without needing fp16 accumulation**. This upgrades R2 into two rows: R2a (read-once large-M kernel, exact or W8A8, ~5%) and R2b
   (fp16 in-block accumulation, further ~10%, risky).
2. **Place weights in L2 during the AR** (prefetch the next GEMM's weights into the 72 MB L2 while the link is busy and HBM idle).
   A Window/Place combination my sweep did not list: I only paired AR with *compute* of other rows, not with the *HBM stream of the
   next layer's weights*. Napkin: during a 60 us decode AR, HBM could prefetch ~48 MB; the next GEMM per GPU is 37-45 MB
   (in_proj/qkv or gate_up), so up to ~min(60, 50) us hidden per AR: up to ~6 ms/decode step (**~9%** at BASE AR speed; shrinks
   as X1 makes ARs short — the two compete for the same slack). Riskiest: L2 residency/eviction behaviour across kernels on sm_89
   (no TMA multicast; needs a persisting-L2 window or a prefetch kernel on a second stream).
   Framework note: my sweep missed it because I treated "link-bound window" + "HBM-bound next node" as dependent (the GEMM needs
   the AR result). The *weight read* part of the GEMM doesn't depend on the AR; only the multiply does. Splitting a GEMM node into
   [load W] -> [multiply] exposes the independence. That is a missing generic move: split a node into its data-movement and compute
   halves before applying the Window knob.
3. **Batch-sharded sampling / compute_logits_local** in Qwen3.5: an existing engine path for the R1 site.

## What my map has that T2's (so far) lacks
- A calibrated whole-run split (3-calibration.md): prefill-containing steps are ~42% of the official time and their ARs alone are
  ~15% (20.8 s); decode ARs+AGs another ~15%. T2's site list has no prices yet.
- The fp16-vs-fp32 accumulation rule as the decider for FP8 compute on GeForce (R2b), and the e4m3 product overflow risk.
- The custom host-memory one-shot AR (X1) as a concrete design, not only NCCL tuning.

## Disagreements
None yet (T2 has no verdicts). I will compare verdicts when T2's 4-knobs.md exists.
