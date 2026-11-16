# T1 map update 1 (resync 1)

## Tested since 6-ranking
| candidate | result | evidence |
|---|---|---|
| #1 AR transport: NCCL env knobs | no gain: PROTO=Simple = default; LL128 worse (+25-60%); MIN_NCHANNELS=4 neutral at decode M | GPU job 1, agents/leader-1.w1/ar_notes.md |
| #1 FlashInfer PCIe-IPC AR | impossible here: cudaIpcOpenMemHandle needs peer access | job 1 |
| #1 custom host-mapped AR (new build, worker) | **microbench: bitwise equal to NCCL; 41 vs 61 us at M=42, 46 vs 65 at M=48, 19 vs 39 at M=16** -> napkin -2.7 ms/decode step (~+6-8%) | job 25, agents/leader-1.w1/hmar_result.json; harness not yet run (balance) |
| #2 FP8 draft lm_head | **+6.7% on BASE, promoted (T1-001)**; ~3x napkin, mechanism unexplained (review T1-1) | T1-001 |
| #2 reduced-vocab draft head (K=32768) | 662.4: +4.2% on BASE, loses to T1-001 (acceptance -4.6%) | T1-002 |
| #2 FP8 head on T2-003 | +0.75%, unresolved; needs KV 4.25 GiB to fit (1 preemption/start) | T1-003 |
| #5 k=3 on T2-004 (KV 4.25 GiB) | screen negative, confounded: KV capacity capped running requests at 12 | T1-004 |

## Verdicts changed
- **Mixed-step AR (my calibration's biggest item) was mostly not link time.** Bare NCCL moves 2.66 MB in ~205 us; the
  profile's 481 us was waiting for the other rank's eager launches (T2's finding; T2-001's prefill graphs fixed it).
  My framework use missed this: I priced the AR from its kernel time without checking the producer-side skew. Lesson:
  a collective's kernel time includes waiting; compare against a bare microbench before pricing it.
- Decode AR is the real link-bound item, and NCCL's ~30 us floor is the problem: a custom kernel gets ~20 us back.
- **Memory is now a binding dimension** I didn't have in the map: champion code + prefill graphs leave <0.6 GiB of
  headroom. FP8 draft copy (+0.6 GiB) and k=3 (+30% KV per request) both collide with it.

## New candidates
- k=3 with ~5.5 GiB KV (if memory can be freed elsewhere, e.g. prefill graph cap, or dropping the bf16 lm_head
  copy... not allowed for the target) - untested.
- Pipelined host-mapped AR (per-chunk flags): napkin ~25-30 us at M=42.

## Current ranking
1. Host-mapped AR on T2-004 (harness, ~8 GPU-min) -> record + champion candidate if it holds.
2. Pipelined hmar variant (microbench 1 GPU-min, then harness).
3. k=3 with enough KV for 16 requests (needs memory headroom; ~8-16 GPU-min).
