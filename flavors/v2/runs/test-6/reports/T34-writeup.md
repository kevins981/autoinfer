# T34 write-up (leader-34, explore): mixed-step all-reduce at its PCIe duplex floor — CLOSED (measured hard limit)

GPU: granted 80 (seed 20 + start 60); account shows 5.5 used, 74.5 balance to return. Job 345 (4.0 min, crashed) appears not in the
current account. Records: bank/T34-001 (transport floors + hybrid AR), bank/T34-002 (same-job hybrid vs champion paths).
Worker: leader-34.w1 (assets: agents/leader-34.w1/ASSET.md, h/, PLAN_stage1.md). Proposal: proposals/T34-1.md. Notes: local_kb/leader-34.md. Sparring: T34-1.

## Direction and premise
In big mixed steps the champion's copy-engine (DMA) AR is fully exposed. Exposed mixed-step AR is ~4.0 s of ~42 s per 200 prompts (T32-5).
Premise: the AR transport runs far above its PCIe duplex floor. The basis was leader-30's node trace, ~0.8 ms at 1100 rows with ~0.4 ms of one-direction fill/drain.
The proposed fix was a hybrid AR (copy-engine D2H in pieces + an SM reader of the peer's partial in host memory) at the floor, then codec, a fused epilogue and waves.
Payoff if it works was forecast at ~+5% (at the bar), central ~+2%.

## Results (Observed)
Both ranks, NUMA node 0 verified, graph-captured chains of 16 exposed ARs with a producer kernel, per-AR us on rank 0 (rank 1 within 1-2%).
1. **Floors (T34-001, job s6).** CE D2H alone 441 / SM host read alone 456 / both concurrently 631 us at R=1100 (11.26 MB/rank); 213/211/325 at R=512.
   At 0.78x bytes (codec size): 543 at 1100, 257 at 512. Duplex is nearly free (~25 GB/s per direction).
2. **Hybrid, first design (job s4).** Every reader CTA polled the host flags: 1233-4573 us at 1100, growing with CTA count (16/32/64). This is a polling storm.
   **Fix:** one thread polls the host flags with nanosleep back-off and publishes them to a device word. H1 (P=8, 16 CTAs) then ran at 639 us at 1100 (floor 631) and 311 at 512 (floor 325).
   Bitwise equal to bf16(float(x)+float(peer)): 0/24 at every R tested (zeros, inf, 1e30 rows), 0 flag timeouts.
3. **Same-job comparison with the champion's paths (T34-002, job s7), us/AR:**

| R | H1_8_16 | champion DMA (pieces 4) | champion split path | champion hostar (0.78 codec) |
|---|---|---|---|---|
| 1100 | 660 | 699 | 705 | **620** |
| 768 | 462 | 476 | 491 | **446** |
| 512 | 320 | 329 | 333 | **303** |
| 256 | 180 | 183 | 189 | **161** |
| 128 | 106 | 113 | - | **89** |

## Interpretation
- **Correction:** the champion DMA AR is ~700 us at 1100 in an isolated chain, within ~11% of the raw duplex floor. The ~800 us in leader-30's trace includes rank skew (GPU0 clock lock) and two lockstep chunks sharing the link. It does not show transport inefficiency.
  Caveat: the floor (job s6) and the champion DMA (job s7) were measured in different jobs. At R=512 the gap is ~1% (329 vs 325). The skew + lockstep explanation of the in-situ ~800 us is an inference, not a measurement.
- H saves 3-39 us/AR, ≤~0.8% of wall at best. That is below the stage-1 gate (+1.0%), so the served integration and ABBA were not built (no GPU spent there).
- **Classification:** hard limit (measured) for transport. The AR transport is at its PCIe floor; only fewer bytes or hiding can still move the exposed AR.
- Re-forecast payoff if it works: ~+2-3%, below the bar.

## Remaining leads (for a later team)
- **Bytes.** Hostar's 0.78 codec (620 us) already beats the raw floor. The codec-size floor is 543 (-12% vs hostar), but encode/decode and waits are unaccounted. Worth ≤~1%.
- **Hostar vs the DMA lane in exposed ARs.** Hostar is faster at every R >= 128 (-11% at 1100, -12% at 256). The champion uses DMA for split chunks so chunk 1's GEMM can run beside it, which only pays if the chunks are staggered, and leader-30's trace shows them in lockstep. Caveat: leader-30's arm C (VLLM_DMA_SPLIT=0) was slower in situ at pt 401-1000. Exploit-sized, needs an in-server A/B.
- **Scheduling (lockstep chunks, rank skew).** T30/T32's waves achieved 131 -> 116 ms at 734 pt in replay against an AR-off floor of ~90 ms. The remaining ~26 ms is unexplained: the "~2% ceiling" is an achieved result for one schedule, not a proven bound (spar T34-1). The cheapest next test is a dependency/critical-path accounting of that 26 ms.
- Rank skew: GPU0 clock lock; tested uneven splits were null. Unavoidability under every schedule is assumed.

## Assets
- **hyb2.cu** (agents/leader-34.w1/h, bank/T34-001): a graph-capturable hybrid AR at the duplex floor, bitwise, with a single-poller reader. Useful wherever an SM-side consumer can read the peer's partial straight from host memory, e.g. a fused AR epilogue.
- **A safe 2-rank AR bench** (bench2.py, job.sh). It has F_ce/F_sm/F_both floors, the champion arms (WITH_A=1), globaltimer timelines and a bitwise checker.
- **Served flag/ack protocol design** for a hybrid AR with any AR count per graph and eager/graph mixes (PLAN_stage1.md). No device traps in served code: an error counter plus continued polling.
- **Safety pattern** (after job 345's crash forced a server reboot): no unbounded waits in benches; pointer-lifetime asserts at capture/replay; a smoke stage (eager + one checked replay) before timing; os._exit on any error; a kill-both watchdog.

## KB candidates
1. Mixed-step AR transport is at its PCIe floor: in isolated exposed chains the champion DMA AR is 699 us at 1100 rows vs a 631 us raw duplex floor (CE D2H 441 / SM host read 456 alone, both together 631); hostar with the 0.78 codec is 620 (below the raw floor; codec floor 543). A hybrid CE-write + SM-read AR reaches the floor (639-660) but saves only 3-39 us/AR. In-situ ~800 us at 1100 was rank skew + lockstep chunks. (T34-001/002)
2. Polling host-memory flags from many CTAs throttles concurrent copy-engine writes (polling storm: 1.2-4.6 ms vs 0.64 ms with one poller thread + device-word broadcast). (T34-001)
3. Process/safety: a CUDA fault in a 2-process bench that uses stream wait-value ops on host flags left unkillable processes holding GPU contexts, and the server needed a reboot (job 345). Benches must use bounded device polls, assert pointer lifetimes across graph capture/replay, smoke before timing and os._exit on error. (T34-001)
