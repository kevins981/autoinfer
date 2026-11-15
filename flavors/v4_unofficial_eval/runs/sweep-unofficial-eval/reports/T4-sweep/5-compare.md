# T4 sweep, step 5: my map against T1/T2/T3 maps and the bank

Read after files 1-4: T1/T2/T3 sweep folders (4-knobs, 6-ranking, final updates), the three write-ups, all
bank record summaries, shared_kb.md. Status of my candidates (file 4) on current code:

| my id | candidate | status | evidence |
|---|---|---|---|
| P3 | pipelined (up/down duplex) host-memory AR, all sizes | **tested negative** | T1 jobs 40-46 (write 22 + read 22 us at M=48, "both at once 30 us"; dependent AR 45-48), T3-001 (10 us + 78 us/MB serialized, grid 4..32 flat, pipelined variant no better), T3-006 (copy engines ~half rate, no duplex). My "two independent resources" assumption about PCIe up/down is wrong on this host: combined traffic saturates ~25 GB/s. I stopped my worker's microbench before any GPU spend. |
| R1 | decode W8A16 GEMM at >0.9 TB/s | tested to the limit of the kernel family | T1 jobs 187-192 / T1-020 (0.77-0.90 TB/s ceiling, -0.31%), T2-017, T2-013. W8A8 at decode M: T1 job 184 (gate_up 1.28x slower). Open only via a new kernel design (T1 rank 1, ~3-4%). |
| R2 | AR + add + RMSNorm fusion | tested null/negative | T1-007 (-1.6%), T1-009 (noise) |
| R3 | lm_head kernel | tested negative in situ (addmm, T1-014); Triton GEMV untested | ~0.3 ms/step |
| R4 | FP8 mma with fp16 accumulation inside a K-block (prefill) | **untried; never asked** | listed by T1 (R2b, "risky") and T2 (KE3, "needs a ruling", parked). I asked the manager for the ruling. |
| S1 | k=3 drafts | tested negative 4 times | T1-003/008/021, T2-008 (acceptance +15% but step +28%; M=64 is compute/dequant-bound on this card) |
| S2 | smaller draft vocab | partially built (98K subset, T2-011) | further cut untested, small |
| tree | tree drafts | measured no-go | T2-020 (best +3.5-4.1% coverage, not paying) |
| K1-lazy3 | GDN commit every 3rd step | tested negative | T2-019 (-0.59%) |
| P2 | decode column-split GEMM||AR overlap | untested as a record; related L2-prefetch-during-AR negative on host AR | T2-005..007: "link and HBM are coupled" on the host AR path |
| P4 | **layer-level two-micro-batch overlap in mixed steps** (split the step's rows into 2 halves that each run the whole layer stack, staggered so AR(half A) runs under compute of half B) | **untried in this form** | what was tried is the *within-one-GEMM* row split (T1-006, T1-012 +0.52% eager only, T3-003 -0.72% in graphs): only AR(chunk 1) || GEMM(chunk 2) of the same GEMM, then everything waits. T1 ranked the layer-level version (W1) #2 at ~4-13% but built only the GEMM-level one. |
| W1 | hold prefills to batch prompts | built and withdrawn (T3-002: value picked on benchmark data) | |
| W2 | host gaps | T1-010, T2-004, T2-014 harvested/closed | nsys overstates gaps ~10x |

## What my sweep found that the other maps lack
1. **FP8 compute constant.** My calibration shows W8A8 reaching ~180+ TFLOP/s in large mixed steps, so the
   FP8/fp32-acc peak is 330, not 165 (SPEED_OF_LIGHT). The other maps priced prefill at 165 TF. This changes
   the R4 (fp16-acc) prize: it is 330 -> 660, and the current W8A8 already runs at ~75-85% of 330.
2. **NCCL cost in mixed steps is flat in size** (173 us at 1.4 MB ... 330 us at 9 MB, both GPUs equal, start skew ~36 us
   in job 199): on today's clocks the "half is rank skew" explanation (T1 update-1) does not hold; it's link time.
   Comm is ~0.1 ms per prompt token (agrees with T3 job 198), ~10% of wall.
3. The layer-level micro-batch overlap (P4) as distinct from the GEMM-level row split.

## What they have that my sweep missed
- **Host CPU as a device** (T1-010 sync removal, T2-004 graph capture of mixed steps): my graph had no CPU node.
- **Weight layout as a shared data-node property** (T1-017/019): I took HEAD's layout as given.
- **Push a reduction before the collective** (T3-004 local argmax): already in HEAD; I didn't see it as a site.
- **Written-but-never-read tensors** (T2-016 GDN state writes): in HEAD already.
- **Link and HBM coupling** (T2-005..007) and **no duplex** (T1/T3): my two-resource link model is wrong here.
- **Runtime JIT as a cost node** (T1-018).

## Where I disagree / what I'd revisit
- P4 is not refuted by T3-003: T3-003 split every GEMM into two half-M launches inside one serial chain, so it
  paid the re-read and kernel-efficiency cost for at most one GEMM's worth of overlap per AR. The layer-level
  schedule hides each AR under a *whole layer half* of the other micro-batch. The risk is contention (T2's
  coupling finding, T1-012's TPOT rise), which a 2-stream microbench can measure before any engine work.
- R4: worth a ruling, not a park. If allowed, it's the cheapest large item (a Triton `out_dtype` change plus a guard).
