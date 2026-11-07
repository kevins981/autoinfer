# TREE (run test-3)
root: whole-engine speedup of c=16 output tok/s (focus.md)

## Round 1 ideas (scored by manager; status: proposed; selection pending the human's review)
Sources: brainstorm/r1/systematic_engineer.md (SE), brainstorm/r1/adjacent_field.md (AF).
Scores: novelty (N) and upside (U), 1-5 each.
Open discrepancy between the two brainstormers: SE says prefill steps are CPU/launch-bound (eager,
~115 ms of CPU per step, under nsys). AF measured GPU 0 busy 95% of the time, with ~1.4 s of the
6.35 s AR time being skew (waiting for the peer). Both agree prefill steps are ~35% of wall time;
they disagree on why. This affects R1-A versus R1-B/R1-L.

| id | idea | from | N | U | reason |
|---|---|---|---|---|---|
| R1-A (done r1: T1-001 unofficial +2.2% alone; +4.8% stacked in T1-005) | CUDA-graph mixed prefill+decode steps (lift the 96-token prefill-graph cap) | SE1 | 2 | 4 | 35% of wall time in prefill steps (both measured), and a one-flag first test. The mechanism is contested: CPU launch overhead may be partly an nsys artifact (AF: GPU 95% busy). Needs the 128K test (graph memory). |
| R1-B (done r1: T1-002 official +5.0%; +AG T1-004; with R1-A = T1-005 official +10.7%, CHAMPION) | Custom host-staged 2-rank all-reduce for no-P2P PCIe (also used for the logits all-gathers) | SE2, AF1 | 3 | 4 | AR is ~21-23% of GPU time; NCCL gets 7.7 GB/s vs a 20 GB/s measured bidirectional floor; exact (a+b commutes). Real engineering. Proposed independently by both brainstormers: weak extra signal. Backend switch as a code default: (ruling 3: allowed). |
| R1-C | NCCL env tuning (protocol / channels / buffsize) | SE2 (first step) | 1 | 1 | AF's microbenchmark (job 2) measured it: at most -6% at prefill size, worse at decode size. Effectively ruled out. (ruling 3: allowed). |
| R1-D | Prefetch the next GEMM's FP8 weights into L2 during each all-reduce (side stream in the graph) | SE3, AF2 | 5 | 3 | Novel. DRAM is idle during ARs (NCCL grid=2) and the 72 MB L2 fits the next GEMM. Upside +4-15%, napkin math on assumed L2 behaviour. It competes with R1-B for the same window. Has a 1-GPU-min kill test. |
| R1-E | Retune Marlin configs per shape for M=48 decode (+ split-K for small N) | SE4 | 2 | 3 | Marlin is 47% of decode. Measured per-shape bandwidth 490-750 GB/s vs ~950 achievable; not compute-bound. Config sweep est. +5%, a new kernel +10% (low confidence). |
| R1-F | Local argmax for draft logits (existing flag use_local_argmax_reduction) | SE5a, AF4a | 1 | 1 | ~0.6 ms/step ≈ +1.5%: below noise alone, nearly free. Belongs in a bundle. |
| R1-G | Local-argmax fast path for greedy target verification (no full-logits all-gather) | AF4b | 2 | 2 | 0.84 ms/step ≈ +2%. Generic greedy fast path. Exactness of tie-breaking needs checking. |
| R1-H | In-place FlashInfer draft-metadata update (remove ~1 ms of CPU gaps per step) | SE5b | 2 | 2 | Up to +2.5%. Measured gaps of 110-240 us (under nsys, so possibly inflated). |
| R1-I | FP8 draft lm_head (existing draft_lm_head_quantization="fp8") | AF4c | 1 | 2 | ~+3%. FP8 activations in the drafter only; verification stays exact. allowed (ruling). |
| R1-J | Reduced-vocab draft lm_head (vocab prefix, frequencies from non-benchmark text) | SE5c, AF4c | 3 | 3 | Draft heads cost 2.7 ms/step (measured); 64K prefix covers 96.7% of non-benchmark tokens; est. +3-6% if acceptance holds. allowed (ruling). |
| R1-K | GDN spec-decode: write one recurrent state per step instead of three (replay accepted tokens) | SE6 | 4 | 2 | GDN decode is 9% of the step (measured); ~+4% est.; high effort (state bookkeeping, prefix-cache align). |
| R1-L | Overlap the prefill all-reduce with compute via two micro-batches | AF3 | 4 | 2 | AR is 33% of prefill-step time (measured); +5-8% est.; heavy engineering; works best on top of R1-B. |
| bundle F+G+H | MTP/logits overhead trims together | SE5, AF4 | 2 | 3 | Combined est. +4-7%, mostly exact and low risk; a candidate to share a slot. |

Ruled out by the brainstormers (with numbers): PP=2 (doubles weight reads; TPOT gate); 3 draft tokens
(~+4% net, at the noise limit; worth revisiting after R1-F..J); NCCL-only tuning (R1-C, measured).

### Round 1 results
- T1-001 R1-A flag alone: unofficial 645.2 (+2.2%, noise).
- T1-002 R1-B host-staged AR: official 662.8 (+5.0%).
- T1-003 diagnosis (no code). T1-004 + host-staged all_gather: unofficial ~0.
- T1-005 = T1-004 + R1-A flag: official 699.2 (+10.7%) -> CHAMPION.
- Open leads from round 1: copy-engine path for large-M AR; prefill-step rank skew (~214 us/AR); full (non-piecewise) mixed-batch graphs; prefill stalls decodes (ITL p99 ~290 ms).

## Round 2 ideas (scored by manager)
Sources: brainstorm/r2/systematic_engineer.md (SE), brainstorm/r2/adjacent_field.md (AF). Base: champion T1-005.
| id | idea | from | N | U | status | reason |
|---|---|---|---|---|---|---|
| R2-A | Logits diet: draft-logits local argmax (flag) + FP8 draft head (flag, ruling 1) + greedy local-argmax fast path for target verification (code); then the 3-draft re-check | SE1, AF2 (=R1-F/G/I) | 2 | 4 | **done r2: T1-006/7/9; T1-009 official +4.5% vs T1-005 -> CHAMPION** | Measured ~3 ms of a 38.5 ms decode step (lm_heads + logits all-gathers 6.0 ms total); flags exist; exact; est. +5-6%. Both brainstormers' top-2. Continues the comms line (it shrinks the remaining logits all-gathers, T1-004). |
| R2-B | GDN spec-decode state: checkpoint + log replay (write 1 state, not 3) | AF1, SE4 (=R1-K) | 4 | 3 | proposed | New measurement: 100 MB/layer of state traffic, a spill into the next GEMMs (gate_up 141 vs 122 us) -> ~5 ms/step; est. +3.5-5%, bitwise exact; medium-high effort; 1-GPU-min ceiling test. Makes 3 draft tokens cheaper. |
| R2-C | L2 prefetch of the next weights during the host-AR windows (8 SMs, 43 us x 137) | SE2 (=R1-D) | 5 | 3 | proposed | Unchanged premise; est. +3-7%, low-medium confidence; 1-GPU-min kill test. |
| R2-D | Marlin decode M=48 retune (670 vs 800 GB/s at M=3, same bytes) | SE3 (=R1-E) | 2 | 3 | proposed | The loss is inside the kernel (measured contrast); est. +5% from configs. |
| R2-E | Copy-engine full-duplex AR for large M | AF3 | 2 | 2 | proposed | ~+1.5-3%; ~half of the prefill AR time is the GPU 0 skew (hardware), which transport can't fix. |
| R2-F | 3 draft tokens re-check | SE, AF | 1 | 2 | tried r2 as T1-008: -6.9% unofficial (TTFT p50 210->792 ms, p99 gate fail); cause undiagnosed | ~+4% net now; better after R2-A/R2-B. |
Ruled down (measured): chasing the prefill skew or uneven TP (GPU 0 hardware, net ≈ 0); the decode step-start gap (an nsys artifact under node tracing); dequant + cuBLAS for prefill (Marlin at ~75-85% of peak).

### Round 2 results
- T1-006 local-argmax drafts: +1.1% unofficial. T1-007 + FP8 draft head: +3.0% unofficial, acceptance unchanged.
- T1-008 3 draft tokens: -6.9% (TTFT blow-up, undiagnosed).
- T1-009 + greedy verification fast path: official 730.9 (+4.5% vs T1-005, +15.8% vs BASE) -> CHAMPION.
- Open: k=3 TTFT blow-up diagnosis; reduced-vocab draft head (ruling 2); R2-B GDN state, R2-C L2 prefetch, R2-D Marlin still untried.
