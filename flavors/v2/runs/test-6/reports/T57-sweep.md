# T57 sweep (leader-57): the framework applied to the champion T52-005

Written before any T57 GPU work and before reading T56's analysis. Champion T52-005, 1212.4 tok/s (75.2 s/trial).
Numbers cite KB entries / records. "Napkin" = my arithmetic; "measured" = a cited job.

## 1. Inputs
**Model** (config, checked by T54): 64 layers = 48 GDN + 16 full attention (24 Q / 4 KV heads x 256, output gate);
hidden d=5120, MLP m=17408, vocab V=248,320; GDN 16 k-heads / 48 v-heads x 128, fp32 recurrent state
(1.5 MB/request/layer/GPU at TP=2); MTP drafter = 1 attention layer + MLP + fc.
Body params 24.3 B -> 12.17 GB FP8 per GPU (gate_up 178 M, down 89 M per layer; GDN qkvz 84 M + out 31.5 M; attn qkv+gate 73 M + o 31.5 M).
**Hardware** (per GPU): DRAM ~0.98 TB/s (P2 memory clock, KB119); exact-math MMA: bf16 ~150-165 TF, e4m3 with
per-128-K fp32 promotion ~340 TF roof, 230-296 achieved at M=256-2048 (KB130); L2 72 MB (~3.4 TB/s); 128 SMs.
Link: no P2P, host-staged over PCIe 4.0, ~22-26 GB/s per direction, AR = write up then read down (~0.6 us/row/AR,
KB91/KB109); CE copies ~20-30 us latency in graphs (KB74). Host: 48 cores, launch cost ~10-20 us/kernel eager.
**Workload**: c=16 closed loop, 400 prompts, ~262 in / ~227 out tokens, k=3 MTP, acceptance 2.66 tokens/step.
**Contract**: task_definition.md (exact math up to fp reordering; FP8 weights/KV; checkpoint W8A8 allowed with
conditions; bf16 comms; exact spec decode).

**Champion design (the starting point)**: TP=2; decode step = 64 rows (16 x (1+3)); decode GEMMs e4m3 register-direct
(t31b) + W8A8 bs_gemm for gate_up, AR-window L2 prefetch of the next GEMM's weights; host-staged AR with lossless
0.78 codec; GDN replay + lazy checkpoint + two-slot state; certified target lm_head; PCA-screened FP8 draft lm_head;
mixed steps: W8A8 GEMMs, token-split copy-engine AR lanes >=256 rows with link priority, 2 layer-pipelined waves >=768
rows with captured GDN core; piecewise prefill graphs to 1536 tokens.

## 2. Graph + napkin cost model, calibrated
Per decode step, per GPU (64 rows):
| node group | napkin floor | measured (T53 anat of T45-005, -0.2 ms cert lm_head from KB121) |
|---|---|---|
| body GEMMs, 12.17 GB + drafter GEMMs 0.57 GB | 13.0 ms (DRAM) | 12.6 ms (t31b 6.8 + bs_gemm 5.3 + Marlin 0.54): at floor thanks to L2 prefetch |
| 141 ARs x 64 rows x 8 KB (codec) = 72 MB/direction | 2.9 ms duplex / 5.8 ms half-duplex | 7.0 ms |
| GDN fp32 state read 1.2 GB | 1.2 ms | 1.85 ms |
| target lm_head (certified) | ~0.6 ms | ~0.8 ms |
| drafter non-GEMM (3 levels: ARs, sampling, screen) | ~0.2 ms | ~1.2 ms (KB126: ~0.5 ms/level non-byte) |
| attention + conv + small ops | ~0.5 ms | ~2.3 ms |
| **total** | **~18-21 ms** | **24.9 ms** |

Closed-loop trial model (T51's calibrated sim, agents/leader-51/sim/pig.py, reused): decode step 24.9 ms, one prefill
event per prompt costing F + c x prompt tokens. With (F, c) = (15 ms, 0.166) it gives 1133 tok/s vs T45-005's 1173
measured (-3.4%; KB111-era constants); (15, 0.12) reproduces the champion's 1212 (1206). So: **decode-only time 57.4 s
(76%), prefill events ~17.8 s (24%) per trial.** Sensitivities: -1 ms/decode step = +3.0% e2e; -0.01 ms/prompt token =
+1.4%; -1 ms fixed per event = +0.5%.
Prefill floor per token per GPU: 24.4 GFLOP at 340 TF = 0.072 ms (0.084 at 290 TF); AR link 141 x 8 KB = 1.13 MB/token
= 0.044 ms duplex (hideable), 0.085 ms half-duplex exposed. Today ~0.12 + 15 ms/event.

Where the gap to floor is: decode ~4-7 ms/step (AR structure ~1.2-4, small ops ~1.8, drafter overhead ~1.0,
GDN 0.65) and prefill events ~9 s/trial (fixed event cost 6 s + per-token excess ~3.5 s).

## 3. Knob sweep (node / index by node). Status: B = built in a record, T = tested (record/KB), U = untested
### Knob 1 Keep/Recompute
| candidate | verdict | status |
|---|---|---|
| keep k,v (fp8) / GDN S (fp32) | baseline; required (Rule 1/2) | - |
| ReplaySSM/DeltaLog (state write avoidance) | built: replay, lazy checkpoint, two-slot | B T1-012/T1-035/T37-002 |
| token-history keep instead of S (Row C) | loses: crossover ~190 tokens, contexts 260-500, O(T^2) recompute | T (T55 triage) |
| recompute instead of communicate (replicate out_proj, all-gather 6144-wide core output) | AG 60% of AR bytes but +15.7 MB/layer extra weight read/GPU -> decode wash; +1 GB/GPU | T napkin (T47 notes), U in code |
| keep prefix state across requests | +0.5% (KB86) | T |
### Knob 2 Rewrite
| candidate | verdict | status |
|---|---|---|
| 1/rms past the GEMM (bf16 and W8A8) | allowed; used in T52-002 cut-through (parity) | T |
| fused glue (norm->quant, silu->quant) | built | B T29-006 |
| distributivity on MTP fc (cat(e,h) W = eW_e + hW_h) | fc is 26 MB; nothing to gain | pruned |
| MLA-style absorption / RoPE commute | no low-rank latent in this model | n/a |
| dual form for GDN verify (M=4 rows/request) | already chunk-free replay kernel at M=4 | B |
### Knob 3 Speculate
| candidate | verdict | status |
|---|---|---|
| x[t+1] chain k=3 | champion | B |
| k=4 | KB127: break-even at <=0.115 ms/row above 64; W8A8 gate_up may get there; ~0..+1% | U on T45+ code |
| trees | -6..-8% at measured row prices (KB135) | T |
| load-adaptive k (drain at trial end) | allowed (clar. T36); drain ~4 s/trial, ~+0.5% ceiling | U, small |
| layer-output y[l] | p=0, approx only (forbidden) | pruned |
| no MoE / sparse-attention sites | model has none | n/a |
### Knobs 4/5 Split/Merge + Place
| candidate | verdict | status |
|---|---|---|
| TP=2 (heads/m split, AR after out/down) | baseline; AR bytes per token are a TP invariant (RS+AG = AR; d-split residual needs AG before the next column GEMM; K-split of gate_up needs a 34816-wide AR) | dominance |
| PP=2 | decode 0.91-0.92x TP even with trees (KB112), closed-loop sim loses (KB123) | T |
| PP for prefill only | needs whole layers per GPU: weights don't fit twice | pruned (capacity) |
| asymmetric alpha (GPU clocks) | GPUs symmetric since the reboot (KB113); uneven split lost when they weren't (KB82) | T |
| drafter on one GPU / replicated | removes ~9 small ARs (~0.33 ms) but +0.57 ms of extra weight reads per step | pruned (napkin) |
| embedding on host (zero-copy gather) | ~0.2% + frees 1.27 GB/GPU (KB125) | T microbench, U in server |
| CPU-side reduction | same bytes per GPU, DDR adds latency | pruned |
| split rows of a mixed step (token-split lanes) | built >=256 rows | B |
| split threshold <256 with link priority | measured only before link priority existed (T1-045, KB38) | U on current code |
### Knob 6 Window (+ combinations)
| candidate | verdict | status |
|---|---|---|
| merge decode rows of 16 requests | baseline | B |
| merge prompt rows into decode passes (piggyback) | loses with today's GEMM band (KB131) | T |
| coalescing prefill events (wait for a 2nd arrival) | slot idles ~7 steps for ~13 ms saved: ~0 (KB77, re-priced: still ~+0.5% at best) | T (old code), napkin |
| dual-batch overlap of decode half-batches | M=32 GEMMs cost like M=64, weights read twice (KB132/T53-007) | T |
| AR duplex pipelining in decode (chunked/LL/paced) | 2-6 us/AR found (KB78, KB116) | T (5 teams) |
| waves inside a mixed step (rows split) | built >=768 rows (T52-005); host-bound below | B |
| **asymmetric waves: decode rows || prompt rows, WITH link priority** | KB128 measured them WITHOUT link priority; T52 showed waves lose without it. Napkin per 320-row step: joined G(320)+A(320) ~27+26 ms vs max(G(260)+G(60), A(260)+A(60)) ~35 ms | **U with priority** |
| **mixed step host vs GPU span (eager pieces between graphs)** | T51: small mixed step ~51 ms wall, "wall above GPU span, host enqueue likely, unproven"; T2-006: ~1790 launches/step was 18 ms host run-ahead when GPU was slower | **U (attribution missing)** |
| **spatial prefill lane (compute-bound prefill on f SMs beside HBM-bound decode)** | the framework's Mechanism 2 (F vs BW). Killed by T2 (KB59: decode x128/(128-f)) on kernels that need every SM; T47 (KB129) showed DRAM saturates from 16-32 SMs, so it is a kernel-sizing property | **U on re-gridded kernels** |

## 4. Ranking (napkin gain vs 75.2 s/trial)
| # | candidate | napkin gain | wins if | riskiest assumption | cheapest test that could show it wrong |
|---|---|---|---|---|---|
| 1 | Spatial prefill lane on re-gridded decode kernels | +9..+20% (removes ~17.8 s of events; costs decode slowdown x duty + longer TTFT dead time ~2 s) | decode step on 128-f SMs (f~48) is <=1.25x while a W8A8 lane gets >=70 TF | the whole decode step (GEMMs, GDN, AR kernels, small ops) can run on ~80 SMs at <=1.25x with prefill traffic beside it; T2 measured x1.65 in-server on old kernels | attribute T2-style in-server slowdown per kernel family on the champion (which part is GEMM grid = fixable vs GDN/AR/small ops/DRAM-L2 interference); or chain bench with GEMM grids capped at 80 SMs + concurrent bs_gemm lane |
| 2 | Mixed-step host overhead -> capture more of the mixed step | +1..+4% (T51 sim: +2.5..5% with a captured pass and 3-5 ms glue) | small/mid mixed steps are host-bound by >=5 ms | the ~51 ms floor is host enqueue, not GPU | host-probe vs GPU-span per step class on the champion (T29_HOSTPROBE / T51_STEPLOG, ~3 GPU-min) |
| 3 | Asymmetric decode/prompt waves with link priority at 256-767 rows | +1..+3% | overlap >=0.6 of the decode wave and host not binding | the wave steps' host cost (117 ms eager) doesn't bind at these sizes | T47's host-free replay harness re-run with T52 link priority on (J vs P\|\|D at P=128/256/512) |
| 4 | k=4 on the champion | 0..+1% | row price above 64 <=0.115 ms | W8A8 gate_up + AR at M=80 cheap enough | in-server ABBA k=3 vs k=4 (needs kMaxLogTokens rebuild + 128K) |
| 5 | Drafter non-byte overhead fusion | +1..+1.5% (0.3-0.5 ms/step) | ~35 tiny kernels/level fuse | not already at its floor (KB80 said floor) | nsys of drafter levels on champion |
| 6 | Token-split threshold 128-255 with link priority | 0..+1% | halves at 64-128 rows hide AR | GEMM at M/2 doesn't cost ~2x (it does below the ridge: framework predicts loss under ~340 rows) | flag-only ABBA (~8 GPU-min) |
| 7 | Host embedding | +0.2% + 1.27 GB/GPU free (enabler for 2/3) | - | - | already measured (KB125) |
Pruned with reasons in section 3. Items 4-7 are each below +1.5%.

## 5. Where the framework didn't fit
- No host/CPU resource: launch-bound (eager) steps are a fourth resource here (T52: wave steps host 117 ms vs GPU 131).
- No SM-partition resource: `max(F, BW)` assumes free overlap; on this GPU the overlap is bought with SMs, and kernels
  sized for all SMs make it zero-sum (KB59). Item 1 is exactly this gap.
- Ridge uses spec FP8 rates; under the exact-accumulation rule the ridge is ~80 (bf16) / ~170 (e4m3) rows, and kernel
  m-tile steps make row cost a staircase (KB118).
- Link cost needs per-transfer latency and half-duplex sequencing (write then read) to match measurements.
- Closed-loop dynamics (arrival lumps, dead slot time, drain) are outside the window model; I used T51's simulator.

## 6. Test results so far
- #2 mixed-step host overhead: T57-001 (job 484). Mixed steps are GPU-bound: idle between steps ~0, and host wait inside steps is at most 1.5 s per 41 s (loose bound). Demoted.
- #1 spatial lane: T57-002 (jobs 486/488, microbench). f=48 gives lane 85 TF with decode 1.40x. The contention is DRAM/L2 arbitration, so my closed-loop gate fails (~+2% napkin). On T56's per-token metric it passes on the GEMMs alone (0.07-0.08 ms/token), so T56-001 in-server decides.
- #5 drafter: T57-003 (CPU anatomy). 1.90 ms/step, ~0.76 ms above the byte floor; realistic exact saving ~0.13-0.3 ms/step (+0.5-1%). Being built (task 4), with T56's fc-embedding table.
- Lane closed by T56-001 (job 489, in-server): GEMM-only cost per lane token is 0.137-0.156 ms vs ~0.167 for the joined mixed step; adding the lane's ARs, GDN work and the slot-wait term gives break-even or worse.
- T57-004 (drafter package v1) official #1 1225.5 tok/s (+1.08% vs T52-005; ABBA job 497 +1.68% all B > all A).
- T57-005 (target embedding in host memory): bitwise, frees 1.17 GiB/GPU, ~-0.05 ms/step. T57-006 (drafter v2: radix select + one-launch flush + HOSTEMB): exact, -0.1 ms/step on the probe, but the 6-start ABBA (job 510) is null (+0.01%). Not submitted.
- Wave capture to lower T30_WAVE_MIN (worker scoping, task 6): +0.2-0.5% realistic, days of work, risk of being host-bound: not built.
