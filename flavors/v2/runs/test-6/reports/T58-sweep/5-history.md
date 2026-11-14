# T58 sweep, step 5: candidates against the run's history

Phase 2. Read: shared_kb.md (KB1-KB135), graph.md, journal.md (searched), the reports/ write-ups (searched),
bank records cited. Files 1-4 are unchanged after this point. Corrections to them are here.

Legend:
- **built**: a record exists with the change.
- **tested**: measured, record or KB cited; "on what code" given where it matters.
- **untested**: no measurement found.

## Candidates of 4-knobs.md

| # | Candidate | Status | Evidence, and whether it still holds on T52-005 |
|---|---|---|---|
| W1 | Lower-latency decode AR (my "a = 25 us fixed") | **tested, closed** | KB109 (T36): the ~43-45 us AR is 17.4 us writing + 2-3 us flag + 16.5 us reading + 6-8 us prefetch tail. So my intercept is not protocol latency: kernel host traffic moves ~22 GB/s per GPU **in total** whatever the direction mix (KB109, KB115/116). Closed transports: LL fence-free AR (KB78), copy-engine/hybrid (KB74: CE start ~20-35 us inside graphs), pacing (KB116), cut-through (KB109). All are capped by the prefetch coupling (KB78: credit given back). A full-duplex AR is worth ~3.5% but needs GDRCopy/root (KB130). On T37/T45-era code, same AR kernel as T52-005: holds |
| W3 | Decode GEMMs that don't need the AR's prefetch | **tested** | KB28/39/66/68/81/83/98/101/108: many kernels tried. The t31 register-direct kernel is the winner and is the champion; run-ahead streaming loses (KB108). My microbench's "60-70% of DRAM cold" used a WRITE flush; KB76 shows write flushes inflate cold times ~20%. **Correction to 3-calibration §1:** read-flushed cold decode GEMMs are at ~0.8-0.92 TB/s (KB108). |
| P6 | Column-streamed AR under the producing GEMM | **tested** | KB24 (T1-018), KB72 (T13), KB132 (T53): saves 3-12 us per boundary vs a 15-25 gate. ≤2%. Closed |
| P5 | Row-split DualPipe at decode | **tested** | KB42, KB132 (two offset half-batches 1.31-1.47x the chain) |
| G1 | FP8 MMA with f16 accumulate in prefill | **tested, dropped** | journal 2026-11-10 (T38, job 386): f16 K-block partials overflow on essentially every output under checkpoint scaling; kernel gain only 1.1-1.5x. Question withdrawn before a ruling. Also KB69: the ".f32" FP8 accumulator keeps only ~13 bits, which I did not know |
| R1 | add+norm+quant fused into the AR epilogue | **tested** | KB56 (T1-043, -0.6%), KB132 (T53: whole-row fused AR+norm+quant+qkvz slower). Closed |
| R2 | Lossless FP8 weight coding | **tested** | KB47 (entropy 0.818), KB49, KB102 (0.8925 code wins cold, loses warm; ≤4.8% even with free decode). Closed |
| K4 | Shared-prefix keep (system prompt) | **built** (T21-001/002) | KB86: hits ~17.6K tokens/trial but +0.5% only. Small mixed steps sit at a floor, so removing tokens per request is nearly free; my "18K x 150 us" napkin used the slope, the same mistake KB86 warns about. Not in the champion |
| K2a | Lazy checkpoint cap 8 -> 16+ | **untested** (cap 8 built in T1-035, KB48, +1.0%) | GDN replay kernel was later found latency/LSU-bound, not HBM (KB51), then RED8/mode 5. KB134: the dense fp32 read of 38-40 us/layer is what's left (~0.7 ms/step). Fewer state writes are an untested sub-part. The writes partly hit L2 (KB48: microbench 0.68 vs in-server 0.50 ms) |
| S1 | k=4 at full batch | **tested on old code, re-open noted** | KB63/95 (M=48 era: loses). KB127: on T45-005 (gate_up W8A8 at 48<M≤96) uniform k=4 is "plausibly break-even to ~+1%"; not measured. Needs kMaxLogTokens 4->5 and kLogCap (my reading of the kernel: kMaxLogTokens=4, kLogCap=8). T56's ranking lists k=4 as not worth testing |
| S2 | Load-adaptive depth in the drain | **untested** | Costed ~0.8% (T10, local_kb/leader-10). KB69: drain 2.3% of slot capacity. KB85: adaptive/ragged-K plumbing has cost -8.5% before |
| S3 | Confidence-based dynamic slots / trees | **tested** | KB63/85 (budgeted-K loses on plumbing), KB109 (≤+2-4% ceiling), KB112/135 (trees -6..-8% with measured costs). Closed for now |
| S4 | Retrained drafter | **tested** | KB60 (T3): fine-tunes give no held-out gain |
| P1 | PP=2 | **tested** | KB112 (T38), KB123 (T49): closed |
| P2/P3 | Replicated mixer/attention | **costed** | KB75 (T18: "DP mixers" rejected with numbers), KB126 (lending KV to replicas ~1-3% at best, and KV is needed by GDN spec pages) |
| P4 | Asymmetric TP split | **tested on the old clock state** | KB82 (uneven split ABBA -0.4/-0.7%), KB113 (GPU0's clock lock is gone after the reboot). My measurement (ratio 0.99-1.01, microbench GPU0 = GPU1 within 0.5%) is on the current state and agrees: nothing to balance |
| P7 | Piggyback prefill rows on decode passes | **tested** | KB64 (T6: -30% at budget 96), KB108 (T35: budgets 128/192 -19%/-2.6%), KB118/120/131 (T45/T51: a row in the decode pass ~0.14-0.16 ms vs 0.166 in a prefill event; budgets 192/160 lose 1.9%/10.2%). Closed with today's kernels |
| W6 | GPU idle | **tested** | KB46, KB61, KB89: GPU never idles; my GPU-side measurement (98.0-98.4% busy) agrees |
| W7 | Wave steps / big mixed steps | **built** (T52-005 is the champion's own last change) | KB133: lowering the wave threshold is limited by host time (~85-117 ms per eager wave step); capturing cores per wave size is open (~+1%, memory-limited at 0.45 GiB free) |
| M1 | AR-window L2 prefetch for 97-255-row hostar ARs (`VLLM_HOSTAR_L2PF_MSG_MAX`, 1 MiB) | **untested** | KB131 (T51) names the mechanism ("all GEMMs leave t31b and lose the AR-window prefetch"). It doesn't record raising the size gate: no record, KB or journal entry mentions MSG_MAX after T1-019 set it. On T52-005 the gate is still 1 MiB (my code read) |
| M2 | Mid-M (97-255) GEMM path | **tested (configs), open (kernel)** | KB130 (T50-001: config table at its best; "mid-M slack needs a kernel change"), KB131 (ncu: L2 55-70%, math-pipe throttle ~74% of the e4m3 roof). No new kernel built |
| M3 | GDN prefill glue in non-wave mixed steps | **tested** | KB107 (T32 captured core: host -33 ms, GPU span unchanged, +0.29%). My Task-5 numbers (741 eager torch launches, 1.3 ms GPU) are GPU time that capturing alone doesn't remove. KB131's sim: a fully captured mixed pass with 3 ms glue = +5% (unbuilt, ~40-60 GPU-min) |
| M4 | Mixed-step AR per-row cost | **tested** | KB91/104/116: at the PCIe floor (latency/bandwidth), ~0.56 us/row/AR |
| L1 | 4-bit lm_head screen | **tested-adjacent** | KB52/121: lm_head is a ~1% lever; the 7-bit plane is already the champion |

## Things the history shows that my sweep missed (findings about the framework)

1. **FP8 MMA accumulator precision** (KB69): the ".f32" e4m3 accumulator keeps ~13 bits below the leading addend.
   "F" depends on the accumulation semantics, and that interacts with the contract. The framework treats F as one number per dtype.
2. **Capacity coupling of speculation depth to state pages** (KB13/17/110): GDN spec-state pages scale with k and share
   the 5 GiB pool, so k=3 at 16 requests needed the two-slot state.
   The framework has capacity C, but my sweep didn't price capacity per knob setting (k).
3. **NUMA placement of pinned host buffers** (KB37) and **PCIe duplex behaviour of SM-driven host traffic** (KB5, KB73, KB109, KB115):
   - the "link" here is a host-memory tier reached by SM loads/stores or copy engines, with different duplex and latency per mechanism;
   - the framework's single Link_BW cannot express this.
   My "fixed 25 us latency" reading was wrong for this reason: it is a ~22 GB/s total cap on kernel host traffic, half-duplex in practice.
4. **Copy-engine start latency inside CUDA graphs** (~20-35 us, KB74), which differs from my standalone 3 us memcpy intercept.
5. **Scheduler lifecycle effects** (KB75/77): slot refill at N+3, TTFT dead time. No knob in the framework covers request admission.
6. **Tie-break exactness of greedy verification** (KB12): a correctness detail outside the framework.
7. **Microbenchmark hazards**:
   - write-flush inflation (KB76);
   - persisting L2 lines (KB103);
   - unbuilt workspaces (KB61);
   - host-side step attribution (my own Task 3 to Task 4 correction; also KB46/58/61).
   These are measurement layers, not structure, but they decided many of the run's conclusions.
8. **Interference is zero-sum on this stack** for spatial/temporal overlap of comm and compute (KB59/62/70/132). Exception: DRAM read saturates from few SMs (KB129).
   The framework's Window rule max(F, BW, Link) assumes concurrency for free; devil's-advocate point 3 in framework/07 is the operative one here.

## Corrections to files 1-4

| File | Claim | Correction |
|---|---|---|
| 3-calibration §1 | decode GEMMs at 60-70% of DRAM cold | write-flush artefact (KB76); read-flushed ~0.8-0.92 TB/s (KB108) |
| 3-calibration §1 | AR = 25 us fixed latency + 24 GB/s | AR ≈ write then read at ~22 GB/s total per GPU + ~2-3 us flag + prefetch tail (KB109). At R=4 the 29 us is mostly the prefetch tail and the fixed path |
| 4-knobs W1 (rank 1) | +4-8% | closed; the remaining full-duplex idea needs root (KB130) |
| 4-knobs G1 (rank 2) | +4-6% | dropped (overflow; T38 job 386) |
| 4-knobs K4 | +1.2-2.7% | +0.5% measured (KB86) |
| 4-knobs P7 | pending | closed (KB131) |
