# T30 write-up (leader-30, explore): two-wave layer-pipelined mixed steps. Closed, below the bar

- **GPU:** 69.4 of 85 granted (seed 20 + start 65); 15.6 returned.
- **Records:** T30-001..005.
- **Workers:**
  - leader-30.w1: attribution, DMA pieces, host-cost costing.
  - leader-30.w2: the wave prototype, the host-free replay harness, k waves.
- **Proposal:** proposals/T30-1.md (payoff if it works +8%, central +3%).
- **Champion during the team:** T29-006 1062.1, then T31-003 1088.9.

## The problem (measured; still open for anyone)
- Big mixed steps (pt >= 209, ~19% of GPU time) carry ~30% exposed TP all-reduce.
  - AR-off floor (T30-001, timing only): pt 209-400 93.6 -> 62.1 ms; pt 401-1000 138.6 -> 95.7 ms.
  - Marginal cost per prompt token 0.160 -> 0.109 ms.
- The champion's token-split ping-pong (T9 S3c) runs both chunks in lockstep (T30-002, nsys jobs 249/254):
  - both out_projs run at once, then both chunks' copy-engine ARs share the link with no kernel running;
  - copy-only time at pt 401-1000: 42.6 of 141.8 ms;
  - ~55-65% of the exposed time is after the mixer out_proj, the rest after down_proj;
  - turning the split off costs +16 ms at pt 401-1000 and nothing at 209-400.

## What we tried
1. **Finer DMA pieces** (VLLM_DMA_PIPE 4 -> 8/16; NP cap raised; bitwise) (T30-003, job 264):
   - 8 is flat; 16 is +7..14 ms worse;
   - per-piece overhead is ~0.2 ms per layer-AR;
   - H2D is already flagged per piece.
2. **Two waves with no per-layer join** (T30-004, workspaces/t30-wave, `T30_WAVE`):
   - Design: wave B's core waits only on wave A's core of the same layer; the GDN 64-aligned continuation goes through the cache; metadata per wave via the V2 runner's slicing; DMA lanes per wave.
   - Correct to GEMM-M noise: layer-0 stages 1e-3..1e-2 rel vs full rows; mq bitwise; `ba` M-dependent; generation coherent.
   - In server it is host-bound: 900.1 vs 972.9 tok/s, same job (-7.5%).
     - The wave steps run eager, and eager alone costs host 134-142 ms vs 69-81 ms graph.
     - The second core per layer adds ~0.8 ms host.
3. **Host-free replay** (T30-005; `T30_WAVE_BENCH`: whole eager forward of one live step captured as one CUDA graph, timed on GPU only). Joined / waves / AR-off, ms:

   | n | joined | waves | AR-off |
   |---|---|---|---|
   | 527 | 108 | ~102 | 72 |
   | 734 | 130.5 | 123.6 | 90 |
   | 1010 | 176.5 | 170.6 | 122-128 |

   - Two waves recover only ~6 of 36-54 ms.
4. **Cause** (trace of the replay, n=734): no contention and no dependency bug.
   - Copies run 20.7 GB/s beside kernels vs 21.3 GB/s alone; GEMMs are not slowed in the wave arm (143 vs 145 us). They are in the joined arm (217 vs 142).
   - The 33 ms of idle sits in 128 gaps (~200 us each): both waves wait on AR results at once.
   - One CE AR's latency chain (D2H -> peer flag -> H2D -> add) is ~0.3-0.5 ms per ~1 MB message. The other wave has only one GEMM (~0.1-0.2 ms) of cover per AR.
5. **k waves (the variant aimed at the cause; NL 2 -> 4 DMA lanes), n=734 host-free, with AR / AR-off, ms:**

   | k | with AR | AR-off |
   |---|---|---|
   | 2 | 123.6 | 87.4 |
   | 3 | 116.2 | 90.2 |
   | 4 | ~124 | 92.7 |

   - k=3 is best: -14.8 ms vs joined (-11% of the step), still ~26 ms above its floor.
   - k=4 is worse: smaller waves give less cover per AR, and the M/k GEMMs raise the floor.

## Why we stop (classification)
- **Hard-ish limit of this design on this machine:** each CE AR's latency (~0.3-0.5 ms per message) exceeds the compute any wave can offer between its own ARs. The link carries raw bf16 (~1.9 GB per step at n=734), and the link time is ~2/3 of the compute time.
  - More waves trade cover against M/k GEMM efficiency and k x host work.
  - The best (k=3) recovers ~15 ms/eligible step host-free => ~+2% tok/s, before paying the host fix.
  - That is below the 5% bar (1143.3 vs T31-003).
- **Host, for anyone running mixed steps eager or with per-wave cores:** the eager GDN core costs ~1.24 ms host/call, and 91% of that is Python, Triton launcher and torch glue. w1 costed a lean path at ~0.21 ms: frozen-config direct `CompiledKernel.run` launches for the 10 Triton kernels + a glue cut; pin list in agents/leader-30.w1/host/RESULT.md. This is not measured. It is only needed if a design runs the core more than once per layer.

## Open (for later teams)
- **Fewer bytes on the mixed-step link: the only lever left for the exposed mixed AR.**
  - Idea, costed on paper only: RS + norm/quant on own rows + AG of the FP8 activations + scales. The next W8A8 GEMM consumes FP8 anyway, so this is bitwise to the W8A8 path. It cuts link bytes ~17-24% vs raw bf16.
  - A lower AR latency per message would also help (the ~0.3-0.5 ms chain).
- **The 3-wave schedule** (+15 ms/step host-free) plus the lean core: ~+2%, an exploit-sized item; the code is in t30-wave (`T30_WAVE_K`).
- **Tools:**
  - the host-free whole-step replay harness (`T30_WAVE_BENCH`): a clean GPU-only timing of any mixed-step schedule;
  - the untraced per-step probe on T29-006 (workspaces/t30-probe);
  - `T30_NOAR` (an AR-off timing floor; use `--ignore-eos`, or the bench hangs: job 235 lost 15 GPU-min).

## Forecast vs outcome
- Forecast: payoff if it works +8%, central +3%; the assumed part was "the fraction of the exposed AR a 2-wave pipeline hides" (100% / 60%).
- Measured: 17% host-free (2 waves), 36% at best (k=3).
- My error: I took T9's microbench overlap (86-96%) as the model and did not cost the AR's per-message latency chain against the cover per wave.
- What held up: the exposed AR (~30% of big steps) and the size of the prize.

## KB candidate
- Mixed-step CE AR on the W8A8 champion is latency-bound, not contention-bound.
  - Copies keep ~21 GB/s beside FP8 GEMMs, and the GEMMs are not slowed (wave arm).
  - But one AR's chain (D2H -> flag -> H2D -> add) is ~0.3-0.5 ms per ~1 MB message, which exceeds the compute any wave has between its own ARs.
  - Host-free replay at n=734 (joined / k=2 / k=3 / k=4, with AR): 130.5 / 123.6 / 116.2 / ~124 ms against AR-off floors 88-93.
  - The eager GDN core costs ~1.24 ms host per call (91% Python/Triton launcher/glue).
  - (T30-002..005)
