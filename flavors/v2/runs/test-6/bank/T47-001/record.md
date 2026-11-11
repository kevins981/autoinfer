team:         T47
builds_on:    T45-005 (champion at measurement time; harness first written on T37-002 in workspace t47-waves, moved to t47-waves2 = T45-005 at the leader's request; T45-005 touches none of the harness files)
performance:  "unofficial, host-free replay: gate FAILS. Best J - P||D: -4.9 ms at P=64, -4.1 at P=128, +0.65 at P=256, +2.7 at P=512 (positive = waves faster); two DMA-AR waves overlap 0%"
summary:
  hypothesis: (proposal T47-1, stage 1) splitting a mixed step by PHASE (decode rows = wave D on a high-priority stream, prompt rows = wave P
              on the main stream) lets D fill P's exposed all-reduce gaps: J - P||D >= 8 ms at P~256, P||D < J at P~128.
  change:     test-only, default off (T47_WAVE_BENCH=1; no change to the serving path when off, except NL 2->3 DMA lanes, see validity).
              Ported from T30_WAVE_BENCH (workspaces/t30-wave, bank/T30-005):
              - vllm/t47_wave.py: step picker (decode-request prefix >= T47_WAVE_MIN_DEC=12, then prompt rows; one step per P bin ~64/128/256/512).
              - model_runner.py: forces the picked live mixed step eager and builds per-wave attention metadata (mamba_hybrid.t47_wave_attn,
                ported from T30's t30_wave_attn, cut at the request boundary).
              - model.py (_t47_bench): snapshots GDN/conv state + replay flags + touched KV blocks, captures each arm as its own CUDA graph,
                replays interleaved (2 warm-up + 12 timed reps, rotating order, CUDA events on the main stream), restores state, then the
                step runs normally. Included: all 64 decoder layers + final norm (and its AR). Excluded: embedding, lm_head, sampling, drafter.
                Arms:  J = the champion's eager forward on all rows (token-split chain + ping-pong where it applies; graph J == eager J bitwise)
                       P = the same code on the prompt rows only (their own metadata)
                       D = decode rows, plain per-layer path, every AR on a NEW DMA lane 2
                       Dh = decode rows through the champion's own path (64 rows -> no split -> SM host-staged AR) = what a decode graph runs
                       PD1 = one graph, two branches (D on a hi-prio stream inside the capture)
                       PD2hi / PD2lo = P graph on main stream + D graph launched on a high / normal priority stream, event join
                       PDh = P graph + Dh graph on the hi-prio stream (only when P >= 256: then P uses DMA lanes 0/1 only and never the host AR)
              - dma_staged_ar.cu: NL 2 -> 3 (lane 2 for wave D, 4 MB tmp).
              D does NOT reuse the server's captured FULL decode graph (its buffers/metadata are tied to the runner); Dh is the same per-layer
              code path a decode step runs, captured fresh.
  results:    Observed (job 427, T45-005, 15 decoding requests at k=3 = 60 decode rows, random-token prompts of exactly P tokens; rank 0;
              median of 12, p10-p90 spread <= 0.6 ms everywhere; raw lines in agents/leader-47.w1/s2/server.log, grep T47):
              | P   | J      | P      | D(lane2) | Dh    | PD1    | PD2hi  | PD2lo  | PDh    | J - best P||D | overlap(best)         |
              | 64  | 38.34  | 22.03  | 29.16    | 21.79 | 50.83  | 43.19  | 43.15  | n/a    | -4.85 (PD2)   | 0.36 (PD2, vs D)      |
              | 128 | 48.41  | 37.94  | 29.24    | 23.36 | 64.39  | 52.53  | 52.45  | n/a    | -4.12 (PD2)   | 0.50 (PD2, vs D)      |
              | 256 | 73.14  | 59.03  | 28.96    | 23.44 | 87.83  | 88.45  | 88.74  | 72.49  | +0.65 (PDh)   | 0.43 (PDh, vs Dh); PD2 0.00 |
              | 512 | 110.45 | 96.86  | 29.05    | 23.48 | 125.51 | 126.88 | 127.13 | 107.74 | +2.71 (PDh)   | 0.54 (PDh, vs Dh); PD2 0.00 |
              overlap = (P + D - PD) / min(P, D), with D = the decode arm used in that PD.
              Job 423 (same code minus PDh/depth/trace, same champion family) reproduces every J/P/D/PD2 number within ~2.5 ms
              (P=128: PD2 overlap 0.51; P=256/512: PD2 overlap 0.00; Dh 21.8).
              J - P (marginal GPU cost of the 60 decode rows inside the joined step): 16.3 / 10.5 / 14.1 / 13.6 ms at P=64/128/256/512.
              Stream priority makes no difference (PD2hi == PD2lo within 0.3 ms; priority range (0,-3)).
              PD1 (one graph with two branches) never overlaps (0.01-0.10) while PD2 (two graph launches) does at P<=128.
              Trace (torch.profiler, one replay, P=256): PD2hi span 102 ms, any-activity 85.4, >=2 concurrent activities 35.4 ms;
              PDh span 74.5, any-activity 72.9, concurrent 51.0 ms (agents/leader-47.w1/s2/trace_*.json, profiler-inflated).
              Crossover: no P in 64..512 where P||D <= J with the DMA decode wave; with Dh, P||D < J from P~256 on (by 0.65 / 2.7 ms).
              Numerics (eager, same restored state, vs J): final hidden states of the waves differ from J: decode rows rel-norm 0.16-0.24
              (max abs 8-29), prompt rows 0.037-0.051; D(lane2) == Dh to the bit. Depth check (waves vs J after L layers): L=1 rel 1e-4..6e-3
              (P=512 prompt rows bitwise), L=4 0.003-0.023, L=16 0.012-0.042, growing steadily with depth.
              Estimate before measuring (proposal): P=256 T_J 81 -> T_S ~65 (-16 ms); P=64 -5 ms; crossover P~30-40.
  conclusion: Interpretation:
              - The proposal's premise is off by the decode rows' marginal cost: inside the joined step they cost only 10-16 ms (J - P),
                not ~23 ms; run alone they cost 22-29 ms (they re-stream the weights). To win, P||D must hide more than D - (J - P)
                ~= 8-13 ms of D, i.e. an overlap of ~0.45-0.6 with Dh. Measured: 0.43-0.54. Break-even at best, not 8 ms.
              - Two DMA-AR waves do not overlap at all (P >= 256): kernels are concurrent ~40% of the time, yet PD = P + D. Most likely the
                waves' copies share the same copy engines, so the latency-bound D chain (128 small ARs) queues behind P's large ones
                (not verified per copy engine). A decode wave on the SM host AR (Dh) is both faster alone (23 vs 29 ms) and overlaps ~50%.
              - At P <= 128, P itself uses the host AR, so Dh can't run beside it without a separate host-AR instance; even the D(lane2) wave
                with 50% overlap loses 4 ms.
              - Priority doesn't matter; launching D as a separate graph does (one-graph branches serialized here).
              - Numerics: depth growth from ~1e-3 at layer 1 is consistent with M-dependent rounding compounded over 64 layers, not a
                metadata bug, but the decode-row final diff (0.2) is large; a greedy token-identity check would be needed before any
                in-server use. Not done (no stage 2).
              - Gate (J - P||D >= 8 ms at P~256 AND P||D < J at P~128): FAILS on both counts (+0.65 ms; +4.1 ms worse).
validity:     (worker leader-47.w1, for the leader's review) test-only code, env-gated (T47_WAVE_BENCH, default 0). With the flag off the only
              behaviour change is the DMA AR extension's NL=3 (one more lane: +4 MB device tmp per GPU, +48 MB pinned shm; lane 2 is never
              called), plus inert imports. Do not promote this diff as a serving change.
files:        harness s1.sh + drive.py here; logs agents/leader-47.w1/s1_job423/, agents/leader-47.w1/s2/ (job 427); workspace t47-waves2.
GPU:          job 423 2.7 min, job 427 2.8 min (5.5 of the 8-min allowance).
