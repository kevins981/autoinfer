# leader-30 (T30, explore) notes

## Cards (first set)
Per-item overhead / timing / coordination made me look at the per-layer JOIN in the champion's token-split
chain: both chunks meet at every GDN/attention core, so the post-core AR has nothing to hide behind (coordination
cost created by dividing the work). Imbalance: in big mixed steps the tensor pipe and PCIe are both busy but in
series. Hardware cards (no P2P, unified memory, texture units) gave nothing directly usable.

## Orientation (2026-11-10, CPU only)
- Probe job 219 (T29-004, 300 prompts, untraced, agents/leader-29.w2/probe/run/probeB.rank0.jsonl), GPU time/step:
  decode 1767 steps 24.8 ms (43.3 s = 64%; of which nreq<14 drain 5.5 s); mixed 24.2 s (36%):
  pt1-80 83x48.3, pt81-208 74x56.0, pt209-400 47x92.7, pt401-1000 72x146, >1000 4x288 ms.
- Marginal cost per prompt token in big steps 0.161 ms (314->644->1522 pt). bs_gemm marginal ~0.10 ms/token
  (summary2.md M=512->1024), cuBLASLt per-tensor 0.084, host AR raw 0.072 (128 x 0.56 us), misc ~0.03.
- Inference (to measure): big steps have ~25-30 ms of exposed AR; at pt209-400 the split also costs ~8 ms of GEMM
  (two M~180 GEMMs vs one M~360) and hides ~nothing.
- Mid-M bs_gemm is 45-70% above roofline max(bytes 12.9 ms, compute M*0.084 ms) per forward at M=96-384.

## Direction candidate: two-wave layer-pipelined mixed steps
Waves progress independently through layers (wave 1 = continuation of wave 0 for split sequences: needs wave 0's
GDN state / KV at the same layer, which it has, since wave 0 runs ahead). No per-layer join, so each AR overlaps
the other wave's core/MLP. Payoff estimate (probe composition): pipeline alone ~3.5 s/67.5 (+5.5%); + mid-M GEMM
nearer roofline ~+2.5%.

## Measured (jobs 235, 244; agents/leader-30.w1/ar_floor/RESULT.md)
- T29-006 probe, 200 prompts: big-step fit 0.160 ms/token + 39.5 ms (n=83). AR-off (T30_NOAR=1, --ignore-eos, 60 prompts):
  0.109 + 23.1 (n=24). Class medians A->B: decode 24.9->21.8, pt1-80 43.7->35.2, 81-208 49.4->35.9, 209-400 93.6->62.1,
  401-1000 138.6->95.7, >1000 293->177.
- Process lesson: a timing-only arm with garbage outputs ends at EOS at once, and the bench then retries until the job cap
  (job 235 lost 15 GPU-min). Use --ignore-eos plus a fail-fast curl check. The bench counts delta.reasoning as a chunk.
- Proposal T30-1 sent (65 GPU-min requested).

## Attribution (w1, jobs 249/254, agents/leader-30.w1/attr/RESULT.md) -> T30-002
- pt401-1000 rank 0: span 141.8 = kernels 94.8 + copy-only 42.6 + idle 3.9 (matches AR-off -43). Mixer AR ~55-65% of the
  exposure. The two chunks run in lockstep: both out_projs at once, then both ARs at once over a shared PCIe (no stagger).
- AR fill/drain: D2H-alone ~190 us + H2D-alone ~220 us of each ~0.8 ms AR (VLLM_DMA_PIPE=4 pieces). Testing 8/16 (w1 task 3).
- Split off (arm C): pt401-1000 155 vs 139 (the split saves ~16 ms there); pt209-400 is flat.
- Host execute_model 75-87 ms at pt 300-900 = 85% of GPU at pt~350, 45% at pt~900: the host is near binding for small big-steps.

## Prototype (w2, workspaces/t30-wave; jobs ~269-289)
- Correctness: layer-0 stages within GEMM-M noise vs full rows (1e-3..1e-2); generation coherent. Not bitwise (ba GEMM M-dependence).
  The earlier "0.98 rel" was a capture artifact (the PINGPONG chain adds out into residual in place).
- Job 289 (120 prompts/arm): W0 972.9, W2 (eager, joined) 958.4, W1 (waves) 900.1 tok/s. Eligible steps 41/955.
  The wave step is HOST-BOUND: host enqueue 183-199 ms vs GPU span 161-174, flat in size. Eager alone: host 134-142 vs graph 69-81.
  The second core per layer adds ~0.8 ms host each.
- Failure class: this attempt failed (host path), not yet a hard limit. Next: host-free GPU timing of the schedule (w2) and
  a host-cost breakdown of the eager GDN core (w1). If the schedule doesn't hide AR on the GPU either, stop.
- Host-free (whole-forward graph replay of one eager step, T30_WAVE_BENCH): joined W2 vs waves W1 vs AR-off:
  n=527 108/~102/72, n=734 130.5/123.6/90, n=1010 176.5/170.6/122-128. Waves recover only ~6 of 36-54 ms exposed AR.
  The schedule does not hide the AR even with a perfect host. Cause test (nsys of the replay) pending, then close/pivot.
- w1: lean GDN core host path costed (~1.24 -> ~0.21 ms/call: frozen Triton launchers + glue cut), agents/leader-30.w1/host/RESULT.md.
- k waves host-free n=734 (with AR / AR-off): k2 123.6/87.4, k3 116.2/90.2, k4 ~124/92.7. Fails the floor+10 line.
  Cause: CE AR latency chain 0.3-0.5 ms/message > per-wave cover. Recommended closing; write-up reports/T30-writeup.md.
