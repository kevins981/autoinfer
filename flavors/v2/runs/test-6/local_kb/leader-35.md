# leader-35 (T35 moonshot) notes
## Napkin (assumed, from specs; verify)
- 27B FP8 ~27GB weights; TP=2 -> ~13.5GB/GPU/step -> ~14 ms at ~950 GB/s.
- Total compute of bench: ~104k prefill tok + ~117k verify tok-forwards x 54 GFLOP ~ 12 PFLOP; 4090 bf16(fp32acc) 165 TF dense, fp8 330 TF per GPU.
- Champion: 1088.9 tok/s, ITL p50 24 ms (per decode step), acceptance 2.33 -> ~29 steps/s average incl prefill.
## Inspiration cards (first set)
- Predictability/overqualified work -> speculation depth/tree; per-item overhead -> 128 ARs per step; representation -> nothing yet.

## Measured (my jobs)
- job 352: nsys node profile of champion T31-003 (agents/leader-35/prof_champ_node; kps.py, ar_by_m.py).
  - decode step M=48: 23.3 ms busy (period 24.3 under nsys). hostar AR 6.0 ms (137 calls), Marlin gate_up 6.0, t31b 6.6, GDN replay 1.7, lm_head 1.0.
  - decode step vs M (busy): M=3 19.0, 24 20.2, 48 23.3 ms -> ~80% fixed per-step cost.
  - hostar median 34 us even at M=3 (30 KB): its duration is dominated by the fused L2 prefetch of the next GEMM weights, not PCIe.
  - per GDN layer ~345 us of kernels for 191 MB weights (+state) vs ~230 us if DRAM never idled.
- job 353 (200 prompts, fresh servers, same job): max-num-batched-tokens 2048 -> 1069.0, 192 -> 1041.1, 128 -> 870.5 tok/s.
  Step counters: prefill marginal ~0.124 ms/token (big chunks) vs ~0.26 (small chunks). Prefill adds ~16% to wall time (not 38%: nsys-inflated).
  => naive "dribble prefill into decode steps" is dead as a flag; mixed steps carry big fixed cost (GDN chunked path, ping-pong).
- CPU: FP8 weight byte entropy ~6.55 bits (exp entropy 2.6 of 4 bits) -> lossless weight compression max ~18% of weight bytes (~+9% overall upper bound). Kept as a fallback idea.
- PCIe link reads gen1 when idle (gen4 max, x16).
- job 355: MTP k=3 1041.3 tok/s (acc 2.687, cum 0.765/0.536/0.386) vs k=2 1069.0 (job 353): screen negative. k=4 init failure. Bank T35-002; budget sweep T35-001.
- Champion already fuses L2 prefetch of the consumer weights into EVERY hostar AR (hint mode; its comment: ~40% lands; ld mode needs many SMs).
- Proposal 1 (proposals/T35-1.md): streaming-bound decode (run-ahead weight streamer / persistent layer kernel / lossless codec). Sent to manager, awaiting answer.
## Spar T35-1: beliefs
- naive prefill absorption: refuted by job 353. Tree-speculation benefit (branch coverage at rejection points): untested, deprioritized because verify marginal cost is high (k=3 negative).
## Cards: per-item overhead (kernel boundaries/AR count), representation (weight entropy coding), fetching without waiting (run-ahead streamer) all fed proposal 1.
- Milestone 1 (bank T35-003, worker jobs 358/359): KILL. Cold in-chain GEMMs are DRAM-bound (0.8-0.9 TB/s); concurrent streamer only slows them. Only idle-DRAM windows = AR/small ops (~50 us/layer), already used by champion's in-AR prefetch. Remaining layer gap = AR latency (comm), not streaming.
## Spar T35-1 (turns 4-6, after closing)
- Streamer refuted (supported). Residual of V0 chain: ~15 us/layer small DRAM-idle ops (~1 ms/step gross, fusion, untested recovery), GDN ~9 us above byte floor, AR latency.
- AR opportunity: matched chain gives 42.7 us/layer (~2.05 ms over 48 GDN layers) going from 39/50 to 12 us ARs; my 2.5-3 ms was overstated because the prefetch must still happen somewhere.
- Still open (not tested): tree branch coverage at rejection points; k=4 (init failure).
