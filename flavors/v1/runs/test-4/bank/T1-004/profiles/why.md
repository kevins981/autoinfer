# Why the chunked GEMM+AR overlap measured ~0 end to end (nsys, 64 prompts c=16, GPU 0)
OFF = leader-1.w1/prof_t1003 (job 17), ON = prof_on (job 27, VLLM_OVERLAP_AR_CHUNKS=4, reserve 2). Scripts: why.py, why2.py, ../leader-1.w1/anatomy.py.
Observed
1. Big prefill steps (cluster span >60 ms): OFF 52 steps, 9.87 s (37.6% of the 26.2 s span); ON 50 steps, 8.87 s (34.1% of 25.96 s).
   Biggest step: OFF 590 ms (Marlin 360.7, NCCL 182.3, 0 ms Marlin/NCCL overlap); ON 531 ms (Marlin 389.6, NCCL 196.9, 105 ms overlapped).
   Idle gaps inside steps unchanged (~3 ms). Marlin kernel time inside big steps +8% (389.6 vs 360.7), NCCL +8% (196.9 vs 182.3): both slower when run concurrently / on 126 SMs / at M=512 chunks.
   Napkin: ~25% of AR (2.09 s) + row-GEMMs (~1.7 s) ~3% of the span; measured prefill-step reduction 1.0 s = 3.8% of the profile -> the step-level saving IS there and matches the napkin.
2. Timeline of one layer (biggest ON step): the down_proj (K=8704, 4 chunks of 512) pipelines as designed: Marlin chunk i (~330+46 us) runs on stream 17 while AR chunk i-1 (~370 us) runs on stream 37; chunks are balanced.
   The last chunk's AR (~370 us) cannot hide (tail), and the o_proj/out_proj K=3072 GEMM chunks (~75-130 us) are far shorter than their AR (~370 us), so there the AR dominates (~4 x 370 us AR back to back, little to hide behind).
   Overlapped Marlin/NCCL time in the biggest step = 105 ms of 197 ms NCCL time (53%).
   Cross-GPU AR start skew (ARs > 100 us): OFF p50 190 us / p90 597 us; ON p50 52 us / p90 240 us (smaller, as each chunk sync is shorter; 6940 ARs vs 3564).
3. Whole profile (GPU 0 span): OFF 26.22 s, ON 25.96 s (-1.0%). Cluster time by class (s / count):
   prefill >60 ms: 9.87/52 -> 8.87/50 (-1.00 s);  30-60 ms steps: 9.48/282 -> 10.74/311 (+1.26 s);  5-30 ms: 5.08/214 -> 4.96/203;  <5 ms: 0.78 -> 0.40;  inter-cluster gaps 1.008 -> 0.995 s.
   So the 1.0 s saved in prefill steps was offset by 29 more mid-size steps (+1.26 s) in the ON run: step composition differs between runs (scheduling/arrival timing is not deterministic), the same size as the saving.
4. Decode steps (steady c=16 decode, ~960-1050 kernels): Marlin 18.46 -> 18.66 ms, host AR 5.88 -> 5.90, GDN same; step period 36.95 -> 37.36 ms (+1.1%), main-cluster span 32.99 -> 35.38 ms (the latter mostly cluster-merging differences: other 1.59->2.57, AllGather 0.53->1.03). Decode GEMM/AR essentially unchanged; period +0.4 ms is within profile-to-profile variation (decode kernels counts differ by cluster composition).
Interpretation
- The overlap works mechanically (53% of NCCL time hidden in the biggest step, prefill steps -10% span) but only the K=8704 down_proj pipelines well; the K=3072 GEMMs are too short to hide a 370 us AR chunk, the tail chunk is exposed, and both Marlin and NCCL slow ~8% when concurrent.
- Gain bound: prefill steps are ~35% of the span; ~10% off them is ~3.5%, but the net seen in bench (0% +-1%) and in the profile (1%) is eaten by run-to-run step-composition noise and possibly small TTFT-p50 costs. A same-job result at bench 400 is 687.2 vs 687.5 tok/s.
- Uncertain: whether the extra mid-size steps in ON are a real effect (e.g. interplay with scheduling) or noise; one profile each cannot say.
