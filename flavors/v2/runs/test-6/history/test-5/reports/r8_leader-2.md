# Round 8 report from leader-2 (T1): per-start bimodality
Best: bank/T1-026 = T1-025 + NUMA self-bind of TP workers. Official 880.9 tok/s (878.4/883.4, CV 0.4%), +0.7% vs T1-025 874.8
(whose trials were 863.7/885.8, CV 1.8%), +38.5% vs BASE; GSM8K 94.0%, TTFT p99 1312, TPOT p99 34.3, acceptance 2.331, gates + 128K
pass. Both official trials logged "bound to NUMA node 0". Propose as champion.

Experiments:
- T1-026 (w14): cause of KB36's bimodality = NUMA placement of the pinned host-AR/DMA /dev/shm slots. Both GPUs hang off node 0;
  numactl is missing, so vLLM's --numa-bind/auto-bind never runs; rank 0 first-touches the slots in cudaHostRegister on whatever CPU
  it runs on -> slots wholly on node 1 in 7/10 unbound starts. Node-1 starts 754.8 vs node-0 767.7 tok/s (+1.7%, 110 prompts); the
  only variable tracking it is prefill-size AR cost/token (+27% on node 1); decode period, decode ARs, clocks, output tokens identical.
  Fix: workers pin all threads to the GPU's node CPUs + libnuma preferred memory at init (opt-out VLLM_NUMA_SELF_BIND=0; skips when
  --numa-bind is used or affinity is constrained). 4/4 bound starts on node 0 at 767-769.5. (w14 lost 9.8 GPU-min to a script bug.)
- w15 microbench (in T1-027): the R7 "12 GB/s in-chain DMA copies" (KB35) don't reproduce: jobs 88/92 had a job-wide host-traffic
  slowdown (hostar also ~2x slower) -> R7-A's -13%/layer gate was vs a slowed baseline (today ~-7-8%/layer). Node-1 slots slow copies
  only when both GPUs write at once (D2H 26 -> 14.7 GB/s); hostar alone at M=300 229 (node 0) vs 315 us (node 1).
- T1-027 (w15, measurement on T1-026): split 256 841.1 vs split off 824.5 (+2.0%) vs split 128 833.7 (-0.9% vs 256); within-arm
  spread <= 0.4%. R7-A pays clearly once slots are near; keep 256.

Proposed SHARED_KB:
- KB-a: KB36's bimodality was NUMA placement of the pinned AR/DMA slots (first-touch by an unpinned rank 0; numactl absent so
  vLLM's auto-bind never ran): 7/10 starts on the far node, prefill-size ARs +27%/token, throughput -1.7%. Workers now self-bind to
  the GPU's node (T1-026): official trials 878.4/883.4 (CV 0.4%). (T1-026)
- KB-b: KB35's "12 GB/s in-chain DMA copies" was a job-wide host<->GPU slowdown in those jobs (hostar also 2x slower; cause outside the
  code). Near-node copies run 23-26 GB/s; far-node slots hurt only when both GPUs write at once. Microbench numbers involving host
  traffic must be compared within one job. (w15, T1-027)
- KB-c: With near-node slots the R7-A token split is +2.0% same-job (split off 824.5 vs 841.1); threshold 128 is worse (-0.9%). (T1-027)

Left open: R7's small items (no-op flush launches ~0.45 ms/mixed step; replay kernel 46 vs 25 us in mixed steps: ~1 ms/mixed step
together, ~0.4%); split threshold 384; whether the official harness trials of earlier records were hit by the far-node mode.
Next (one direction): the biggest remaining lever left on the decode step is Marlin when weights are warm in L2 (KB28: tensor-pipe
bound, 1 warp per scheduler, ~65% of floor; up to ~1-1.5 ms/step, ~3-5%). It is a multi-day hand restructure of Marlin's decode
path (2 warps/scheduler with leaner smem stages), exact up to fp32 reorder. With 7 rounds left there is time for a 2-round effort:
round 1 a standalone kernel prototype at M=48 for gate_up/in_proj (go if warm >= 15% faster at equal cold), round 2 integration.
Smaller, cheaper alternative if that's too risky: the R7 mixed-step leftovers (~0.4%).
Notes: GPU 52.6/80 (w14 26.5, w15 15.4, eval 10.8); 27.4 unused. No GPU contention. Workers stopped. Workspaces kept: leader-2.numa
(T1-026), leader-2.w15abc, leader-2.bimodal, leader-2.dmacopy (unused).
