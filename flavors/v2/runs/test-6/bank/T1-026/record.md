team:         T1
builds_on:    T1-025
performance:  "official #1: 880.9 tok/s (878.4/883.4, CV 0.4%; +0.7% vs T1-025 874.8 whose trials were 863.7/885.8), +38.5% vs BASE, GSM8K 94.0%, TTFT p99 1312, TPOT p99 34.3, accept 2.331, gates + 128K pass; both trials bound to node 0"
summary:
  hypothesis: KB36's per-start bimodality = random NUMA placement of the pinned host-AR/DMA /dev/shm slots: both GPUs hang off node 0;
              numactl is missing so vLLM's --numa-bind/auto-bind never runs; rank 0 first-touches the slots in cudaHostRegister on
              whatever CPU it runs on; node-1 slots make simultaneous GPU->host writes cross the inter-socket link.
  change:     numa_utils.self_bind_to_device_node(): at worker init (after set_device, before distributed init/communicators), pin
              all threads to the GPU's NUMA node CPUs (node from the platform/NVML) and set a libnuma preferred memory policy
              (affinity-only fallback). Skipped if VLLM_NUMA_SELF_BIND=0, single-node host, --numa-bind used, or affinity already
              constrained; errors logged, never fatal. Worker leader-2.w14, ws leader-2.numa (NOTES.md).
  results:    Observed: job 97 (6 unbound starts): slots all on node 1 in 5/6 (whole-buffer, all-or-nothing). Job 100: unbound A got
              node 1 in 2/4; bound B node 0 in 4/4. Per start (tok/s, slots node): A 752.2 N1, B 768.8, B 769.5, A 762.0 N0, B 767.8,
              A 757.4 N1, A 771.1 N0, B 767.0. Only variable tracking the mode: prefill-size AR cost per token r0 1.23-1.25 (N1) vs
              0.96-0.98 us (N0) (+27%); decode step period 27.3-27.4 ms, M=48 ARs, clocks, output tokens (24496) identical.
              w15 microbench (job 99): node-1 slots slow copies only when both GPUs write at once; hostar alone M=300 315 vs 229 us.
              Matches T1-025's official slow trial (same TPOT p50, loss in prefill/TTFT). Estimate: lifts ~70% of slow starts by
              ~1.7-2.5% -> +1.2-1.7% mean, and removes the ±1.3% start noise.
  conclusion: Interpretation: bimodality explained and fixed generically.
validity:     leader-2 review: CPU affinity + preferred memory policy only; no math, no data path change (outputs identical across
              starts at T=0). Generic (same as upstream auto-bind with numactl; decided from the GPU's NUMA node, not the
              benchmark); no root, no environment change. OK.

official #1 (2026-11-08T07:32): 880.9 tok/s at c=16 (mean of 2 trials: 878.4 / 883.4, CV 0.4%) (+38.5% vs BASE), GSM8K 94.0%, MTP acceptance 2.331, TTFT p50 184 ms, TPOT p50 16.4 ms; gates: GSM8K pass, latency pass, 128K request pass (official/1/results.json)
