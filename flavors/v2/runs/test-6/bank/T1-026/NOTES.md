# leader-2.w14 R8: per-start bimodality (KB36) = NUMA placement of pinned AR slots
Topology: both GPUs (17:00.0, 2a:00.0) on NUMA node 0 (CPUs 0-11,24-35). numactl not installed -> vLLM --numa-bind/auto-bind
inactive; workers unpinned; kernel numa_balancing=1. /dev/shm hostar (24608 pages) + dmaar (12289 pages) slots are first-touched by
rank 0 inside cudaHostRegister -> land on whatever node rank 0 ran on.
Step 1 (GPU-free): T1-025 official trials: TPOT p50 equal (16.58/16.55); slow trial: +3% prefill time, +1.2% ITL/draft, TTFT p50 193
vs 171, TPOT p99 47.7 vs 35.2; clocks equal (only 4 samples). Trial-1 startup 155 s = JIT AR ext rebuild in fresh ws (build.ninja
carries the source path; shared TORCH_EXTENSIONS_DIR by hash still rebuilds when the path changes).
job 97 (s2/run1, benches lost to a script bug): placement N1 in 5/6 starts, N0 in 1/6.
job 100 (s2/ab1, ab.sh, 110 prompts c=16, order ABBABAAB; A=ws leader-2.bimodal (T1-025), B=ws leader-2.numa (self-bind)):
  start place tok/s  TPOT50 TTFT50 acc   decode-period  AR48 r0/r1 us  prefill AR us/tok r0/r1
  1A   N1   752.2  17.88  205.9  2.337  27.39 ms       51.1/53.4      1.228/1.594
  2B   N0   768.8  16.98  207.5  2.352  27.32          51.6/53.0      0.962/1.288
  3B   N0   769.5  17.24  223.0  2.338  27.35          51.7/53.6      0.976/1.310
  4A   N0   762.0  17.40  209.3  2.308  27.36          51.8/53.4      0.961/1.317
  5B   N0   767.8  17.31  199.8  2.316  27.40          52.5/53.7      0.962/1.311
  6A   N1   757.4  17.94  208.7  2.321  27.41          51.6/53.3      1.246/1.600
  7A   N0   771.1  17.31  210.2  2.331  27.38          52.2/53.6      0.972/1.287
  8B   N0   767.0  17.05  216.1  2.340  27.33          51.5/53.3      0.971/1.317
  output tokens 24496 in every start (T=0). GPU clocks identical across starts (g0 2235 locked, g1 ~2650), no throttle reasons.
  N1 starts mean 754.8 vs N0 starts 767.7 (+1.7%). B mean 768.3 vs A mean 760.7 (+1.0%; A had 2/4 N0 by chance). B range 767.0-769.5.
Interpretation: remote slots slow prefill-size ARs ~+27%/token (both GPUs DMA to node 1 at once; matches w15 job 99 microbench);
decode ARs (M=48) and decode step period unaffected -> the loss is in mixed steps (TTFT/tail), as in the official slow trial.
Fix: ws leader-2.numa: numa_utils.self_bind_to_device_node() called in gpu_worker.init_device (before communicators): all worker
threads -> GPU node CPUs, libnuma preferred policy (affinity-only fallback). Opt-out VLLM_NUMA_SELF_BIND=0; skipped with --numa-bind,
constrained affinity, or single node. Exact (no math change).
