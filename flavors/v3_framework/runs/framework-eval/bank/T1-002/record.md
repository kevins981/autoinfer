team:         T1
builds_on:    BASE
performance:  "unofficial: +4.35% at c=16 (ABBA, 200 prompts, all B > all A), T1 GPU job 13"
summary:
  hypothesis: NCCL (ring via host SHM, no P2P) costs ~53-60 us per decode-size all-reduce (42-48 rows x 5120 bf16),
              135 per decode step = 7.7 ms of 38.5 ms. A one-shot all-reduce through mapped pinned host memory
              (each rank stores its partial to host, flags, loads the peer's) should cost ~25-30 us (napkin) -> ~+10%.
  change:     new backend HOSTMEM (vllm/distributed/device_communicators/hostmem_all_reduce.py + hostmem_ar.cu,
              JIT-built with torch cpp_extension), first in the TP all-reduce dispatch for world_size 2 without peer
              access, bf16, contiguous, <= VLLM_HOSTMEM_AR_MAX_BYTES (default 1,310,720 B = 128 rows x 5120; 0 disables);
              NCCL above. POSIX shm segment cudaHostRegister-ed in both processes; per-block epoch flags in host memory,
              epoch counters in device memory (CUDA-graph capturable), double-buffered slots.
  results:    Observed: microbench (worker jobs 3,4,6; agents/leader-1.w1/hm/): 43 us vs NCCL 53-60 us at n=42,
              17 vs 31 us at n=14, neutral at n=260-375, slower above; output bitwise equal to NCCL (torch.equal).
              Napkin was 25 us: concurrent store+load to host memory reaches only ~15-17 GB/s/direction (T2 job 8),
              PCIe Gen4 x16 confirmed under load (job 13). Server A/B (job 13, agents/leader-1.w1/jobs/hm_abba/):
              A 637.7/629.1, B 653.8/668.1 tok/s, +4.35%, acceptance unchanged (B 2.348/2.341, A 2.366/2.341);
              TPOT p99 B 51.8/42.4 vs A 41.9/42.1 ms (one outlier). gsm8k 50 on B: 88.0% (= BASE's documented 88.0%).
  conclusion: Interpretation: ~17-20 us saved per decode all-reduce, larger than its napkin share (~4%) suggests;
              mixed steps unaffected (they use NCCL).
validity:     Exact: same bf16 values summed in the same order on both ranks (fp32 add, round to bf16), bitwise equal to
              NCCL in the microbench; communication stays bf16 (no compression). Generic (any bf16 TP=2 all-reduce
              below a size threshold on a P2P-less box), nothing benchmark-specific. Memory: ~5 MB pinned host,
              negligible GPU memory. Leader review: acquire fence (__threadfence_system) added after the flag poll; change.diff includes it (the A/B ran without it).
