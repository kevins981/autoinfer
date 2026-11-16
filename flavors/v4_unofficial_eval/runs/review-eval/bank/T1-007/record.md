team:         T1
builds_on:    T2-004
performance:  +3.7% vs the champion (T2-004, 743.9): 771.3 tok/s, starts 762.6 / 779.9 (eval/1, GPU job 36).
              Same kernel measured 3 times on T2-004: 768.6 (T1-005, job 26), 777.2 (T1-006, job 30), 771.3 (this);
              all 6 starts above 743.9; mean of the 3 runs 772.4 (+3.8%).
summary:
  hypothesis: decode all-reduces ([M<=96, 5120] bf16, ~135 per decode step) pay NCCL's ~30 us SHM floor; a custom TP=2
              all-reduce through host-mapped pinned memory cuts each by ~20 us. Napkin -2.7 ms/decode step (~+6-8% max).
  change:     by worker leader-1.w1 (change.diff = T2-004's tuned configs + 3 T1 files in vllm/distributed/device_communicators/:
              hmar/hmar.cu, host_mapped_all_reduce.py, cuda_communicator.py). Kernel: 16 blocks; each writes its bf16 partial
              to its host slot (zero-copy over PCIe), st.release.sys flag, ld.acquire.sys spin on the peer's, reads the peer's
              slot, local+peer in fp32, one rounding to bf16; device epochs, double-buffered slots, CUDA-graph safe.
              Default on for TP=2 without P2P (VLLM_HOST_MAPPED_AR=auto|1|0); eligible: bf16, contiguous, 2D, 16-B aligned,
              <= 96 x 5120; refused under ubatching/DBO; any setup failure on either rank -> both fall back to NCCL.
              Extension built from source (torch cpp_extension, name keyed by source hash), prebuilt on CPU before jobs.
              Optional host-only launch counter VLLM_HOST_MAPPED_AR_STATS (used in this measurement; reads the mmap, no CUDA).
  results:    Observed: 771.3 tok/s (762.6, 779.9); acceptance 2.340 vs 2.348; TTFT p99 1010 ms, TPOT p99 35.4 ms
              (gate_latency pass); 128K pass. Log: "Using ['HMAR', 'PYNCCL'] all-reduce backends".
              Launch counter: 324,274 / 324,605 kernel launches (rank 0/1) by the end of trial 2, vs 2,720 engine iterations
              (incl. mixed steps that use NCCL) -> ~119 per iteration, consistent with ~135 per decode step (not separated
              by step type).
              Exactness: stress test GPU job 33 (agents/leader-1.w1/hmar_test3.py): 1,200 graph-captured + 73 eager collectives
              per rank, fresh random inputs each replay, every captured output compared to NCCL as int16 bit patterns, sizes
              shuffled, rank skew injected, edge values (+-0, +-inf, a qNaN, subnormals, max finite): 0 mismatches.
              Microbench (jobs 25, 27): us per AR hmar/NCCL: M=16 18.9/39.1, 42 41.3/61.0, 48 46.0/64.7, 96 82.5/96.9.
  conclusion: Interpretation: a real gain (+3.7% here, +3.3..4.5% over 3 runs), about half the napkin; why exactly is not
              profiled. A pipelined variant was slower (job 27).
validity:     Exact: the 2-rank bf16 sum is computed in fp32 and rounded once on both ranks (operand order swapped; one fp32 add
              is commutative), bitwise equal to NCCL in the stress test; communication stays bf16. No change to math,
              sampling, parsers, KV or GPU memory (4 MiB pinned host memory). GSM8K not needed (bitwise equal op).
              Known limits (review T1-4 turn 2, open, being fixed): alignment is checked per rank, so an aligned/unaligned
              split between ranks would hang (model all-reduce inputs are fresh allocations, aligned on both ranks);
              incomplete host-unregister cleanup on partial setup failure; P2P detection ordinal handling; build cache key
              lacks compiler/arch identity. None affects the measured path on this machine.
              Reviewed: reviews/T1-4 (2 rounds, codex) with the worker's answers; stats-thread capture bug found in job 34
              and fixed before this run.
