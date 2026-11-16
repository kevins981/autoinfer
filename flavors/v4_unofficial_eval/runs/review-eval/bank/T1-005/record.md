team:         T1
builds_on:    T2-004
performance:  +3.3% vs the champion (T2-004, 743.9): 768.6 tok/s, starts 755.8 / 781.3 (eval/1, GPU job 26)
summary:
  hypothesis: decode all-reduces (~135 per step, [M<=96, 5120] bf16) pay NCCL's ~30 us floor over SHM; a custom
              TP=2 all-reduce through host-mapped pinned memory (GPU zero-copy stores/loads over PCIe, flag sync)
              cuts it. Napkin from the microbench: -20 us x ~135 = -2.7 ms per decode step (~33 ms) -> +6-8% at most
              (decode steps only; mixed steps > 96 rows still use NCCL).
  change:     by worker leader-1.w1 (change.diff = T2-004's tuned configs + 3 T1 files):
              vllm/distributed/device_communicators/hmar/hmar.cu (16 blocks; each writes its bf16 partial to its own
              host slot, st.release.sys flag, ld.acquire.sys spin on the peer flag, reads the peer slot, sums in fp32
              in fixed rank order -> bitwise identical on both ranks; device epoch counters, double-buffered slots,
              CUDA-graph safe), host_mapped_all_reduce.py (shared /dev/shm file, cudaHostRegister Mapped|Portable),
              cuda_communicator.py (tried first when VLLM_HOST_MAPPED_AR=1, TP group of 2, bf16, <= 96 x 5120).
              As measured, enabled by env: VLLM_HOST_MAPPED_AR=1 VLLM_HMAR_BUILD_DIR=$AUTOINFER_RUN/agents/
              leader-1.w1/hmar_build CUDA_HOME=/usr/local/cuda-12.6 (prebuilt extension). A record-ready version
              (extension built under $TORCH_EXTENSIONS_DIR, default on when the 2 GPUs lack P2P, both-rank fallback
              to NCCL) is in workspaces/t1-car2, same kernel; not yet measured.
  results:    Observed: harness 768.6 tok/s (755.8, 781.3) vs T2-004 741.1/746.8; acceptance 2.335 vs 2.348;
              TTFT p99 1031 ms, TPOT p99 34.8 ms (gate_latency pass); 128K request pass. server.log: "Host-mapped
              all-reduce enabled (rank 0, 4 MiB pinned)".
              Mechanism (microbench, 2 processes, graph-replayed, GPU jobs 25 and 27): bitwise equal to NCCL and
              equal across ranks at M = 16/42/48/64/96; us per AR hmar vs NCCL: 18.9/39.1, 41.3/61.0, 46.0/64.7,
              58.8/82.7, 82.5/96.9. A pipelined variant (per-sub-chunk flags) is slower (48.6 us at M=42, job 27).
  conclusion: Interpretation: real gain, both starts above the champion's mean; about half the napkin (+3.3% vs
              +6-8%): plausible causes are fewer ARs in captured decode graphs than 135, or AR not fully on the
              critical path. The start spread (3.4%) is large; a profile would confirm the per-step saving.
validity:     Exact: 2-rank bf16 sum computed in fp32 and rounded once, same order on both ranks; bitwise equal to
              NCCL in the microbench; communication stays bf16 (baseline precision). No change to math, sampling,
              parsers, KV or memory on the GPU (4 MiB pinned host memory). GSM8K not needed (bitwise-exact op).
