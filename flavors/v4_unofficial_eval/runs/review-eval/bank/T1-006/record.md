team:         T1
builds_on:    T2-004
performance:  +4.5% vs T2-004 (743.9): 777.2 tok/s, starts 775.2 / 779.1 (eval/1, GPU job 30). Superseded by T1-007 (same kernel, reviewed fixes).
summary:
  hypothesis: as T1-005; record-ready packaging.
  change:     T1-005's simple host-mapped all-reduce (hmar), packaged: extension built from source via torch cpp_extension
              under $TORCH_EXTENSIONS_DIR (prebuilt on CPU before the job), default on for TP=2 when the two GPUs lack P2P,
              both-rank agreement on fallback to NCCL. No env vars needed. Pipelined variant included but off.
  results:    Observed: 777.2 tok/s (775.2, 779.1); acceptance 2.335; TTFT p99 1039 ms, TPOT p99 37.1 ms (gate_latency pass);
              128K pass. server.log: "Host-mapped all-reduce enabled (rank 0/1, 4 MiB pinned, simple)".
  conclusion: Interpretation: confirms T1-005's gain with the default-on build. Not submitted: review T1-4 found issues
              (alignment gate, ubatching gate, logging) fixed in T1-007.
validity:     see T1-007. Reviewed: reviews/T1-4.
