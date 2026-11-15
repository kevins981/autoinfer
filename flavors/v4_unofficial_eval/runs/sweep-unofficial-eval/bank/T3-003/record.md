team:         T3
builds_on:    T1-017
performance:  "unofficial negative: -0.72% vs T1-017 (826.9/827.4 vs 833.1, 2 starts); 128K pass"
summary:
  hypothesis: graphed mixed steps (<= 512 tokens) run their NCCL all-reduces serially: T3 profile of T1-017 + interval 8
              (agents/leader-3/prof_t3002_node, 64 prompts) shows median prefill step (395 prompt tokens) with 45 ms NCCL of a
              146 ms period. T1-012's row-chunked GEMM + side-stream AR overlap exists but is skipped during CUDA-graph capture
              (VLLM_CHUNKED_AR_GRAPH=0). Compile mode is NONE (breakable graphs), so capture can record the 2-stream pattern.
              Napkin: hide ~half the AR of steps with >= 192 rows: -10..-20 ms per such mixed step, ~1-3%.
  change:     default of VLLM_CHUNKED_AR_GRAPH flipped to "1" in linear.py (one line). Rows >= 192 (VLLM_CHUNKED_AR_MIN_ROWS,
              T1's value) use the 2-chunk pipeline also in captured graphs. Workspace t3-cag.
  results:    Observed: unofficial #1 B 826.9 / 827.4 vs HEAD mean 833.1 (-0.72%), p99 TTFT 964 ms (HEAD ~1000), TPOT 40.6.
  conclusion: Interpretation: overlap inside graphs does not pay at <= 512-token steps: halving M splits each W8A8 GEMM into two
              smaller, less efficient launches (re-reading weights), and each chunk's AR still runs serially on NCCL while the
              link and the GEMM contend (cf. T2's link/HBM coupling). Not pursued; no profile taken.
validity:     Same math: the GEMM is split by rows (independent), each chunk all-reduced with NCCL at bf16; T1-012 verified the
              chunked output bit-equal to the unchunked path. Generic default, not benchmark-keyed. Graph memory: side stream +
              events only; run --long. Reviewed by leader-3.

unofficial #1 (2026-11-15T12:25): negative: -0.72% vs HEAD T1-017 (B 826.9 / 827.4 vs reference mean 833.1 over 2 starts, 0 run in this job; 200 prompts); 128K pass, p99 TTFT 964 ms / TPOT 40.6 ms, MTP acceptance 2.330 (unofficial/1/unofficial.json)
