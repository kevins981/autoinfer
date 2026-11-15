team:         T1
builds_on:    T1-002 (microbenchmark only, no engine change)
performance:  "not an engine change: null result (no NCCL knob beats plain Simple beyond noise at decode/prefill sizes)"
summary:
  hypothesis: on top of NCCL_PROTO=Simple, channel count / threads / buffer size might cut all-reduce latency further at 48-260 rows.
  change:     none. 2-process standalone sweep, CUDA-graph replay of bf16 [M,5120] all-reduce, bench.py from leader-2.w1 (T2), job.sh + nccl.jsonl here.
  results:    Observed (GPU job, label t1-nccl-simple-sweep, ~1.6 GPU-min), graph us at M=48 / 260 / 2048: Simple 67/211/1472 and again 58/205/1470 (repeat spread ~9 us at M=48 = the noise); nch8 65/210/1359; nch8+buf1M 61/202/1340; nch4+thr256 58/201/1446; thr512 59/213/1473; buf8M 57/203/1456; thr128 73/311/2348 (worse); nch1 65/237/1696 (worse).
  conclusion: Interpretation: at the sizes that matter (48 rows decode, ~260 rows mixed) every variant is within the ~9 us repeat noise of plain Simple; more channels help only at >=512 rows (-8% at 2048), which mixed steps rarely reach. Not worth an engine eval. The all-reduce floor through host memory at decode size is ~55-60 us with NCCL.
validity:     n/a
