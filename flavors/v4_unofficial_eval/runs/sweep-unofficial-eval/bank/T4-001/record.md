team:         T4
builds_on:    T3-004
performance:  "not evaluated (run ended); correctness check open"
summary:
  hypothesis: in mixed steps (>=192 rows) the 128 all-reduces (~35 ms/step, NCCL, serial) can hide under GEMMs if the
              row-independent chain between mixers (out_proj -> AR -> add+norm -> gate_up -> down -> AR -> add+norm ->
              next in_proj) runs as two row halves on two streams, each half with its own host-memory AR instance.
              Microbench (8 real-shape layers, CUDA graph, both GPUs): job 203 (NCCL halves: 0 overlap), job 204
              (host-AR halves vs HEAD's serial NCCL: 9.84->8.40 ms at 304 rows, 12.96->11.23 at 400, 17.28->14.36 at 608,
              27.24->24.74 at 1024 per 8-layer stack; bitwise equal) => ~11-23 ms per mixed step, napkin ~4% of wall.
  change:     worker t4-w1, workspace t4-w1: two-half fork/join in the Qwen3.5 nvidia model layer loop (mixer core on
              the full batch), two HostAllReduce instances (10 MB slots), env VLLM_TWO_HALF / VLLM_TWO_HALF_MIN_ROWS=192,
              eager and inside mixed-step graph capture; supersedes HEAD's chunked GEMM+AR. Also contains an unused
              pipelined host_ar2 kernel (csrc/host_all_reduce.cu) from task 1.
  results:    Observed: builds; server starts with HEAD serve args; long_request 128000 pass (job 205); two-half steps
              fire at rows 192..2048 eager and in capture (job 207). c=1 greedy vs VLLM_TWO_HALF=0 (job 208): 64/64
              agreement on 5 of 8 prompts, 63/64, 60/64, 49/64; max |dlogprob| on agreeing prefixes 0.09-0.29.
              In-engine hidden-state check (jobs 209/210): eager rows 531/1159: plain-vs-plain 0, split-vs-plain max
              94.5/115.8, mean 0.33/0.36 (max|x| 518/715); under capture the plain-vs-plain reference itself differs
              (dummy inputs), so those lines are not informative.
              At 2048 rows split-vs-plain is exactly 0 (bitwise). Job 210 forced one W8A8 tile config for all M: numbers
              identical to job 209, so tile configs do not explain the 531/1159-row deviation (worker's guess: an
              M-dependent bf16 cuBLAS kernel such as in_proj_ba, or a defect in the split path; not localized).
              Job 212 (per-layer localization) failed at startup: a debug hook added last (_dbg_list/_dbg_plain) breaks
              model.py; the diff therefore carries debug hooks (VLLM_TWO_HALF_CHECK*, _two_half_check, _dbg_*) that must
              be stripped. No A/B, no unofficial_eval. GPU: jobs 203-212, ~13 GPU-min.
  conclusion: Interpretation: the mechanism is built, but the max deviation (~95-115 on values up to ~700, mean 0.3)
              is larger than I'd expect from tile-config reordering alone: a race at the fork/join (e.g. a half reading
              a residual buffer before the other stream's write) is not excluded. It must be resolved (bitwise
              comparison of split vs plain with identical GEMM configs per half M, or VLLM_TWO_HALF with the same config
              forced) before any eval. Then: same-job A/B vs HEAD and unofficial_eval --gsm8k --long.
validity:     NOT reviewed for promotion. Intended math: row split of row-independent ops + same bf16 sum AR per row
              (exact up to GEMM tile reordering at M/2). Unverified until the deviation above is explained.
