team:         T1
builds_on:    T1-035
performance:  "OFFICIAL #1 922.8 tok/s (+1.8% vs T1-035 906.3; p99 TPOT 30.9/28.8 ms, normal); unofficial: ABBA same job (job 151), B mean 871.5 vs A 861.4 tok/s = +1.17% (both B > both A); ITL p50 26.49 -> 25.94 ms"
summary:
  hypothesis: The bf16 target lm_head (1.271 GB/rank, ~1.6 ms/step in cuBLAS) is HBM-bound. Storing it losslessly re-encoded with T1-030's block-exponent codec (0.781 of the bytes) and decoding in registers inside the GEMM cuts its time by about 30%.
  change:     ws leader-2.lmhc (w22), new vllm/model_executor/layers/lmh_codec.{cu,py}.
              Load (gpu model_runner.load_model, after the drafter built its FP8 vocab-subset head from the raw weight):
              - encode the target lm_head shard; one block = the 32 values a lane holds as m16n8k16 B fragments; raw fallback per block, plus a side table for groups with > 2 raw blocks.
              - verify_kernel decodes every block with the same decode_group the GEMM uses and compares bitwise. Any mismatch -> ERROR, raw kept, cuBLAS.
              - sanity vs F.linear on random input (max rel diff < 1e-2, else ERROR plus cuBLAS).
              - swap the quant_method, then FREE the raw shard: no full-size copy remains.
              GEMM: bf16 MMA, fp32 accumulation over all of K in order, one bf16 rounding; M chunked by 64; CUDA-graph capturable.
              The final-norm AR's L2 prefetch is repointed at the encoded buffer. The draft head is untouched.
              Opt-outs: VLLM_LMH_CODEC=0, VLLM_LMH_CODEC_PF=0. No serve-arg change.
  results:    Observed:
              - Encoding: 1.271 -> 0.993 GB per rank (0.781); 0.27% raw blocks; bitwise verify 0 mismatching words on both ranks. Model memory 14.63 -> 14.37 GiB; KV unchanged (267,721 tokens).
              - Kernel vs cuBLAS (job 150): M=3 1342 -> 1043 us, M=16 1361 -> 1043, M=24 1470 -> 1044, M=48 1594 -> 1056 (-33.8%), M=64 1624 -> 1071. Encoded stream at ~95% of the read roof.
              - Accuracy vs F.linear: bitwise where cuBLAS sums sequentially (M=2-16 etc.); a few bf16 ulps where cuBLAS split-K rounds through bf16 (M=17-26, 43-64); max rel 5-6e-3.
              - ABBA (job 151, fresh servers, 200 prompts, c=16):
                tok/s 1A 862.4, 2B 870.7, 3B 872.3, 4A 860.4
                ITL p50 26.47 / 25.95 / 25.93 / 26.50 ms
                acceptance 2.335-2.352
                p99 TPOT 30.3 / 52.8 / 42.1 / 33.4 (2 worst requests; p90 TPOT and p90/p99 ITL flat)
              - gsm8k50 88% (= BASE subset). Greedy c=1 10/10 identical vs T1-035 (M=3 verify, same order as cuBLAS there).
  conclusion: Interpretation: -0.54 ms/step as estimated, +1.2% tok/s. Same math up to fp32 summation order (allowed), with weights decoded bit-exact. Watch p99 TPOT in the official run.
validity:     leader-2 reviewed lmh_codec.py/.cu and the hooks. Weight re-encoding meets the round-12 rule: lossless, verified bitwise over all blocks at load with the same decode routine as the GEMM, raw fallback per block, raw shard freed, no lower precision (bf16 x bf16 -> fp32 accumulate, bf16 out like cuBLAS). Only the fp summation order changes. No KV, comm or serving change, nothing benchmark-specific. OK.

official #1 (2026-11-08T12:14): 922.8 tok/s at c=16 (mean of 2 trials: 925.9 / 919.8, CV 0.5%) (+45.1% vs BASE), GSM8K 93.2%, MTP acceptance 2.333, TTFT p50 174 ms, TPOT p50 15.7 ms; gates: GSM8K pass, latency pass, 128K request pass (official/1/results.json)
