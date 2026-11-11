team:         T1
performance:  "official #1: 699.2 tok/s (+10.7% vs BASE; 3 trials CV 0.4%), GSM8K 93.2%, TTFT p99 1309, TPOT p99 50.6, gates pass"
summary:
  hypothesis: with the faster AR in place, CUDA-graphing prefill-containing steps (T1-001 flag) reduces the rank skew that
              dominates prefill-step ARs (T1-003) and stacks.
  change:     T1-004 (host-staged AR + AG) + serve flag --compilation-config '{"max_prefill_cudagraph_capture_tokens": 2048}'
              (serve_args.txt). Worker leader-1.w2, ws leader-1.hostar2.
  results:    Observed (out/srvB, same job as T1-004's run, single runs): 699.0 tok/s, TTFT p99 1376 ms, TPOT p50 21.1 /
              p99 45.9 ms, accept 2.34, gsm8k50 88%. nsys (out/nsysB): prefill-step AR mean start skew 214 us (316 without the
              flag), post-arrival 519 us; decode AR mean 39.6 us (49.6), decode start skew 9.9 us (30).
              Note T1-001 (flag on BASE/NCCL) gave only +2.2%.
  conclusion: Interpretation: the flag pays more on top of the host-staged path than on NCCL (+4.8% vs +2.2%): graphs cut
              launch-induced skew, which the faster transport then turns into time saved. Not separated: skew reduction vs fewer launches.
              Large-M AR still ~500 us post-arrival (~12-14 GB/s effective).
validity:     leader-1 review: diff = T1-004 (reviewed); flag only changes CUDA graph capture sizes. Memory change -> 128K test:
              passed on BASE code in T1-001, and re-tested on this exact config (job 19, out/srv128k, out/job11.log): 128000-token
              prompt -> HTTP 200, 19 coherent tokens; follow-up request and /health OK; graph capture 1.33 GiB/rank (2.07 GiB in
              T1-001 with NCCL); no errors. OK.

official #1 (2026-11-07T00:38): 699.2 tok/s at c=16 (mean of 3 trials: 700.7 / 700.8 / 696.2, CV 0.4%) (+10.7% vs BASE), GSM8K 93.2%, MTP acceptance 2.339, TTFT p50 210 ms, TPOT p50 20.9 ms; gates: GSM8K pass, latency pass (official/1/results.json)
