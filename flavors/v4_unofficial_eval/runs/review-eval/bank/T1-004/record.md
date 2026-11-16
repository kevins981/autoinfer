team:         T1
builds_on:    T2-004
performance:  stopped after 1 start, 641.6 tok/s (-13.8% vs the champion T2-004, 743.9); latency gate FAIL (TTFT p99 3457 ms).
              Confounded by KV capacity (below): a screen negative for "k=3 at 4.25 GiB KV", NOT for k=3 itself.
summary:
  hypothesis: with W8A8 (T2-004) the verify rows are cheaper; MTP k=3 raises tokens/step (napkin p3 ~0.35-0.40).
  change:     flags only: T2-004 serve args with num_speculative_tokens 3 and --kv-cache-memory 4563402752 (4.25 GiB,
              chosen for memory headroom before knowing the per-request KV need). Workspace from T2-004 (its tuned configs).
  results:    Observed (trial 1 only, GPU job 23): acceptance 2.688 (per position 0.768/0.538/0.383) vs ~2.35 at k=2;
              mean TPOT 17.2 ms vs 20.3 (T2-004 trial 1), median ITL 39.2 vs 37.5 ms; but mean TTFT 1526 ms vs 227,
              benchmark 141.5 s vs 122.2 s. Server stats: mostly "Running: 12 reqs, Waiting: 4, KV usage 96.3%"
              (capacity 220,255 tokens) vs T2-004 "Running: 16, Waiting: 0, KV usage 83.8%" at 5 GiB.
              KV per running request: ~8.0% of 4.25 GiB = 0.34 GiB at k=3 vs ~5.2% of 5 GiB = 0.26 GiB at k=2.
              No OOM, 0 preemptions.
  conclusion: Interpretation (revised after review T1-3): KV capacity limited this run to ~12 running requests
              (blocks: 163 total at k=3 vs 192 at k=2; ~13 vs ~10 blocks per running request) and explains the TTFT
              regression (server queue time 1.32 s vs 0.02 s per request). Mean TPOT was 15% lower, but at reduced
              concurrency (12 vs 16), so it is confounded; median ITL rose 4.6%. Whether k=3 improves throughput with 16
              running requests is untested. Estimated need: 16 x 0.34 = ~5.5 GiB KV (+0.5 GiB vs T2-004), an
              extrapolation; would need the 128K test and an OOM check.
validity:     flags only; exact speculative decoding (method unchanged, more draft tokens). KV change would need the
              128K test (not reached: stopped after trial 1).
              Reviewed: reviews/T1-3 (codex); corrections adopted. change.diff = T2-004's code (no T1 change).
