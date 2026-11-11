team:         T36
builds_on:    T31-003 (no code change; change.diff is T31-003's; serve flag num_speculative_tokens 3)
performance:  "unofficial: K=3 vs K=2 on the champion: tie at c=16 (1015.3 vs 1015.7, 2 of 16 requests wait); +2.1% at c=14 (one same-job pair, 150 prompts). Not a candidate (TTFT, memory)."
summary:
  hypothesis: a decode pass is ~80% fixed cost on the champion, so a third draft token (and then confidence-budgeted
              depth, proposal T36-1) should buy clearly more accepted tokens than it costs.
  change:     none in code. Champion workspace t36-champ, num_speculative_tokens 2 / 3 / 4. Scripts and outputs:
              agents/leader-36/jobs/, agents/leader-36/{prof,ksweep,k3mem,k3c14}/.
  results:    Observed:
    - job 356 (nsys, 64 prompts): GPU 0 busy 97-98% of wall; decode-only step 18.6 ms at M<=12, ~+1.6 ms per 16 rows,
      23.4 ms at M=48; mixed step busy = 30.8 + 0.160*P ms; one MTP draft step 0.55-0.62 ms; AR 42.9 us x 128.
    - job 361 (150 prompts, c=16, one start each): K=2 1015.7 tok/s, acceptance .775/.557 (2.33); K=3 1015.3,
      acceptance .770/.542/.388 (2.70), TPOT mean 12.23 vs 13.66 ms, TTFT mean 541 vs 207 ms, "Running 14, Waiting 2,
      KV usage 95.3%"; K=4 does not start (GdnReplayRegistry: min() of empty).
    - job 362: K=3 with --kv-cache-memory 5.8 GiB: CUDA OOM in graph capture. K=2 same job 1032.7 (1.7% above job 361:
      cross-job noise). nvidia-smi 23,804 MiB on both GPUs at K=2.
    - job 363 (c=14, same job, 150 prompts, one pair): K=3 993.9 vs K=2 973.5 tok/s (+2.1%); ITL median 25.03 vs 22.73 ms
      (+10.1%); acceptance 2.69 vs 2.32 (+15.9%); TTFT mean 199 vs 189 ms; 14 running in both.
    - job 360: lost (my script used a shell variable named CC; JIT builds failed). 2.0 GPU-min.
    Estimate before measuring: K=3 +5% overall (cost model +9% step for +17% tokens). Measured +2.1% (one pair).
  conclusion: Interpretation: static K=3 nets about +2% when memory lets every request run; the step grows 10% for 16%
              more tokens, and the decode share dilutes it. This reproduces what T28-003 found (+2.9%, step +10.2%, with a
              two-slot state fix) and T1-015/T5-004 before it. Together with T5-006 and T27-001 (in-server confidence
              budgeting at M=48 raised accepted tokens only +5.0-5.6%), the "more accepted tokens per pass" lever cannot
              reach +5% with the stock MTP head: T36-1's riskiest assumption (confidence good enough to budget rows) was
              already measured in-server by those teams. T36-1 withdrawn before its milestone-1 job was run.
validity:     no code change; exact spec decode. Not for official eval (K=3 does not fit 16 requests in the 5 GiB KV budget).
