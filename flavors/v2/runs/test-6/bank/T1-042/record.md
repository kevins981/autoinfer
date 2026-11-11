notes:        see STACK.md for the full self-contained description of the stack
team:         T1
builds_on:    T1-041
performance:  "OFFICIAL #1 942.2 tok/s (+0.9% vs T1-041 934.1); unofficial: ABBA same job (job 174), B mean 893.4 vs A 882.0 tok/s = +1.3% (both B > both A)"
summary:
  hypothesis: The step anatomy of T1-041 (job 171, agents/leader-2.w23/anat/REPORT.md) shows exposed AR communication in mixed steps is the largest non-floor gap (~11% of wall). The DMA AR lane of the hybrid split runs D2H (~270 us) then H2D (~250 us) strictly serially per 3.9 MB AR. Pipelining them in pieces overlaps the two directions.
  change:     ws leader-2.dmapipe (w23), dma_staged_ar.{cu,py}: VLLM_DMA_PIPE pieces (default 4; 1 = old path).
              - caller stream: D2H of piece p, then doorbell F[me][lane][p] = 1.
              - per-lane aux stream (event fork/join, graph-capturable): wait peer F[p] == 1, reset it to 0, H2D piece p; then ack A[me] = 1.
              - the same add kernel after the join (same bytes, same bf16 add -> bitwise).
              - the next call's producer waits on and resets the peer's ack before ringing, so a stale doorbell is impossible (reset protocol, constants, replay-safe).
              - init asserts the piece count agrees across ranks.
              - test hooks (dbg bits) are inert in production (dbg = 0).
  results:    Observed:
              - Microbench (jobs 172/173, both ranks at once, near-node slots), us/AR for 1 / 2 / 4 pieces:
                2.0 MB 192 / 160 / 158; 3.93 MB 338 / 275 / 255; 6.0 MB 501 / 410 / 365; 3.93 MB alongside a hostar AR 431 / 376 / 367.
                Estimate for 2 pieces at 3.9 MB was 295.
              - Bitwise (job 176): pieces 2/3/4 == 1 piece == the CPU reference at M = 1-1200, incl. a graph-captured run.
              - Controls (rank 1 spins ~200 us before each piece's D2H; differing elements on rank 0): readies on 0; skip piece 0/1/2/3 alone 12.0M / 10.5M / 12.0M / 12.0M; skip all 47.9M. With 2 pieces: skip piece 0 24.0M, skip piece 1 22.5M.
              - ABBA (job 174, fresh servers, 200 prompts, c=16):
                tok/s 1A 878.1, 2B 892.2, 3B 894.6, 4A 885.9
                p99 TPOT 36.0 / 44.2 / 42.0 / 43.0 ms
                acceptance 2.318-2.349
              - gen_check 10/10 identical; gsm8k50 0.88.
              - Note: the test hooks were added after the ABBA; the production path is the same code (source hash 2bf53444eb in the ABBA vs 57cf75741b final).
  conclusion: Interpretation: ~-64 us per in-server DMA AR at mixed-step sizes; +1.3% same-job (projected ~+1%). Exact.
validity:     leader-2 reviewed the diff: copy-engine scheduling and flags only.
              - same bytes and the same add kernel, bitwise-tested;
              - stream-ordered doorbells after the D2H and acks after the H2D;
              - stale-flag safety via the ack-gated reset protocol;
              - test hooks off by default.
              Baseline-precision communication (bf16 payload, as T1-025). No serving or memory change beyond a small aux stream and event. OK.

official #1 (2026-11-08T15:08): 942.2 tok/s at c=16 (mean of 2 trials: 941.4 / 942.9, CV 0.1%) (+48.2% vs BASE), GSM8K 93.2%, MTP acceptance 2.325, TTFT p50 168 ms, TPOT p50 15.4 ms; gates: GSM8K pass, latency pass, 128K request pass (official/1/results.json)
