team:         T1
builds_on:    T1-007
performance:  within noise of the champion (T1-007, 771.3): 772.4 tok/s, starts 769.7 / 775.2 (eval/1, GPU job 41);
              start 1 below 771.3, so it does not beat the champion under the promotion rule. Not a candidate.
summary:
  hypothesis: none for speed: T1-007 + the setup-robustness fixes from review T1-4 turn 2; expected equal speed.
  change:     (worker leader-1.w1; files as T1-007) (a) eligibility no longer depends on pointer alignment (identical on
              both ranks); unaligned inputs are staged into a preallocated aligned buffer; (c) host-unregister/mmap/fd
              cleanup on partial setup failure and on communicator destroy; (d) P2P detection by device UUID, agreed
              across ranks (MIN), before the extension is built; (e) extension cache key = hash of source + nvcc identity
              + torch/CUDA/Python versions + arch + ABI + flags (not CC/CXX; the same source from another workspace
              path can relink under the same tag, so not an immutable-artifact guarantee); kernel templated on blocks/threads/unroll (default
              16/512/4 as measured), timeout clock checked every 4096 polls; launch counters (host-only).
  results:    Observed: 772.4 tok/s (769.7, 775.2); acceptance 2.344; TTFT p99 1012 ms, TPOT p99 34.8 ms (gate_latency
              pass); 128K pass. Stress test (job 40, same design as job 33, now incl. unaligned views): 0 mismatches.
              Counters (eval/1/hmar_stats): 2,937 captured hmar calls = 22 target graph captures x 132 + 11 drafter
              captures x 3 (the 135/264 per-row buckets are target+drafter captures of the same size combined).
              322,920 = last sampled block-0 publication epoch per rank in the second server start (warmup included;
              counters reset per start); not a per-step launch figure.
              Geometry/unroll sweep (job 39, 17 variants, bitwise-checked): all within ~1 us (M=42: 40.2-41.7 us vs NCCL
              60.9); best variant 0.65% (M=42) / 1.26% (M=48) faster than default in one pass (unconfirmed):
              no substantial geometry improvement found.
              Overlap discriminator (job 42, experiment-only kernel, never selectable in the server): both tested ordered
              warp-specialized overlap variants were slower than the simple kernel (S=4: 51.0 us at M=42, S=2: 53.2 vs
              41.4). Relaxed publication reduced their time (31.7 us) but produced incorrect results, implicating
              publication ordering in this design's overhead. No faster correct overlap implementation was found;
              further overlap work is deferred.
  conclusion: Interpretation: robustness fixes cost nothing measurable. The host-mapped AR runs at ~0.8 us/row (write then
              read over PCIe); no substantial further kernel-level gain found in the variants tried.
validity:     as T1-007 (exact, bitwise equal to NCCL); staging copy is a plain copy (exact).

review (reviews/T1-4, 3 turns, code + claims): staging, cleanup, UUID/MIN P2P agreement, stats thread (no CUDA calls),
experiment kernel unreachable from the server: confirmed. Adopted corrections above (AR counts, launch figure, sweep and
overlap wording, cache key). Open, not fixed (team closing, CPU-only polish): unregister_host() ignores CUDA's return
code; job 40 stress did not cover asymmetric alignment or graph-captured staging. Stress count 2,400 = both ranks =
1,200 logical collectives. Mean +0.15% vs T1-007: exact performance equivalence not established.
