team:         T2
builds_on:    T1-004
performance:  "unofficial: +6.42% at c=16 vs T1-004 (same-job ABBA, 300 prompts, all B > all A), T2 GPU job 27"
summary:
  hypothesis: BASE/T1-004 capture CUDA graphs only up to 96 rows (max_cudagraph_capture_size 96), so every
              prefill-containing (mixed) step (~100-560 rows) runs eagerly and is CPU-launch-bound. Measured without
              nsys (per-step GPU event timestamps, ws t2-steplog, T2 job 10, BASE): mixed step = 72 ms + 0.21 ms/row vs
              decode 38.4 ms, i.e. ~34 ms fixed eager overhead per mixed step; mixed steps = ~35% of wall time. In the
              BASE nsys profile, mixed-step all-reduces show 150 us median rank skew (one rank waiting on the other's
              launches). Capturing graph sizes up to 512 should remove most of that overhead (napkin ~+6-8%).
  change:     serve flag only: --compilation-config cudagraph_capture_sizes = BASE's 25 sizes (1..96) + [128,160,192,224,
              256,320,384,448,512]. No code beyond T1-004's.
  results:    Observed: (a) BASE + same sizes (T2 job 23, steplog, 200 prompts): mixed step = 38.3 ms + 0.255 ms/row;
              672.4 tok/s vs 639.3 in T2 job 10 (cross-job). (b) sizes up to 1024 (83 sizes): OOM during capture (T2 job
              15). (c) T1-004 + sizes: graph capture 1.31 GiB/GPU, 23.56 GB used/GPU, 128K request pass (53.2 s, server
              answers after). ABBA (job 27, agents/leader-2/abba_cg/): A 690.1/690.4, B 732.4/736.7 tok/s (+6.42%);
              acceptance 2.34 both; p99 TTFT/TPOT A 1216/1271 / 41.2/40.8 ms, B 1073/1188 / 37.2/33.0 ms.
  conclusion: Interpretation: mixed steps were CPU-launch-bound; graphing them removes ~34 ms per mixed step. TPOT p99
              improves too (decode requests no longer stall behind slow eager mixed steps). Steps >512 rows remain eager.
validity:     Serving flag (CUDA graph capture sizes), allowed (task_definition: CUDA graph settings open). Same math: the
              captured graphs run the same kernels; padding to the next capture size adds dummy rows that are discarded.
              Memory: graph pool +~0.05-0.7 GiB; one 128K request tested and passed with T1-004's flags (job 27). No
              benchmark-specific behavior.
