team:         T51
builds_on:    T45-005 (+ env-gated per-step GPU-event probe T51_STEPLOG in v1/worker/gpu_worker.py; off by default)
performance:  "measurement record: budget 192 -1.9%, 160 -10.2% vs 2048 (job 432, 200 prompts, 1 start each)"
summary:
  hypothesis: Spreading prompts as 96-144-row chunks over decode passes ("piggyback") hides prompt GEMM compute under
              the decode weight stream; T35-001's loss at small budgets came from fixed per-mixed-step costs, not physics.
  change:     serve flag --max-num-batched-tokens {2048,192,160} on the champion; step logger (T37_STEPLOG) + GPU
              event probe; nsys node trace of the 192 config (job 434). Worker leader-51.w1; files agents/leader-51.w1/.
  results:    Observed (job 432, k=3, acceptance 2.69-2.71 everywhere, 55.8K prompt tokens each):
              2048 1106.4 tok/s (173 mixed steps), 192 1085.9 (465), 160 993.5 (581). TTFT p50 169/226/271 ms.
              Wall per step: pure decode 24.9 ms in all configs; ANY step with 1-144 prompt rows ~50.5-52.3 ms (flat in P);
              2048's steps with 129-256 rows 64.8 ms, 257+ 70.6 ms.
              nsys (job 434, 192 config; periods inflated 51->67 ms) GPU-0 kernel ms, decode vs mixed P65-128:
              bs_gemm 5.3 -> 21.6, t31b 6.7 -> 0 (all GEMMs move to the M>96 W8A8 path), AR 7.0 -> 14.1,
              GDN prefill core 0 -> 2.0, quant/attn/norm/other +2.8, GPU busy 24.8 -> 46.0 (idle ~21 ms under nsys).
              Plain run: GPU event span 42.6-47.7 ms vs wall ~51-52 in mixed steps (probe can't localize the gap).
  conclusion: Interpretation: today's mixed step has a ~51 ms floor independent of prompt rows up to 144 (likely host
              enqueue of the eager/piecewise mixed path, KB89; not proven), plus a GPU-side excess of ~21 ms at ~100 rows:
              GEMM +9.5 ms (in situ ~0.095 ms/row: standalone cold band at M~160 plus the lost AR-window prefetch),
              AR ~0.07 ms/row, glue ~5 ms. With those costs piggyback cannot pay; this rejects today's implementation,
              not the direction (calibrated sim agents/leader-51/sim/pig2.py reproduces -3.8/-11.7 vs measured -1.9/-10.2).
              Needed: captured mixed pass (removes the host floor), in-situ GEMM band <=~0.04 ms/row, glue <=~5 ms.
validity:     measurement only (serve flag + env-gated probe); not for official eval.
