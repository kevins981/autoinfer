team:         T3
builds_on:    T1-017
performance:  "WITHDRAWN: unofficial PASS +1.12% with N=8 picked on benchmark prompts (not allowed); on generic data (job 174) N=1 best"
summary:
  hypothesis: a mixed (prefill-carrying) step costs a decode step + ~65 ms fixed + ~0.1 ms/prompt token (T3 job 155 on T1-016),
              and in steady state each prompt gets its own mixed step. Admitting prompts at most every N steps merges ~1+N/10
              prompts per mixed step (shares the fixed part) at the cost of a briefly empty slot. Napkin was ~8% at N=10.
  change:     serve flag only: vLLM's generic --prefill-schedule-interval 8 (existing scheduler "prefill cadence": prefill is
              admitted at most once per 8 executed steps while decodes run; capacity escape when the queue backs up). No code.
  results:    Observed: mean ITL 39.6 (HEAD) -> 39.1 / 38.5 / 37.9 / 36.5 ms for N=4/6/8/16; throughput +0.9/+0.5/+1.1/-0.2% vs HEAD
              mean 833.2 (cross-job). p99 TTFT 1033 ms at N=8 (HEAD ~1000, gate 1610); p99 TPOT 32 ms (HEAD 40-51).
  conclusion: Interpretation: merging works (mixed-step cost per prompt falls) but the empty-slot cost is close to the saving;
              the napkin overpriced the fixed part on T1-017 (W8A8 made mixed steps cheaper) and underpriced idle slots. N~8 best.
validity:     Generic serving flag of vLLM; scheduling only (batch composition), math unchanged, outputs complete. Not keyed to
              prompts/seed; tuned for the target load c=16, other loads still work (capacity escape; no starvation). Prefix caching
              and parsers untouched. No memory change. Reviewed by leader-3.

unofficial #1 (2026-11-15T12:04): PASS: +1.12% vs HEAD T1-017 (B 845.7 / 839.1 vs reference mean 833.1 over 2 starts, 0 run in this job; 200 prompts); p99 TTFT 1193 ms / TPOT 30.5 ms, MTP acceptance 2.338 (unofficial/1/unofficial.json)

Generic-data choice of N (manager ruling 2026-11-15: N must be chosen on non-benchmark data), job 174, HEAD T1-017 code,
vLLM synthetic `random` dataset at its default lengths (1024 in / 128 out, ignore_eos, seed 1), 1 start per arm:
  c=16: N=1 341.4, N=4 330.1, N=8 334.8, N=16 327.6 tok/s (acceptance 1.75-1.84 varies by arm on random tokens)
  c=1:  TTFT median 275/271/267 ms (N=1/4/8), TPOT 11.8 all; c=4: 188.5/194.6/190.7 tok/s.
Verdict: on generic data no N > 1 beats N=1, so there is no defensible N; T3-002 is WITHDRAWN as a candidate. The
benchmark-screened N=8 PASS (+1.12%) stays recorded but must not be promoted. Interpretation: the gain depends on the
arrival/prefill mix (holds pay only when admissions are rare relative to steps), i.e. it is workload-specific.
