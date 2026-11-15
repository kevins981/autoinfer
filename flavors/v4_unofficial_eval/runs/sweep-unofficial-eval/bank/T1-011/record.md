team:         T1
builds_on:    T1-010
performance:  "unofficial negative: -1.05% vs T1-010 (2 starts)"
summary:
  hypothesis: With max_num_batched_tokens=2048 (default), prefill steps over 512 tokens run outside the captured
              graphs (HEAD captures prefill graphs up to 512 tokens). Capping the budget at 512 makes every mixed
              step graph-eligible and spreads long prompts over more steps, which also smooths TPOT. Napkin: T2's
              profile has steps >512 at 16.8% of time; they are GPU-bound (Marlin 54%, NCCL AR 33%). So the expected
              gain is small (0-1%), and the risk is TTFT p99 (more steps per long prompt; headroom 1146 vs 1609 ms).
              This is T2's W1b idea (reports/T2-sweep/update-1.md); leader-2 confirmed T2 has not tested it.
  change:     serve args only: HEAD's args + --max-num-batched-tokens 512. Code = T1-010.
  results:    Observed: unofficial #1: B 743.1 / 746.4 vs HEAD mean 752.6 (-1.05%); p99 TTFT 1262 ms (HEAD 1146), TPOT 35.5 ms
              (HEAD 36.4); acceptance 2.328; 128K pass.
  conclusion: Interpretation: graphing steps >512 tokens does not pay; those steps are GPU-bound, as leader-2 predicted.
              Splitting long prompts into 512-token chunks adds per-step costs (more full weight reads and more ARs per
              prompt token at small M) and raises TTFT. TPOT improves slightly. Dropped.
validity:     Scheduling knob only: same math per token. Chunked prefill splits a prompt's prefill across steps, and each
              chunk attends to the earlier KV, so the computation is the same up to fp reordering. Not tuned to the
              benchmark beyond a round chunk size that matches the existing graph capture limit. Long request gated by --long.

unofficial #1 (2026-11-15T08:51): negative: -1.05% vs HEAD T1-010 (B 743.1 / 746.4 vs reference mean 752.6 over 2 starts, 0 run in this job; 200 prompts); 128K pass, p99 TTFT 1262 ms / TPOT 35.5 ms, MTP acceptance 2.328 (unofficial/1/unofficial.json)
