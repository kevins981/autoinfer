# T21 write-up (leader-21, explore). Recommending close: M1 missed its gate

GPU: 48.1 of 50 granted (seed 20 + tranche 30). Worker: leader-21.w1.
Records:
- bank/T21-001: stock flags, PENDING RULING at the time;
- bank/T21-002: generic prefix-state cache, +0.5% n=1.

Notes: local_kb/leader-21.md.

## Direction
Don't prefill what every request shares. The model's chat template injects a default system prompt whenever thinking is on (chat_template.jinja l.46-52, "Reasoning effort is set to xhigh..."). So every request starts with the same 45 tokens, 17% of prompt tokens. The champion's prefix cache hit 0%: hybrid attention blocks are 1600 tokens and GDN checkpoints are block-aligned. Human ruling: generic fine-grained hybrid prefix caching is allowed, with nothing sized to this prompt.

## Measured
- **The stock flags don't work** (`--prefix-match-unit 16 --enable-mamba-shared-prefix-checkpoint`, T21-001, job 141): 144 hit tokens, ~-4%.
  - Cause (code read + CPU sim): partial-hash entries are registered only at each prompt's own tail, so a sibling's junction is never found. The feature serves multi-turn, where a whole earlier prompt is a prefix.
  - The `eagle_group_ids` requirement also silently disables it with MTP lookahead hashes.
  - A per-prompt tail stop adds a forward pass to every request.
- **The generic fix works** (T21-002): a bounded LRU index of per-token prefix hashes from live traffic gives the junction. CoW private copies are made on hits.
  - Hits >= 44 tokens on ~all requests; 4 different prefixes verified on the server.
  - CPU cost 0.59 ms/request at admission, +0.15 ms per decode step.
- **Champion-specific bug:** the CoW block copy must also clear the champion's GDN replay flags (`layer._gdn_replay_flags`) of the destination block. Otherwise a recycled block replays a stale log onto the copied state, and MTP acceptance drops from 2.33 to 2.21 after block churn. This applies to anyone copying GDN state blocks on the champion.
- **Junction stops need a gain rule:** shared first user words ("Write a ...") create junctions 2-6 tokens past the cached prefix. Each costs an extra mixed step: 542 vs 352 mixed steps in the T19 trace (job 187), -4.2/-4.7% (jobs 182/184). The fix accepts a stop only if it gains >= 16 tokens (first junction) or >= 64 tokens (one GDN chunk, when extending).
- **Result:** same-job A/B (job 189, B A, 400 prompts):

| | A | B |
|---|---|---|
| tok/s | 954.1 | 958.8 (+0.5%, n=1, noise level) |
| acceptance | 2.332 | 2.332 |
| p99 TTFT | 1231 ms | 679 ms |

## Why the forecast was wrong (+3.5% forecast vs +0.5% measured)
- I applied KB77's linear mixed-step fit (38.5 + 0.229 ms/prefill token, with a 49.7 ms floor) per request.
- At small R the mixed step sits near the Marlin HBM/tensor crossover, so the first tens of prefill tokens ride on the weight stream almost for free. The global slope overstates their marginal cost several times over. (Interpretation, consistent with the 7x miss; not measured directly.)
- **Lesson:** cost per-request prefill cuts with an in-situ A/B or a marginal-cost model at the actual R, not with the global slope.

## Left
- M2 (prompt tokens as forced-accepted draft rows through the uniform decode graph, for requests whose residual prompt is short) was not built. It targets the mixed-step floor instead of the token slope. It's modelled at ~+1.9% from the same step model. M1+M2 come to ~+2.5% if they work, below the bar.
- T21-002 is exact at chunk-split level and generic, but not official-eligible:
  - GSM8K/128K not run;
  - exactness not re-run after the flag fix;
  - n=1;
  - not rebased onto T27-002.
  An exploit pass could land it for ~+0.5% and the p99 TTFT cut.

## KB candidates
1. **CoW GDN replay-flag bug (measured).** On the champion, copying a GDN state block (prefix-cache CoW, or any block copy) must also zero the destination block's rows in `layer._gdn_replay_flags`. Otherwise the fused decode kernel replays a stale log onto the copied state after block churn.
   - Measured: MTP acceptance 2.209 vs 2.315 without vs with the clear (job 177 vs 180, same 96 prompts).
   - Fix: `_clear_gdn_replay_flags_of_copy_dsts` in T21-002's model_runner.py.
2. **Small-R prefill marginal cost (interpretation).** KB77's 0.229 ms/token slope is a global fit. It overstates the cost of the first tens of prefill tokens in a mixed step, which ride near the Marlin HBM/tensor crossover and the ~49.7 ms floor. Removing ~44 tokens from every request measured +0.5% (job 189, n=1) against +3.5% predicted by the slope.
3. **Working generic hybrid prefix-state cache (T21-002; observations, n=1).**
   - Live-traffic, token-granular junctions; CoW private copies; gain rule >= 16 / >= 64 tokens.
   - Hits >= 44 of the 45 template tokens on ~all requests (17.6K/104.7K prompt tokens). The 4 prefixes verified on the server: template 44, custom short 56, custom long 95, reasoning_effort=low 32.
   - Acceptance unchanged. p99 TTFT 679 vs 1231 ms (n=1). Throughput +0.5% (noise).
   - The stock vLLM flags alone hit nothing for sibling prompts (cause above).
4. **What M2 would need** (prompt tokens as forced-accepted draft rows):
   - Scheduler: a request with residual prompt u <= ~45 after a hit is scheduled as a uniform 1+k query whose "drafts" are its next prompt tokens.
   - GDN metadata: classify it as a spec decode, accepted = 3 (gdn_attn.py ~l.336-367, num_decode_draft_tokens).
   - Rejection sampler: force acceptance and discard output until the last prompt token.
   - MTP proposer: next token = next prompt token (the partial-prefill backup path).
   - Exactness check vs the chunked prefill.
   - Value: avoids a ~24 ms mixed-step excess for ~157 of 400 requests, at ceil(u/3) slot-steps each. ~+1.9% by the step model, which just overestimated M1 ~7x, so treat it as optimistic.
