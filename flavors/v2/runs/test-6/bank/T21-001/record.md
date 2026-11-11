team:         T21
builds_on:    T14-004
performance:  "unofficial: 917.3 tok/s (1 fresh start, 400 prompts, job 141), vs champion official 956.6 cross-job: about -4%. Prefix-cache hits only 144 of 104,718 prompt tokens."
summary:
  hypothesis: Every request carries the model chat template's default system prompt, a 45-token shared prefix (17% of prompt tokens). The champion's prefix caching hits 0% because hybrid blocks are 1600 tokens. vLLM's off-by-default fine-grained hybrid prefix caching (--prefix-match-unit 16 --enable-mamba-shared-prefix-checkpoint) might recover it. Step-model estimate: +3.5% if the full 45 tokens hit.
  change:     flags only, on the T14-004 code (change.diff = T14-004's). serve_args.txt adds the two flags.
  results:    Observed: 917.3 tok/s, MTP acceptance 2.345, TTFT mean 266 ms (champion official ~219), p99 TTFT 1247 ms, p99 TPOT 32.6 ms. prefix_cache_hits_total 144 of 104,718 queried (hit rate 0.1-0.4% in the log). KV cache usage 83.8% with 16 running.
  conclusion: Interpretation: the stock mechanism doesn't work on this config.
    - Hit count 144 = ~9 hits of 16 tokens. Expected: checkpoint at floor(45/16)*16 = 32, minus one unit for the EAGLE/MTP proof margin = 16 tokens per hit.
    - Hits are rare. Suspected cause: junction checkpoint blocks are recycled out of a nearly full KV pool (83.8%) before reuse. Not verified.
    - The junction chunk stops seem to add a forward pass per diverging request without a hit. That plausibly explains the -4% (unverified: cross-job, n=1).
    - Making it work would need code: a pinned checkpoint, a token-granular junction, and no extra step. That is blocked until the human rules on (a)/(b) (see validity).
  The template evidence is in template_excerpt.txt.
    - chat_template.jinja l.46-52: if enable_thinking is undefined or true, `reasoning_effort|default('xhigh')` sets reasoning_instructions = 'Reasoning effort is set to xhigh. Please think carefully ...'.
    - That text is emitted as the system block for any request. The bench client sends only a user message, so the prefix comes from the model's template, not the client.
validity:     PENDING RULING (human): enabling non-default fine-grained prefix caching whose benefit here is the template's shared prefix. Measurement only; no official eval. Code = T14-004 (already reviewed).
