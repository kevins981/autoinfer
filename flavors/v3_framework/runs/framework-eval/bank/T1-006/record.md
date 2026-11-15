team:         T1
builds_on:    T1-001
performance:  "unofficial: mixed, -1.57% (ABBA 200 prompts, job 17: B 646.1/656.4 vs A 653.9/669.4) — unbalanced first version"
summary:
  hypothesis: A reduced draft vocab (ruling 2; ids from the tokenizer's BPE merge order = generic, non-benchmark) shrinks
              the fp8 draft lm_head; napkin ~+1-2% if acceptance holds.
  change:     env VLLM_DRAFT_VOCAB_TOPK=K (default off): drafter's lm_head copy keeps ids < K + special tokens; argmax over
              the subset, mapped back to full ids. Verification untouched. (change.diff = the later rebalanced version,
              never run on GPU; job 17 ran the first version.)
  results:    Observed: CPU coverage on 379k tokens of package docs (non-benchmark): ids >= K = 17.2% (16k), 8.9% (32k),
              3.2% (64k), 0.12% (128k). A/B at K=131072: mixed -1.57%; acceptance B 2.337/2.346 vs A 2.362/2.359.
              Vocab-parallel split by id range left rank 0 with its full 124,160 rows (rank 1 kept 6,945): no saving on
              the critical path.
  conclusion: Interpretation: small vocab subsets lose too much acceptance (Qwen's id order is a weak frequency proxy for
              chat text); at K where loss is negligible the head shrinks only ~35-50% -> best case <~1% after local argmax
              already removed the gathers. Dropped (low value). A better frequency source could change this.
validity:     exact (draft side only, ruling 2 source = tokenizer merge order). Not a candidate.
