team:         T21
builds_on:    T14-004
performance:  "unofficial, same-job A/B (400 prompts, c=16, fresh starts, order B A, job 189): B 958.8 vs A 954.1 tok/s (+0.5%, within noise; n=1). Hits 17,638/104,718 prompt tokens; MTP acceptance 2.332 both; mean TTFT 210 vs 218 ms; p99 TTFT 679 vs 1231 ms. Earlier versions of this change: -4.7% (job 182) and -4.2% (job 184) before the junction-gain threshold. G1 (+2.5%) NOT met."
summary:
  hypothesis: A generic live-traffic prefix-state cache (token-granular junction) lets every request skip the 45-token template prefix, ~+3.5% by the step model.
  change:     (change.diff = T14-004 + this change; the T21 part is the files below.) Flags `--prefix-match-unit 1 --enable-mamba-shared-prefix-checkpoint`.
    - kv_cache_manager.py: bounded LRU index (65536 entries, first 1024 prompt tokens per request) of per-token prefix-chain hashes of live requests -> `_seen_prefix_junction`: junction = deepest hash an earlier different request carried (token-granular with unit 1). Dropped the `eagle_group_ids` requirement of mamba_shared_prefix_checkpoint (it disabled the feature with MTP lookahead hashes).
    - single_type_kv_cache_manager.py (FullAttentionManager._cache_partial_tail_block): also registers the junction as a partial entry; block_pool.cache_partial_block(keep_existing_hashes=True) so the later prompt-tail entry doesn't erase it. All managers get shared_prefix_checkpoint.
    - scheduler.py _mamba_block_aligned_split: no per-prompt tail stop in this mode (it cost an extra pass for every request).
    - worker/gpu/model_runner.py: after the CoW block copies, zero the champion's GDN replay flags (`layer._gdn_replay_flags`) of the destination blocks.
    - kv_cache_manager.py get_computed_blocks: a junction stop is accepted only if it gains >= 16 tokens (first junction: one hash-aligned run) or >= 64 tokens (one GDN prefill chunk) when extending an existing hit. These are generic cost rules; without them, shared first user words ("Write a ...") caused spurious stops (-4%).
  results:    Observed:
    - The stock path found no junction for sibling prompts. Partial-hash entries are registered only at each prompt's own tail.
    - The acceptance drop (2.21 vs 2.33) was a stale champion GDN replay flag on CoW destination blocks. With the flag cleared: 2.332 vs 2.332 (job 189).
    - The -4.2%/-4.7% (jobs 182/184) was spurious junction stops: B had 542 mixed steps vs 352 for A (T19 tracer, job 187). Fixed by the gain thresholds.
    - Final job 189: +0.5% (n=1). Hits: >= 44 tokens on ~all requests.
    - CPU cost with unit 1: 0.59 ms/request at admission, +0.15 ms/decode step (async-hidden). CoW copy ~0.1 ms.
    - Exactness: hit vs full prefill max logprob diff 0.03-0.17, the same level as the chunk split (0.095-level); greedy first token 16/16. That check ran BEFORE the replay-flag fix and was not re-run.
    - Generality: 4 prefixes on the server (template default 44, custom short 56, custom long 95, reasoning_effort=low 32); sim prefixes of 20/45/77 tokens.
  conclusion: Interpretation:
    - The mechanism is generic and works, but skipping ~44 prompt tokens per request is worth only ~+0.5%, not the +3.5% the step model gave.
    - The likely reason: KB77's 0.229 ms/token is a global linear fit. At small R the mixed step sits near the HBM/tensor crossover and its 49.7 ms floor, so the marginal cost of the first ~45 tokens is far below 0.229. (leader-21's reading; not measured directly.)
    - G1 not met.
validity:     leader-21 review: generic per the human ruling.
    - The match unit is 1 token (the finest the hash machinery gives).
    - The junction comes from live traffic only, with no warm-up.
    - The gain thresholds 16/64 come from the hash and GDN chunk granularity, not from the benchmark.
    - Shown on 4 different prefixes.
    - Correctness: CoW private copies, no shared writes. The replay-flag clear is required on the champion.
    - Exactness is at chunk-split level and must be re-run after the flag fix before any official use.
    - NOT official-eligible as is: GSM8K-50 and 128K not run, n=1 A/B within noise, and the code is not rebased onto T27-002.
