team:         T2
builds_on:    T2-001
performance:  "+2.5% vs the champion (T2-001, 689.5): 707.0 tok/s (starts 708.9, 705.1), eval/1 (GPU job 14)"
summary:
  hypothesis: BASE/T2-001 all-gather full logits over the vocab split for the target verify (48 rows x
              124k x bf16 = 12 MB, 0.95 ms) and for each of the 2 draft steps (4 MB, 0.33 ms each) every
              decode step (reports/T2-sweep/3-calibration.md A). Two existing vLLM options remove most of it:
              (1) --enable-batch-sharded-sampling: each rank samples/verifies half the requests; logits
              exchanged by all-to-all (each rank receives full-vocab logits only for its own requests:
              half the bytes); (2) speculative-config use_local_argmax_reduction: the drafter's greedy
              argmax is computed per vocab shard and only (max, index) pairs are gathered.
              Napkin: ~0.5 ms (target) + ~0.6 ms (drafts) per step -> ~1.1 ms x ~2,500 steps = ~2.7 s (~+2%).
  change:     serve flags only (serve_args.txt): T2-001's --compilation-config line +
              --enable-batch-sharded-sampling +
              --speculative-config '{"method":"mtp","num_speculative_tokens":2,"use_local_argmax_reduction":true}'
              (method and draft count as in BASE). No code change.
  results:    Observed (harness, 2 fresh starts x 400 prompts, GPU job 14): 708.9 / 705.1 tok/s, mean 707.0
              (+2.5% vs T2-001 689.5; both starts above). TTFT p50/p99 196/1185 ms; TPOT p50/p99 20.8/38.6 ms;
              gate_latency pass; 128K request pass; MTP acceptance 2.349 (T2-001 2.337, BASE 2.345).
              Server log: "Using local argmax reduction for draft token generation" and "Batch-sharded
              sampling enabled" on both ranks.
              GSM8K not run: the change is exact (see validity).
  conclusion: Interpretation: gain matches the napkin (~+2%); similar aggregate acceptance (2.349 vs 2.337).
              Communication savings remain the mechanism hypothesis: the combined run establishes the gain
              but does not isolate each flag's contribution; not profiled.
validity:     leader-2: flags of vLLM's parallel/speculative config. Target: the same full-vocab logits
              are formed for each request (on its owner rank) and sampled by the same sampler/rejection
              code: exact. Drafter: greedy argmax over the full vocab computed as max over shard maxima
              (exact; tie-break: lowest index within a shard, lower rank across shards = lowest global
              index, like torch.argmax); draft side only, verification unchanged. No precision, KV, model,
              parser or prefix-caching change; memory: tested with the 128K request (pass).
              Generic, not benchmark-keyed. Draft equivalence holds for finite logits (get_top_tokens does not
              sanitize NaN like greedy_argmax; target verification still protects output). No end-to-end
              bitwise-equality claim (owner sorting may change GEMM row order).
review:       reviews/T2-2: numbers/gates recomputed (708.935/705.114, +2.548%; both starts above; latency and
              128K pass); target path keeps full-vocab logits per request and standard sampler features
              (penalties, grammar, logprobs, prompt logprobs, non-greedy); draft tie-break = lowest global id;
              GSM8K skippable under the exact-change exemption. Three wording corrections applied above.
