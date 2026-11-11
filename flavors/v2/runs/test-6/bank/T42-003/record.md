team:         T42
builds_on:    T37-002 + T42-002 (the certified lm_head screen, now in-server)
performance:  "unofficial: ABBA vs T37-002 mixed, +0.53% (B 1125.1/1116.4 vs A 1116.5/1113.3, job 411); ITL p50 26.0 vs
              26.23 ms. c=1 greedy outputs identical to T37-002 (10/10). gsm8k-50 92%. 128K long request passes. HBM
              -0.34 GiB per GPU." (DRAFT by leader-42.w1; validity review pending: leader-42)
summary:
  hypothesis: T42-002's certified screen can replace the target lm_head on each rank in the server. Greedy decisions
              stay bitwise those of argmax over the baseline's bf16 logits. HBM goes down (plane 0.57 GB instead of
              the 0.99 GB encoded shard). The verify step's lm_head gets ~0.4 ms per step faster.
  change:     Files: change.diff = `ws diff t42-cs`. delta_vs_T37-002.diff = the part this record adds on top of T37-002.
              New: vllm/model_executor/layers/lmh_cert.{py,cu}. JIT extension, build dir named by source hash.
              - Load (model_runner, after the drafter copied its FP8 subset head): maybe_certify_target_lm_head.
                - Builds the 7-bit plane and the bound coefficients A_v/B_v from the raw shard.
                - Plane decode verified bitwise against q.
                - Raw bf16 shard copied to plain host pages, first-touched by the NUMA-self-bound worker, then
                  cudaHostRegister(mapped). Bitwise check vs the GPU shard. Placement logged from
                  /proc/self/numa_maps.
                - Self-test: 201 rows, identical on both ranks (rank 0's vocab rows x400, row pairs, rows + noise at
                  M=48/17, gaussian, one-hot ties). Decisions combined across ranks vs the exact full-shard logits
                  (lmh_codec's chain, from the GPU copy). Must be 0 mismatches.
                - All ranks agree via all_gather of the ok flag. On failure: ERROR log and the lmh_codec path as before.
                - Then the raw shard is freed. The plane is not stored next to the codec; it replaces it.
              - Greedy verify (LogitsProcessor.get_greedy_tokens): per chunk of <= 64 rows:
                hprep -> screen -> all_gather of the per-row LB (64 floats) -> finalize with the cross-rank max ->
                gather candidate rows host->HBM (dedupe, cap 2048 staged rows; beyond: read directly from host) ->
                exact rescore with lmh_codec's mma chain -> fallback -> (val, idx).
                The cross-rank combine is unchanged: T37-002's own code (strict '>', lowest rank wins ties; first
                index within a rank). A rank with no surviving candidate reports (-inf, 0).
              - Bounded fallback: a row whose certificate can't be used gets the exact max over its whole shard.
                Triggers: emission > 65536, kept > 1024, pair list > 8192. The max is computed by rawgemm streaming
                the shard from host with the same chain, always launched and exiting at once when no row is flagged.
                Device counters (fallback rows, candidates) are copied async and logged every
                VLLM_LMH_CERT_LOG_EVERY calls.
              - Pure-greedy steps without drafts (e.g. a prefill-only step): model_runner._certified_greedy_logits
                feeds the sampler logits that are -inf except 0 at each row's certified argmax. The sampler then
                picks the same ids. Only used when every request is pure greedy (temperature 0, no logits
                processing, no logprobs), with no grammar and no drafts.
              - Anything else needing logits (non-greedy, logprobs, the startup dummy sampler runs): exact full logits
                streamed from host with lmh_codec's chain. Correct but slow (~0.1 s per call), warning logged.
              - The final-norm AR's L2 prefetch now targets the plane (lm_head_prefetch_tensor).
              - Not changed: the draft lm_head (T27 screen on the FP8 vocab subset).
              Serve args: T37-002's (serve_args.txt).
  results:    Observed (logs: agents/leader-42.w1/out/s2j1, s2j2/abba, s2j3; jobs 407, 411, 412):
              - Startup, both ranks: "certified ... plane 0.5662 GB + buffers 56.8 MB replace the raw 1.271 GB shard;
                exact bf16 copy pinned on host (node0: 310401 pages); self-test 201 rows 0 mismatches (1 fallback
                row, mean candidates 27.9 / 17.6)".
                Model loading 14.12 GiB (B) vs 14.46 GiB (A) per GPU.
                KV cache unchanged at 276,547 tokens, because --kv-cache-memory pins it at 5 GiB.
                Full-logit host calls: 10 at startup (dummy sampler runs), 0 afterwards. Server ready in 93 s vs 73 s.
              - c=1 greedy, 10 ShareGPT prompts, temperature 0, 300 max tokens, A and B in the same job: 10/10 texts
                (content + reasoning) identical; completion tokens identical.
              - gsm8k-50 on B: 92.0% (46/50).
              - Short bench, 100 prompts, c=16, on B: 1062 tok/s, acceptance 2.73. Certificate stats over 26,498 rows:
                rank 0 mean 1.99 candidates per row, max 89; rank 1 mean 0.02, max 3. 0 fallback rows.
              - ABBA vs T37-002 (abba.sh, 300 prompts, A,B,B,A): verdict "mixed: +0.53%".
                A 1116.5/1113.3, B 1125.1/1116.4 tok/s.
                ITL p50 A 26.23/26.23 ms, B 26.01/25.98 ms. TPOT p50 A 12.65/12.65, B 12.78/12.88.
                Acceptance A 2.671/2.673, B 2.690/2.672.
              - long_request 128000 on B: pass (200, 19 tokens, 44.9 s, health ok afterwards).
              - GPU used in stage 2: 15.9 GPU-min (jobs 407: 4.8, 411: 9.3, 412: 2.1).
  conclusion: Interpretation:
              - Exactness holds in-server: identical c=1 outputs, 0 fallbacks, normal gsm8k. The rank that doesn't
                hold the winner keeps ~0 candidates thanks to the cross-rank LB, so host traffic is ~2 rows per row
                on one rank.
              - Speed: a small, not established gain.
                Evidence for: ITL p50 is -0.9% in both B starts vs both A starts.
                Evidence against: throughput is mixed (+0.53%; B start 3 ties A start 1). TPOT p50 is slightly worse
                for B.
                Expected: stage 1 measured -0.42 ms per rank lm_head at M=64, i.e. ~1.5% of a 26 ms step. The
                observed ~0.5-0.9% suggests the added costs eat part of it. Those costs are the LB all_gather (one
                extra small collective in the eager sample path), more kernel launches in Python, and the host
                gather.
              - NOT measured, though the brief asked for it: the LB exchange's cost on its own, and an nsys
                breakdown of the new sample path. That is the next step if the team wants to push it. Candidates to
                cut: fuse the launches (hprep+screen, finalize+gather+rescore+fallback+decode), and overlap the host
                gather.
              - Memory: -0.34 GiB HBM per GPU. KV doesn't grow under the task's fixed --kv-cache-memory, so the
                headroom is only useful if a later change spends it. Pinned host memory: +1.27 GB per rank (node 0).
              - Cost of non-greedy / logprob requests: ~0.1 s per step (full logits from host). That doesn't affect
                the benchmark (temperature 0), but it is a performance cliff for other workloads.
validity:     (leader to review)
              - Decisions: the greedy path is exact by construction (T42-002 proof; the self-test runs at every start).
              - Prefill-only greedy steps: they use the certified argmax through one-hot logits, only when nothing but
                the argmax reads them.
              - Every other path computes exact full logits with lmh_codec's arithmetic.
              - No benchmark-specific data or tuning. The caps only route rows to an exact fallback.
              - HBM goes down (-0.34 GiB). The 128K gate passes.
              - Stated assumption: the tensor-core error bound (2^-11 relative over the chain; measured 2^-19.4).
