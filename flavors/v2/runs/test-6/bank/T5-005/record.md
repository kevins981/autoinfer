team:         T5
builds_on:    T5-004 (t5-k3.mb) ; T3-008 (route B idea, T3 write-up)
performance:  unofficial: k=3 two-slot, full champion grid, KV 5.0 GiB: 859.8 tok/s (~1.00x champion, cross-job vs job 36's A runs; not paired)
summary:
  hypothesis: GDN can keep 2 state slots per group (committed S[0] + replay log S[1]) for any k (route B), bitwise-exact, so k>=3 fits 16
              requests at the default KV 5.0 GiB with the full prefill-graph grid.
  change:     VLLM_GDN_TWO_SLOT (default 1, k=2..3): num_speculative_blocks=1 under replay; builder/capture width 2; replay fallback slot 0;
              flush stores the final state only into S[0]; align copies bias 0 for the temporal state (conv keeps its bias);
              causal_conv1d_update max_query_len = num_spec+1 (it was taken from the state-index width); startup assert that GDN layers are
              replay-capable; kernel host check width==2 + device trap. Notes: agents/leader-5.w1/routeB.md, routeB_impl.md.
  results:    Observed (job 39, 7.8 GPU-min; job 38 cancelled at 0.2; built _C verified, gdn_replay_op_available True, override unset):
              (i) k=3, T=0, 600 new tokens each, two-slot vs stock slots on the same build: tokens equal AND per-token logprobs bitwise equal
                  for 4 generations; prompts 1480/1560/3129/3129 (synthetic generic text) cross the 1600/3200-token block boundary at generated
                  token 120/40/71/71; gen 3 = gen 2 repeated with a real prefix-cache hit (first 1600-token block cached).
              (iii) two-slot k=3, KV 5.0 GiB, full grid, 150 prompts c=16: Running 16 (x2), 15+1 waiting (x1); 0 OOM; pool 276,547 tokens vs
                  259,441 for stock-slot k=3 in the same job; 859.8 tok/s; TTFT 281/1210, TPOT 16.84/38.9 ms; acceptance 2.695.
              (iv) 128K request pass (59.4 s), health 200.
  conclusion: Interpretation: route B is exact (bitwise across block boundaries and a prefix hit) and removes the memory blocker for k>=3 without
              trimming graphs. The pool-token count rises only 6.6% (my 13->7 blocks/req estimate overstates the freed GiB; admission of 16
              works anyway). It is the prerequisite for budgeted verification with K_MAX>=3. Fixed k=3 itself remains ~flat vs k=2.
validity:     bitwise-exact by test and by construction (same math, different slot); memory change covered by 16-req admission, 0 OOM, 128K pass.
              Env var to be registered in envs.py. Not an official candidate on its own (no speedup).
