# Round 11 report from leader-2 (T1): small-exact-wins stack -> null
No new best; T1-030 (897.3) stays champion. GPU 22.9/80 (57.1 unused); round closed early.

Experiment:
- T1-033 (w20) = T1-030 + T1-028 (cheaper no-op GDN flushes) + T1-032 (Marlin single launch) + every-16 prefill graph grid over 128-512.
  Exact (T1-028 unit tests re-run: 0 mismatches, negative controls fail). Graphs 2.59 GiB/rank (+0.18), KV unchanged, 128K pass,
  gsm8k50 88%. Two guarded same-job ABBAs (8 fresh starts, 200 prompts c=16): A 849.4/857.1/848.2/848.4, B 849.7/853.4/846.9/850.9;
  pooled A 850.8 vs B 850.2 (-0.07%). TPOT/ITL/acceptance flat. Estimate +0.5-1%; measured ~0. Not adopted, not evaluated.

Leftover budget: not spent. The remaining small levers I know are each <= ~0.3% (leaner AR encoder ~1 us/AR at decode; M=16 Marlin
config ~12 us/step; mixed-step replay ~0.1%), below what 2 ABBAs resolve; spending GPU on them would not produce a decision.

Proposed SHARED_KB:
- KB-a: Stacking T1-028 + T1-032 + an every-16 grid over 128-512 on T1-030 gives no measurable gain (pooled 8 starts -0.07%, resolution
  ~±0.3%); sub-0.5% exact tweaks no longer add up visibly at this level. (T1-033)

Where the time is now (T1-030; KB41 + T1-030's AR savings): decode step ~26.5 ms at 48 tokens: Marlin ~14.4 ms (cold at HBM roof,
warm at Marlin's own data-path limit, KB28/KB39), host AR ~5.5 ms (PCIe floor, compressed), GDN replay ~2.1 ms (HBM roof), lm_heads
~2.2 ms (roof), attention 0.3; mixed steps: prefill GEMMs at the tensor roof, prefill ARs compressed and partly hidden (T1-025).
Next (one direction): a fresh brainstorm. Every component we've attacked is at a measured floor; remaining levers need a structural
change of what is computed per output token within the rules (e.g. higher MTP acceptance from a better draft at equal cost — the
target bounds it, 2.33 of 3 leaves room; run-1's "trained draft vocab mapping"/tree drafts; or reducing per-step fixed work), and
should be estimated against the floors above before any GPU time.
Notes: worker w20 stopped. Workspace kept: leader-2.stack11 (T1-033).
