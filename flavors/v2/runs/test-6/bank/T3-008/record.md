team:         T3
builds_on:    T1-042
performance:  n/a (feasibility/measurement record; no serving change; change.diff is T3-001's, copied as the empty/no-op diff)
summary:
  hypothesis: K_MAX=3 (num_speculative_tokens=3) can be served at 16 concurrent requests on the champion stack with a KV pool grown just enough (13 blocks/req x 16 = 208 blocks, ~5.62 GiB), so T3-2 (B=2*reqs budgeted verification) is buildable without GDN-slot changes.
  change:     none to serving code. Jobs on built workspaces workspaces/t3-distill (real _C, 36,106,328 B) and t3-distill.t (= t3-distill + env-gated timers, `ws build` -> real _C). Plan: agents/leader-3.w1/PLAN_T3-2.md.
  results:    Observed:
    - k=3, T1-042 flags, --kv-cache-memory 5.75 GiB (job 21, t3-distill): CUDA "out of memory" during graph capture (PIECEWISE prefill-graph instantiate), pool logged 298,627 tokens (2.28x). Jobs 20 (first try) died earlier on a script bug (shell var CC leaked into nvcc; no GPU measurement).
    - job 23 (t3-distill.t): k=3 at 5.65 GiB (6,066,641,305 B; pool 293,222 tok, 2.24x) OOM in graph capture (4 OOM lines); at 5.70 GiB (6,120,328,396 B; 295,925 tok, 2.26x) OOM (2 lines). No k=3 server started at >5.0 GiB, so Running:16 and long_request 128000 were never reached.
    - Pool scales linearly with --kv-cache-memory: ~185 blocks at 5.0 GiB (1.98x at k=3; 2.04x at k=2 with 10 blocks/req), 208 blocks needs >= ~5.62 GiB. Window between need (5.62) and failure (<=5.65) is empty.
    - Built k=2 champion timers (job 23, 100 ShareGPT prompts, c=16, 864.3 tok/s, MTP acceptance length 2.368, Running 15-16): fwd M=3 17.44, M=18 18.20, M=27 20.54, M=33 20.91, M=36 21.72, M=42 21.91, M=45 22.34, M=48 22.40 ms; sample/reject 1.07-1.12 ms; propose 1.49 (1 req) .. 1.68 ms (16 reqs) = 0.75-0.84 ms per draft step. Base step at M=48 = 22.40+1.12+1.68 = 25.2 ms. Raw: bank/T3-008 notes + agents/leader-3.w1/jobs/step0c/k2/timers.*, analysis anal_timers.py.
    - Not re-measured (k=3 won't boot at 16 reqs): built k=3 timers, the cliff above M=48.
    - Sim (sim_budget2.py MEAS=2, built costs: fwd 22.4+0.15(M-48) for M<=48, cliff above 48 UNMEASURED/assumed old shape, drafter 0.84/step, sample 1.12; held-out batches, 20000 draws): fixed k=2 E 2.348, 25.20 ms. K_MAX=3 B=32 (M=48): raw E 2.539, 26.04 ms, ratio 1.0463; iso 1.0486; lr 1.0482. K_MAX=4 B=32: raw 1.040, iso 1.045. B=40/48 (M>48, cliff assumed): 0.969/0.961. Output: notes/sim2_built_costs.txt.
    - GPU used by step 0: ~11.7 GPU-min (allowance was 8).
  conclusion: Interpretation: K1 triggered: with the champion's prefill-graph list, k=3 at 16 reqs does not fit; the 5.0 GiB pool admits only 14. The remaining routes are fewer prefill graphs (~+1% prefill worth per KB34, a flag change needing review) or touching GDN state-slot logic (risky). Built costs leave the simulated K_MAX=3 B=32 gain at about +4.6-4.9% decode-step (the M=48 point is unaffected by the unmeasured cliff), roughly +3-3.5% end-to-end after overheads, before 3-4 days of ragged-path coding. Recommend stopping T3-2.
validity:     No serving change; env-gated timers only.
