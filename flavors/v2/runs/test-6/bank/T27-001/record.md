team:         T27
builds_on:    T14-004 (+ T5-006 budget plumbing and two-slot GDN, ported by 3-way merge; workspace t27-bud, NOTES.md)
performance:  "unofficial, same-job single runs (job 174, 100 prompts c=16): ON 850.6 vs OFF 873.1 tok/s (-2.6%); decode step +0.86 ms, E +5.0%: NO gain"
summary:
  hypothesis: confidence-budgeted verification (K_MAX=3, B=2*reqs, M=48) applied only on decode-only steps raises accepted tokens/step with no mixed-step penalty.
  change:     T5-006 ported onto T14-004; budget only on decode-only steps (mixed steps verify min(sched,2) drafts/req = stock k=2 layout, GDN uniform shortcut kept). T27_BUDGET=0 = champion. Fix during testing: rejection_sampler.supports_greedy_ids env default.
  results:    Observed (job 174; ON/OFF fresh server each, single runs):
    tok/s 850.6 / 873.1; MTP acc 2.468 / 2.350; TPOT p50 17.06 / 16.15; TTFT p50 242 / 206, p99 1154 / 1130; ITL mean 39.7 / 36.2, median 26.2 / 25.4
    decode-only step (M=48, 16 req) median 26.19 / 25.33 ms (+0.86); 15-req 26.2/25.4
    n_i on decode steps 0/1/2/3 = 11.6/21.3/22.5/44.6%
    exactness: greedy-ids path exercised (greedy 247, generic 3 calls), 0 mismatches over 2778 rows, c=4 max_tokens-only client; gsm8k50 88.0% (job 170, generic path).
    Invalid earlier run (job 170, greedy path disabled by port bug): 824.7 vs 881.3.
  conclusion: Interpretation: E +5.0% for +3.4% decode step gives ~+1.6% on decode tokens/ms, but e2e is -2.6% and TTFT/ITL-mean/p99 TPOT are worse, so non-decode-step time grew; cause not isolated.
validity:     (leader-27) verification unchanged: every admitted token is verified by the target with the full-vocab head (exact); greedy-ids path per-row check 0 mismatches over 2778 rows. Draft-side only: budget selection. No benchmark-keyed behavior (budget = 2 x num_reqs, generic). Memory: two-slot GDN as in T5-005/T25-003; 128K not run (not proposed for official). Valid as an experiment; NO gain.
