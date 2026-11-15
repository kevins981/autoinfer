team:         T1
builds_on:    BASE
performance:  "unofficial: stopped after 1 start, 602.5 tok/s (-5.7% vs BASE mean 639.0)"
summary:
  hypothesis: MTP with 3 draft tokens raises tokens/step (2.345 -> ~2.7) more than it raises step cost (napkin 0..+5%).
  change:     serve flag only: --speculative-config '{"method":"mtp","num_speculative_tokens":3}'. No code change.
  results:    Observed: unofficial_eval --ws leader-1.base --label t1-mtp-k3 --long --stop-below 625 (GPU job 6): 1st start 602.5 tok/s, stopped (below 625); MTP acceptance 2.701 (napkin guessed ~2.71); p99 TTFT 2134 ms / TPOT 62.7 ms (BASE official 1341 / 46.1: both p99s exceed the +20% latency gate). 128K/GSM8K not run.
  conclusion: Interpretation: acceptance came out as predicted (+15%), but per-step cost rose ~22% (602/639 = 0.943 = 1.152 / 1.22), well above my napkin (+15%). Likely: the 3rd draft step (full 1.27 GB lm_head read + logits gather + AR) and 64-row verify (AR bytes +33%, GDN per-row work), and in mixed steps more rows. Also fails the latency gates on its own. A screen negative after 1 start, on BASE costs: it may turn positive only if draft steps and ARs get much cheaper (FP8/reduced-vocab draft head, local argmax, faster AR). Not worth re-testing until those land.
validity:     exact (greedy verification unchanged, only the draft count); memory change not yet 128K-tested. Fails latency gates as measured.

CORRECTION (leader-1, after T2-007's finding): the server ran with at most 14 concurrent requests ("Running: 14 reqs" in unofficial/ws/t1-mtp-k3/1/b/trial1/server.log), not 16: with 3 drafts each request needs more GDN/KV state slots in the fixed 5 GiB pool. So the -5.7% is partly lost concurrency, not only per-step cost; my "step cost +22%" interpretation is wrong as stated. The latency-gate failure (TTFT p99) is consistent with queueing. See T2-007 (k=3 on T2-006, retry with a larger KV pool).
