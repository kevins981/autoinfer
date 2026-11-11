team:         T43
builds_on:    T37-002
performance:  "measurement record" (no serving change); stage-1 gate of T43-1 FAILED: mid-forward signals too weak; confidence-only row masking models at <= +0.6% decode (AR-only)
summary:
  hypothesis: (proposal T43-1, "decision depth") the target's verdict on a draft token may become predictable at an
              intermediate layer, which would let a row-selective all-reduce stop paying for doomed draft rows mid-forward.
  change:     measurement-only instrumentation on top of T37-002, gated by env T43_DD=<dir> (off by default):
              - vllm/t43_dd.py (new): on pure verification steps (rows grouped by 4 with consecutive positions, no prefill
                continuation), take the full residual stream after layers 8/16/24/32/40/48/56 (hidden+residual; the deferred
                TP partial all-reduced on a clone) and the final post-norm state; logit lens = final norm (model.norm) + this
                rank's lm_head shard (the serving head path `_apply_head`); save top1/top2, the next draft's logit (combined over
                ranks via gloo on the TP cpu group) and the count of local vocab entries above it; raw h_L (rank 0, L=16/32/48).
              - vllm/models/qwen3_5/nvidia/model.py: calls the above inside the layer loop (only when T43_DD is set).
              - vllm/v1/worker/gpu/spec_decode/speculator.py: with T43_DD set, after the draft's local-argmax tokens, also
                computes the full drafter logits to record the drafter's softmax prob of each draft.
              Requires --enforce-eager (Python hooks do not run inside CUDA-graph replay). Worker leader-43.w1, workspace t43-dd.
  results:    Observed:
    - job 395 (smoke, 8 prompts x 64 tok): eager start 47 s, gdn_replay_op_available True, no errors.
      (job 394 crashed on a bug in my lens call, fixed; 0 data.)
    - job 396 (full, 2.0 GPU-min): T3's self-generated prompts (agents/leader-3.w1/data/prompts.json, first 96; provenance
      bank/T3-001), chat completions, T=0, max_tokens 256, c=16, champion serve args + --enforce-eager. 96/96 ok,
      23,128 completion tokens. 512 verification steps recorded (7,521 request-steps; the last <16 unflushed steps lost).
      Data: agents/leader-43.w1/dd/full/ (841 MB; formats in agents/leader-43.w1/dd/README.md).
    - Consistency: server counters: acceptance per position 0.726/0.477/0.323 (mean accepted length 2.527);
      reconstructed from the dump's final logits: 0.725/0.479/0.325 (2.529). Lower than the benchmark's ~2.67 (different
      prompt mix, half think-mode, 256-token cap).
    - Logit lens, "reached" draft rows (all earlier drafts accepted), n=16,577 (5,080 rejected), recall / false-drop of the rule
      "flag reject if the draft is not top-1 of the lens at L": L8 0.987/0.970, L16 0.985/0.960, L32 0.972/0.929,
      L48 0.953/0.882, L56 0.937/0.778, final 0.984/0.000. With "not in top-20": L32 0.842/0.726, L56 0.678/0.466.
      Margin threshold (top1 - draft logit) at 1% false drops: recall 0.014-0.023 at every L<=56.
      AUC(accepted vs rejected) of the lens margin: 0.60 (L8) ... 0.59 (L48), 0.66 (L56), 0.99 (final).
      Per-position tables (d1/d2/d3) in dd/full/analysis_lens.txt; AUCs in dd/full/analysis_margin_auc.txt.
    - Linear probe (logistic regression on normalized raw h_L of row j-1 + position one-hot, step-block split 12,147
      train / 3,455 test): test AUC 0.678 (L16), 0.699 (L32), 0.747 (L48); recall of rejections at 1% false drops
      0.034 / 0.044 / 0.068; at 5%: 0.14 / 0.15 / 0.23 (dd/full/analysis_probe.txt).
    - Drafter confidence (softmax prob of the draft under the drafter, full gathered draft logits; this differs from the
      served fp8 local-argmax token in 292 of 28,017 drafts): AUC 0.867 / 0.871 / 0.882 for d1/d2/d3; recall of
      rejections at 1% false drops 0.15 / 0.16 / 0.15, at 5%: 0.38 / 0.36 / 0.39 (dd/full/analysis_conf.txt).
  conclusion: Interpretation: on this model the target's verdict is not readable from the residual stream before the last
    layers with the logit lens: through layer 48 the lens separates accepted from rejected drafts barely better than chance
    (AUC ~0.6), and even at L56 it is weak. A trained linear probe does a bit better (AUC 0.75 at L48) but still recovers
    under 7% of rejections at 1% false drops. The drafter's own confidence, available before the forward, separates much
    better (AUC ~0.87). So with these probes the "decision" is late, and mid-forward pruning looks unpromising; the
    pre-forward confidence signal is the stronger lever. Not tried: nonlinear probes, probes that see the draft token
    (e.g. h_L . lm_head[draft] with a learned per-layer map, tuned lens), combining confidence with h_L.
    Caveats: ties in bf16 logits make ~1.5% of rejected rows show rank 0 at the final layer; eager mode, not the graph
    path (the same kernels, so the dump numbers should hold).
  leader analysis (leader-43, CPU, dd/full/conf_rows.npz; script inline in the T43 resync report): value of pre-forward
    confidence masking with a row-selective AR. Rule: per request-step, cut drafts from the first draft whose (per-draft or
    chained) drafter prob < tau; accepted length capped at the cut. Step model T37-001 (16.9 + 0.10*rows + 0.83*draft steps,
    25.8 ms at k=3); a dropped row saves 0.06 ms (AR bytes only, fixed graphs) or 0.10 ms (rows also compacted out of the GEMMs).
    Best: chained prob, tau 0.15-0.2: 0.32-0.45 rows saved and 0.015-0.027 tokens lost per request-step -> decode +0.6%
    (AR-only) / +1.4-1.7% (with compaction, which needs the ragged plumbing that cost budgeted-K -8.5%, KB85). Per-draft prob is
    worse (best +0.35% AR-only). Break-even: a draft row is worth keeping if P(useful) > ~0.06/0.61 = 10% (one token is worth
    ~0.61 ms of step time); few rows are that hopeless.
    Interpretation (wording corrected after sparring T43-2): logit lens and linear probes find little rejection signal before the
    final layers; this does not prove early states lack it (lens AUC 0.59 at L48, 0.66 at
    L56, 0.99 final), so mid-forward pruning (HiSpec/FASER-style, compute or communication) has nothing to act on, and the
    only early signal (drafter confidence) is already the budgeted-K signal; even free of plumbing it is worth <1%.
    Also implies: early-exit self-drafting from intermediate layers of this target (LayerSkip-style) would draft badly.
validity (leader-43): reviewed the description and the 3-file diff summary: measurement-only, env-gated, inert by default;
    acceptance reconstruction matches server counters (0.725/0.479/0.325 vs 0.726/0.477/0.323). Not for official eval.
validity notes (worker): measurement-only; with T43_DD unset the added code paths are not
    taken (one env check per forward and per draft step). With it set, all dump work runs on clones/new tensors; the extra
    all-reduce and gloo all-reduce are out of place; the forward's tensors are unchanged. Not meant for an official eval.
