# local KB leader-27 (T27, explore): cheap deep drafting + budgeted verification (proposals/T27-1.md)
## Status
- Funded: 30 of the 80 now, 50 more when G1 passes. R1 ruling (PCA-screened draft lm_head) ALLOWED with conditions:
  self-gen data only, r/k selected on held-out, acceptance before/after, 128K test.
- Worker leader-27.w1: ws t27-bud = T14-004 + T5-006 budget (decode-only) + two-slot GDN; env T27_BUDGET=0 = champion.
## Measurements
- CPU, recall.py/recall2.py: data-free SVD screen fails (recall@64 <= 0.69 at r=512). PCA on drafter hiddens with a
  random within-heldout split looked great (0.994 @ r512/k64), but that was optimistic.
- CPU, fit_eval.py (PCA fit on 40 TRAIN seqs, eval on 40 HELD-OUT seqs, 14,138 response positions), cumulative acceptance:
  exact a1 .7805 a2 .5605 | r512 k64 .7720/.5504 (recall .975) | r512 k128 .7763/.5552 | r1024 k64 .7793/.5594 (recall .995)
  | r1024 k128 .7802/.5601. -> pick r=1024, k=64-128 (E loss ~0.1-0.2%). More training seqs untested (360 available).
- GPU job 169 (GPU0, read flush, torch ops in a CUDA graph, local vocab 49152): screen total (M=16) r256 149 us, r512 202,
  r1024 289 us; approx-logits alone r512 93, r1024 181 (torch bf16 GEMM at M=16 is far below its byte roof: 100 MB in
  181 us). Full bf16 lm_head 539 us. A fused kernel (FP8 U, fused top-k, gather-dot) is needed to reach ~85 us at r=1024.
- 2026-11-09: human allowed generic hybrid prefix caching; T21 funded (prefill side). It should stack with T27 (decode E); recheck the stacking if T21 lands a champion.
- Job 174 (bank/T27-001): budget K3 B=32 decode-only on the champion, greedy path: tok/s -2.6%, E +5.0%, decode step +0.86 ms,
  TTFT p50 +17%, ITL mean +9.7%: the same unexplained e2e penalty as T5 bud3; decode-only gating doesn't fix it.
  Direction is below the bar on measured E. Resync requested; proposed: screened lm_head at k=2 alone + one split run + write-up.
- Resync: LAST CHANCE, no new grant; <= ~20 GPU-min for (a) the fused screen at k=2 and (b) the split run, then write up. No replacement team.
- PCA refit on 160 train seqs (fit_eval_n160.out): r1024 k64 recall .9963, a1 .7796 / a2 .5593 (exact .7805/.5605);
  r1024 k128 .9982, .7801/.5598. Artifact lr/pca_train_n160.pt. Task W1-2 sent (agents/leader-27/task_w1_2.md).
