# T27 write-up (leader-27, explore). Closed after the last chance

GPU: ~34 of 50 granted used (jobs 169, 170, 172, 174, 178, 179, 183, 185). Records: bank/T27-001..T27-003.
Worker: leader-27.w1. Notes: local_kb/leader-27.md. Proposal: proposals/T27-1.md.

## Direction
Raise accepted tokens per decode step (E) at a fixed M=48. Two parts:
- confidence-budgeted verification (T3/T5), used on decode-only steps only;
- cheaper draft steps, by screening the draft lm_head, so that a deeper K_MAX pays.

Payoff if it works was forecast at +6.8%; central +2.5%.

## What was measured
- **Draft lm_head screen, CPU only** (agents/leader-27/lr; artifact bank/T27-002/pca_train_n160.pt).
  - A data-free SVD of the lm_head fails: recall@64 is 0.38-0.69 at r=64-512.
  - A PCA of the drafter's own hidden states works.
    - Fit: 160 TRAIN self-generated sequences (T3's data).
    - Eval: 40 HELD-OUT sequences, 14,138 response positions.
    - At r=1024, top-64: recall .9963; a1 .7796 vs .7805, a2 .5593 vs .5605 (E -0.15%).
    - At r=512, top-64: E -1%.
  - A random within-held-out split had overstated recall (.994 at r=512). Always split by sequence.
- **Fused screen kernel** (T27-002, job 179; M=16, GPU0, read flush, CUDA graph): 115 us vs 336 us for the champion's draft lm_head.
  - Parts: h@P 19 us, FP8 U scores 58 us (near the ~50 us byte roof), chunked top-64 19 us, exact FP8 rescoring 24 us.
  - The unfused torch version took 289 us.
- **Screen in server at k=2** (T27-002, job 183; ABBA, 300 prompts, fresh starts):
  - A 939.0 / 931.5, B 937.5 / 947.8: +0.8% pooled.
  - Not all B > all A. A repeat ABBA (job 186, order B A A B) then gave both B > both A, +0.7%; pooled over 8 starts, +0.76%. Requested official.
  - Other numbers: ITL median 25.4 -> 24.9 ms; acceptance 2.335 -> 2.323 (-0.5%); 128K pass; gsm8k50 88%.
  - Interpretation: the expected ~-0.44 ms/step (~+1% e2e) is consistent with this, but it is not separable from noise at 2+2 starts.
- **Budgeted verification on the champion** (T27-001, job 174; K_MAX=3, B=32, decode-only steps, greedy-ids path):
  - In-situ E +5.0% (2.468 vs 2.350); decode step +0.86 ms (T5: +0.96).
  - End to end -2.6% (single runs), with TTFT p50 +17% and ITL mean +9.7%.
  - Gating the budget to decode-only steps does NOT remove T5's e2e penalty.
  - Exactness on the greedy-ids ragged path: 0 mismatches over 2778 rows. This closes T5's open gap.
- **Where the penalty comes from** (T27-003, job 185, n=1):
  - The adaptive path with a uniform k=2 layout and no ranking still loses: 811.5 vs 887.2 (-8.5%), TTFT +14%, ITL mean +10.5%.
  - Its decode step costs only +0.8 ms: the unused third draft step.
  - So the penalty lives in the adaptive plumbing, outside the decode step: per-step host work (compact_batch / reallocate_drafts), the adaptive cudagraph mode on prefill-containing steps, or the scheduler seeing k=3.
  - It is not isolated further. The next step is an nsys A/B.

## Forecast vs outcome
| part | forecast (payoff / central) | measured |
|---|---|---|
| E, K_MAX=3 budget | +8.4% / +5.6% | +5.0% in situ |
| budget step cost | +0.74 / +0.96 ms | +0.86 ms |
| screened draft head | 306 -> ~85 us | 336 -> 115 us (same-job reference) |
| mixed steps unchanged | 0 / -0.5% | an e2e penalty of ~4-10% remains, in the adaptive plumbing |
| e2e | +6.8% / +2.5% | budget -2.6%; screen alone +0.8% (noise level) |

My main error: I trusted the offline E (+8.4%) after T5 had already measured +5.6% in situ. With the in-situ E, the payoff if it works was ~+3.6%, below the bar, even before the plumbing penalty.

## Opinion / leftovers for an exploit owner
- **Screened draft head (T27-002).** It is exact on verification and allowed under ruling R1. It is ~+1% expected. A second same-job ABBA (~11 GPU-min) would settle it; then official. It stacks with anything outside the drafter.
- **Budgeted verification.** It is worth ~+1.6% decode tokens/ms (E +5.0% for +3.4% step) only once the adaptive-plumbing penalty is found. Combined with the screen (cheaper 3rd step), the ceiling is ~+3-3.5% e2e. Not an explore direction.
- **KB candidates:**
  - PCA-screened draft lm_head numbers (above);
  - the adaptive-verify e2e penalty lives outside the decode step and persists without ranking or ragged layouts (T5 and T27 both hit it);
  - the greedy-ids ragged path is exact.

## Official result (added at close)
T27-002 official #1: **967.3 tok/s** (trials 966.3 / 968.3, CV 0.15%). That is +1.1% over T14-004 (956.6; trials 955.8 / 957.5): both trials are above both of the champion's.
- GSM8K-250: 93.6%.
- TTFT p99 1141 ms; TPOT p99 33.0 ms.
- 128K request: pass.
- MTP acceptance: 2.328 (champion 2.33).
