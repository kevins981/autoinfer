# T7 sweep, step 5: my map vs the other maps and the history (read after 4-knobs.md)

| My candidate | Status in the run | On what code |
|---|---|---|
| S1 k=3 drafts | T1-004 (job 23): -13.8% one start, KV capacity-capped at 12 running (4.25 GiB). T1-009 (job 43): 764.2, one start, ~neutral vs T1-007 at 5.5 GiB; acceptance 2.698 (0.768/0.538/0.383). T4/T5/T6 maps: "memory-blocked" (GDN spec slots ~3.6 GB of 4.4 GiB). | T1-009 is old code: bf16 draft lm_head (1.48 ms/step at M=16 vs 0.78 now), no DeltaLog, no L2 prefetch / Triton attn. **Untested on the champion.** |
| S2 k=4 | untried | needs kMaxLogRows=4 |
| PP=2 | T-? maps priced it; not built in the run (as far as the bank shows) | - |
| Window: split r for AR overlap | T6-002/003: row-split GEMM->AR in mixed steps, +0.28% (noise) | T3-007 |
| Materialize trim / DL on mixed steps | T5 writeup #1 (~+0.5-1%), untried | - |
| Freeing spec state slots (log outside slots) | T4 #, T5 writeup #2: unblocks k=3, untried | - |

## Differences
- **What my phase-1 map missed:** the capacity dimension of Speculate. I priced k=3 in time only; the champion's KV budget counts the GDN
  spec-state slots (k+1 fp32 states per request per GDN layer, ~75 MB/GPU each), so k=3 needs ~5.5 GiB for 16 running requests
  (T1-004/T1-009). The framework's Keep table has capacity C, but my sweep didn't apply it to the spec slots: a framework-use miss.
  My first harness submission (4.4 GiB) was cancelled before it ran (job 122) once I saw this.
- **Where I disagree with "k=3 memory-blocked":** it is not blocked, it is a trade: the champion uses 21.06 of 22.89 GiB at 4.4 GiB KV
  (weights 14.81, graphs 1.85; T5-001 eval/1 server.log), so +1.1 GiB KV (5.5 GiB) leaves ~0.7 GiB. T1-009 ran 5.5 GiB without OOM.
  And the per-step cost of the extra draft has dropped since T1-009 (FP8 draft head -0.7 ms/draft step; DeltaLog cheaper per row), which
  is roughly the margin by which T1-009 was neutral. So k=3 + 5.5 GiB deserves a fresh measurement on the champion.
- What others have that mine lacked: rank skew / sync costs in the AR (T1, T6), L2 write-back term (T5). Adopted as model notes.
