team:         T53
builds_on:    T53-003
performance:  "measurement only: tapered chunks LOSE to even C2 (down->AR saving -23.6..0 us vs C2 +8.7/+9.2); gate (>=25 us) FAILS"
summary:
  hypothesis: In C2 (T53-003) ~40 us of the down->AR chain comes after the GEMM: posting the last 2560-col chunk plus
              the peer reading it. Tapered chunks (big first, small last; split-K only on the small ones) should
              cut that tail to ~12-18 us with GEMM done at ~74-78 us, giving down->AR ~90-95 us (saving >= 25 us).
  change:     ext/t53c.cu v6: variable-width chunks (device spec int[16]: slabs and kseg per chunk; split-K slot
              stride = max kseg). benchC.py takes cfg [mt,un,ks,[[cols,kseg],...],P,post,npost,nread,pmode,npf];
              spec tensors are created before any chain runs. Same setup as T53-003: 2 GPUs, node-0 bound,
              warm-up first, cold, 12 reps; chain down -> AR -> qkvz; codec x8 posters, 8 readers (single host
              poller), 8 prefetch CTAs, P=80 GEMM CTAs (UN2, work balanced per CTA). Job 458, 1.0 GPU-min.
  results:    Observed (job 458, medians r0/r1; down->AR saving and full-chain saving are vs the champion hostar
              with codec + prefetch in the same job: 123.4/124.9 and 159.7/160.3 us):
              - C2 even [2560k1, 2560k1]: down->AR 114.7/115.7 (+8.7/+9.2); full +9.7/+9.7.
                Chunks done at 35.8 / 71.7; last chunk posted at 90.1; readers done at 111.6.
              - [3840k2, 1280k2]: 138.2/140.3 (-14.8/-15.4); full -14.3/-15.9. Chunks done at 65.5 / 90.6;
                readers done at 135.2.
              - [2560k1, 1920k4, 640k4]: 124.9/124.9 (-1.5/0.0); full -1.0/-0.5. Chunks done at 36.9 / 74.8 / 91.1;
                last chunk posted at 106.5; readers done at 121.9.
              - [3840k2, 640k4, 640k4]: 146.9/143.4 (-23.6/-18.4); full -23.0/-18.9. Chunks done at
                65.5 / 90.1 / 114.2; the last 640-col chunk is posted 10.7 us after done; readers done at 138.2.
              - Numerics: every AR result is bitwise equal to the sum of the two partials; ranks equal.
  conclusion: Interpretation: the prediction failed. GEMM done came at 89-115 us (predicted 74-78), and the
              smallest last chunk still leaves a ~24 us tail.
              - The data path is throughput-bound, not bound by the last chunk's size. From the first post to
                readers done, every variant moves ~1 MB of wire traffic per GPU at ~13-15 GB/s effective while the
                GEMM runs (hostar alone: ~23 GB/s). A small last chunk waits behind the earlier chunks' backlog.
              - Big-first delays the start of PCIe traffic.
              - Split-K small chunks run slower under the concurrent host traffic (the 1920-col kseg4 chunk takes
                38 us for 37.5% of the work).
              - Even C2 remains the best (~9-12 us per down boundary, T53-003). The producer-side direction is
                capped by PCIe throughput during the GEMM.
validity:     measurement only (no engine change).

leader review (leader-53, 2026-11-11): microbenchmark only, no engine/serving change; nothing to check against TASK.definition. Numerics reported (bitwise vs stock where reduction order matches; AR results bitwise vs fp32 partial sums). Accepted as measurement.
