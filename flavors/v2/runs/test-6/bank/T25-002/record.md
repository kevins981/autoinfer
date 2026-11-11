team: T25
builds_on: T14-004 (+ T5-005 two-slot GDN ported, VLLM_GDN_TWO_SLOT default on)
performance: "unofficial same-job ABBA (job 160, 300 prompts c=16): A 935.0 (939.0, 930.9), B 931.0 (929.6, 932.4): B-A -4.0 tok/s (-0.4%), within noise: NO gain. [older: 64 prompts c=16 (job 159): 766.5 tok/s, TTFT p50 218 ms, TPOT p50 16.1 ms, MTP acc 2.366 (champion build job 157, same 64-prompt bench: 771.3 tok/s; not a valid A/B)]"
summary:
  hypothesis: an uneven MLP TP split (GPU1 gets more intermediate columns) is implementable with exact-up-to-reduction-order output and fits in memory.
  change: env AI_MLP_SPLIT=<rank-1 share> (tp=2, default off = unchanged path). Rank0 gets cols [0,n0), rank1 [n0,17408), n1=round(share*17408/128)*128. 0.5367 -> n0=8064, n1=9344 (G1's fast-config pair). Edits: linear.py (mlp_uneven_partition, _uneven in Merged/RowParallel, loader offsets, per-param uneven_start), parameter.py (uneven_start used by column/merged/row loaders; block scales /128), qwen2_moe.py (flag + startup log), models/qwen3_5/nvidia/model.py (uneven_mlp for the target MLP; MTP drafter MLP stays even via "mtp" in prefix). Memory: T5-005's two-slot GDN ported (delta t5-k3 -> t5-rb for 9 files, agents/leader-25.w1/S1a/twoslot.patch; applied cleanly, ws rebuilt) and --kv-cache-memory 4616880128 (4.3 GiB).
  results: Observed:
    job 157 (7.2 GPU-min): split NOT active (flag was in the wrong model file); numbers there = champion build: model 14.37 GiB/rank, graphs 2.46 GiB, KV 267,721 tokens, 64-prompt c=16 771.3 tok/s, gsm8k50 88%, 128K pass. Mishap, no split data.
    job 158 (7.1): split active, KV 5.0 GiB, no two-slot: model 13.74 GiB (rank0) / 14.90 (rank1), graphs 2.44/2.47; rank1 CUDA OOM at runtime (23.51 GiB used, 1.75 MiB free) -> HTTP 500 for all requests. Champion has <0.5 GiB rank-1 headroom.
    job 159 (3.2): split + two-slot + KV 4.3 GiB: model 13.74/14.90 GiB, graphs 2.44/2.47 GiB, KV pool 237,658 tokens (1.81x of 131072; champion 267,721), nvidia-smi after run 22,480 / 23,738 MiB (rank0/rank1); 64-prompt bench 0 errors, Running 16 Waiting 0, peak KV usage 68.3%; gsm8k50 88.0% (44/50, same as BASE/champion); 128K request pass (56.4 s). 
    Exactness (offline, same Marlin kernels, random weights, even 8704/8704 vs uneven 8064/9344, fp32-accum sum of partials rounded to bf16): max abs diff 256/256/512 at M=48/256/640 (|out|max 3.2e4/6.0e4/7.8e4), max 2 bf16 ulp where |ref|>1% max, 79% elements bitwise equal.
    Not measured: speed benefit (64-prompt single run is noise-level; needs ABBA/official); split=off bitwise equality with champion only argued from code (env-guarded None paths), not run.
  conclusion: Interpretation: the uneven split loads, runs, passes gsm8k50 and 128K with two-slot + 4.3 GiB KV (rank1 now nearly full: 23.7 of 24 GiB). The peak KV usage in this short bench (68%) is not the 16-sequence long-run peak: the champion's 83.8% was from the full 300-400 prompt runs, so admission stalls (Running<16) still need checking in a long run. Note two-slot and smaller KV are part of this build, so the ABBA baseline should be the champion with the same two-slot+KV (or accept the confound).
validity: Math unchanged except fp reduction grouping of down's K (<=2 bf16 ulp class, see above); MTP drafter unchanged; two-slot is T5-005's (bitwise-exact by their tests). Default AI_MLP_SPLIT unset = original behaviour; two-slot default ON in this build (VLLM_GDN_TWO_SLOT=0 restores).

Not fitted to the benchmark run: AI_MLP_SPLIT default 0.5367 (tp=2 only) is derived from hardware (GPU0 SM clock locked 2235 MHz vs GPU1 ~2520-2685) and the T25-001 microbench balance fit, chosen before any end-to-end run; derivation is in the code comment in linear.py mlp_uneven_partition.

ABBA (G2, job 160, 10.2 GPU-min, order A B B A, fresh start each, 300 prompts c=16, no CPU-heavy work):
  A = same workspace, AI_MLP_SPLIT=0, VLLM_GDN_TWO_SLOT=0, champion serve args (KV 5 GiB): model 14.37/14.37 GiB, pool 267,721 tok.
  B = defaults (split 0.5367, two-slot on) + serve_args.txt (KV 4.3 GiB): model 13.74 (r0)/14.90 (r1) GiB, pool 237,658 tok; shard lines on both ranks (8064 start 0 / 9344 start 8064).
  Observed tok/s | TTFT p50/p99 ms | TPOT p50/p99 ms | MTP acc | peak KV:
    A1 939.0 | 174/1151 | 15.2/33.5 | 2.342 | 83.8%
    B1 929.6 | 179/1167 | 15.6/40.6 | 2.334 | 68.3%
    B2 932.4 | 163/826  | 15.5/27.6 | 2.321 | 68.3%
    A2 930.9 | 176/957  | 15.4/29.9 | 2.332 | 83.8%
  A mean 935.0, B mean 931.0 (-0.4%; spreads 8.1 and 2.8): no measurable change. B: 0 preemptions (grep -ci preempt = 0 in all four logs), steady state Running 16/Waiting 0 except transient 15/1 (same pattern as A) and one 14/0 tail line; no admission stall.
  Interpretation: despite the 11-19% GPU0-vs-GPU1 Marlin gap in the microbench (T25-001), shifting 7% of MLP columns to GPU1 did not speed up end-to-end c=16 serving. Likely reasons (untested): at c=16 with MTP k=2 most steps are M<=48 (memory-bound, where G1 predicted ~1% gain) and the compute-bound prefill-chunk steps are a small share of time; the 4.3 GiB KV/two-slot confound did not hurt. A prefill-heavy scenario would be needed to see the benefit.
