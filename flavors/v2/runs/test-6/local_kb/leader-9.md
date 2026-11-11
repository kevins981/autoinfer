# leader-9 (T9, explore) notes

## First-principles budget of a ~96 s trial (T1-042)
- Prefill compute floor: 104K prompt rows x 0.183 ms/row/rank (133 TF GPU0) = ~19 s. Hard floor (FP8/INT8/fp16-acc tricks are lossy or no faster, T2/T4).
- Decode ~2050 steps x 25.5 ms: Marlin 13.5 (12.15 GB/rank at ~0.9 TB/s, near roof), AR 6.2 (eff ~27 us/AR after L2 prefetch), GDN 2.1, lm_heads 1.6, attn 0.9.
- Mixed steps ~40% of wall (~38 s) vs ideal ~25 s (T7). Exposed prefill AR ~8 s (T2-006 buckets 10.2/18.6/42.7 ms x 215/86/99 steps).
- GPU0 locked 2235 MHz: 15-17% slower compute-bound -> rank 1 idles in mixed steps (~2-3% wall).
- AR bytes ~1 MB/row/direction; 2 ARs/layer minimal for TP; PP loses decode (2x weight reads + unsplittable lm_head/drafter on one stage).

## Key correction I build on
- T8-001: CE copies coexist with HBM-bound kernels (x1.01-1.12), SM host traffic doesn't (x1.45). KB33: CE copies don't slow compute-bound Marlin.
- T4-001 and T7-001 benches did NOT NUMA-bind (checked bench.py / ovl.py): T7's ~14 GB/s D2H matches KB38's far-node both-writing figure. KB65 may be contaminated.

## Direction T9: mixed steps to their compute floor
1. CE two-chunk ping-pong (TBO) for all prefill-containing steps (premise: T9-001 microbench).
2. Rank-balanced compute in compute-bound steps (GPU1 takes extra MLP neurons via a duplicated slice; KV budget may shrink if 128K passes).
3. Small-step items (Marlin 2-slab M 105-180, padding, GDN glue) only if 1-2 land.
Decode AR / decode GEMM: T8's area, avoid.

## Manager notes (2026-11-09)
- Credit at R>=256 only what ping-pong hides beyond the champion's DMA-lane split (S2 arm = exact T1-042 path).
- Lever 2 (duplicated weights on GPU1) needs a manager ruling BEFORE building (memory, 128K, "memory picture" clause). KB8: uneven TP ≈ 0.

## T9-001 (job 69) result
- NUMA-local: copies 20-24 GB/s even with both GPUs; T7's 14 not reproduced (far-node not re-run).
- P vs champion split S2 (rank 0): core=90: 86/94/96% at R=256/384/768; core=0: 43/42/86%. R=128/192 fail (21-32%).
- Champion split hides only 13-22% of S-N. Real GDN at R~820 ≈177 us/layer (my core=90 overstates 3x at large R) -> interpolate.
- Proposal T9-1 sent (payoff +5.6%, central +3%, 60 GPU-min). M1 = T9-002 real-core bench (worker).
- 2026-11-09: T9-1 funded, +60 GPU-min. Plan per worker: M1 ~6 (seed), M2 20, M3 10, official ~20, reserve rest.
- Correction (after KB66/T8 close): true exposed AR in R>=256 split steps = 15-23% of span (tl3.py; my S2 microbench 19-22%), not T2's 30%. Payoff -> ~+4%, central +2%. Told manager. Add-on candidates: decode CE AR (T8 leftover, KB68 <=1-2%), lever 2 (ruling).
- Manager: M2 decides. If M2 >= +2.5% same-job -> official eval of it alone before add-ons; (a) decode CE AR approved as add-on after.
- T9-002 (real GDN+FA2 core): hidden vs S2 rank0: 256 39%, 384 58%, 768 51%, 1024 49%; R<256 negative. Split cost small at R>=384. Implies in-server ~ (237/340/459 us/layer saved) -> ~2.9 s/trial ≈ +3% from R>=256. Not bitwise vs S2 (Marlin row-count reorder), GDN handoff bitwise at 64-aligned.
- Next: T9-003 diagnostic (dependency vs HBM), then M2 in-server ping-pong (design note first).
- T9-003: not the dependency (nodep same/worse), uneven worse. Copies 8-14 GB/s in P; chunks share SMs (A core 343 vs 80 us) -> only CE is a real overlap resource. Idea M2b: compressed CE payload (0.78).
- M2 design approved (agents/leader-9.w1/m2_design.md): reuse split_attn_metadata/ubatch slices; split = f(num_tokens) for graph compat -> GDN boundary not seq-aligned -> ULP-level reorder accepted (record max-abs, GSM8K). Diff review before first GPU job.
- Graph fact: mixed/prefill steps replay a BREAKABLE graph (segments cut at eager GDN core/attention); forks must join inside segments -> true P needs 2 capture lists (option B, risky).
- T9-004 (1.5 min): in-segment S3c (both chunks CE, chain extended with next qkvz) saves 130/149/272 us/layer at R=384/768/1024 vs S2 (~45-60% of P's). Est trial gain ~1.2 s ≈ +1.3%. Chose C: build S3c, then decide B / B-lite (external event wait nodes).
- Manager: resync turns on M2; S3c ~+1.3% can be an official win if repeated same-job ABBA is consistent. Reserve ~20 GPU-min for official eval.
- T9-005 S3c diff reviewed/approved (2026-11-09). NL=2 => memory change -> 128K test in ABBA job. Expect residual bitwise, qkvz ULP-level.
- Job 93: S3c in-server garbage text (pp1). Suspects: lane-1 tmp halved (OOB), pre buffer lifetime in breakable graph -> persistent double buffers. Balance 58.2.
- Job 97: eager S3c math OK (residual/qkvz bitwise, ba ULP). Graph pp1 OOMs at capture (+125MB bufs). Decided: no KV cut (KB13 admission risk); alias buffers (~70MB); fallback prefill-graph cap 1536 for pp1 (T5-004 neutral). Balance 53.7.
- Job 99: S3c works with graphs at cap 1536 (persistent aliased bufs ~35MB/parity); bitwise residual; 128K pass; free mem 360 vs 490 MiB. ABBA started (cap 18 min). Balance 46.
- Job 100 ABBA: S3c +1.2% (all B>A), gsm8k .88, 128K pass. Validity reviewed, official_eval T9-005 launched. Balance ~34.5 before eval.
- OFFICIAL T9-005: 948.9 (947.9/949.8) +0.7% vs T1-042; GSM8K 92.8, acc 2.327, TTFT p99 1035, TPOT p99 35.2. Balance ~23.
- Manager: T9-005 valid; wants 2nd same-job ABBA at 400 prompts (+12 GPU-min granted); promote if all B>A. Then T9-006.
- ABBA#2 job 104 (400p): A 940.2/942.9, B 945.1/950.5 (+0.7%, all B>A). Reported for promotion. T9-006 started. Balance 23.6.
- T9-005 PROMOTED champion (948.9). New bar >= 996.
- T9-006 FAILED: DMA decode AR 23-43 us/AR slower than hostar+pf (prefetch worth 15 us). Decode CE dead. Balance 23.1.
- Resync #1: EXTEND +20 (balance 43.1). Plan: T9-007 codec microbench (T1-031 warns -1.4%/layer), then B-lite plan review. Rebalance only with concrete ruling request.
- T9-007 codec dead; B-lite not built; recommended voluntary close (balance 42.1). Draft write-up: reports/T9-writeup.md.
- Write-up done, manager notified for close.
