# T9 sweep, 3: calibration (from the champion's own measured profile, job 118 in bank/T5-001; no new GPU job: time budget)
Full-batch decode step (M=48 rows/GPU), measured vs napkin:
| part | measured ms/step | napkin | note |
|---|---|---|---|
| GEMMs gate_up+down+o/out+qkvz | 7.08+3.32+1.32+1.94 = 13.7 | 12.2 GB body /1 TB/s = 12.2 | ~89% of DRAM BW: near floor |
| all-reduce (hmar, 128 calls) | 5.96 | 128 x ~41 us latency-bound = 5.2 | latency, not bytes |
| GDN decode (48 calls) | 1.72 | state r+w ~1.2 GB fp32 -> ~1.2-1.6 | near floor |
| rest (attention 16 layers, target lm_head 1.27 GB, drafter 2 steps, norms, sampling, launch) | ~6.3 | lm_head 1.3 + drafter ~1.5-2 + attn ~0.5 + misc | |
| total | 29.0 | ~27 | napkin within ~7% |
Throughput check: 29.0 ms per step for 16 x 2.343 = 37.5 tokens -> ~1290 tok/s in pure decode; harness 895 = decode share ~65-70% plus prefill/mixed steps and tails: consistent with the T5 record's ~65% decode share.
Per added verify row (M 48 -> 64): GEMMs memory-bound at M<=64 (ridge ~80-160 rows) -> ~+3-6% GEMM time (tile effects); AR latency-bound -> ~+5%; GDN DL ~+1 log row; attention small. Napkin: +1.0-1.5 ms target + ~1.2 ms one more drafter step = +2.2-2.7 ms/step.
