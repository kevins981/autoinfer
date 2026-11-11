# local_kb/leader-22 (T22, explore)

## Direction under study: "HBM-continuous decode" (software-managed L2 weight streaming across kernels)
Observed (CPU only, my agents/leader-22/onestep.py over T14's in-situ nsys B1, rank 0, 283 decode steps with the modal 1034 kernels):
per GDN layer (us): in_proj Marlin 45.2, splitK 3.6, elem 1.0, conv 4.0, gdn_replay 37.5, out_proj 21.3, hostar(AR2, prefetches gate_up) 52, norm 3.6,
gate_up 84.3, act 3.1, down 56.7, hostar(AR1, prefetches in_proj+out_proj) 39, norm 3.6 (+Kernel2 ba GEMM 4.4) = ~360 us.
attn layer: qkv 38.9, attention ~25 incl glue, o_proj 21.3, then same MLP.
Kernel sum = span (no gaps). Bytes/GDN layer: weights 191.6 MB + state ~25 MB r + lazy w => ~225-234 us at 0.96 TB/s => HBM busy ~62-65%.
Warm/cold refs: T1-034 Marlin M=48 cold(write-flush)/warm: gate_up 108/84, in_proj 51/43, down 54.7/45.3. T17: read-flush cold gate_up 92.5.
=> in situ gate_up (84.3) and in_proj (45.2) already ~warm floor; down (56.7) is cold (+11 vs warm); out_proj/o_proj 21.3 (warm unknown).
Interpretation: remaining L2-economics headroom = down (~0.7 ms/step), out_proj/o_proj (?), GDN state warm (?), AR kernel prefetch tail (AR2 52 vs AR1 39).
Manager's conditions (2026-11-09) for any proposal/E2: read-flush floors (KB76); in-kernel prefetch cost measured on the real Marlin (warm M=48 has no issue slack, KB39/49), not by subtraction;
net measured WITH the champion's AR-window prefetch on (~88 MB/step staged into 72 MB L2, new prefetch competes/evicts; KB25/41/78); show evict_first vs GDN-state spill (KB14) in situ.
## E1 (jobs 148-151, 1.0 GPU-min; agents/leader-22.w1/E1/REPORT.md)
Marlin M=48 cold(read)/warm/in-situ us: qkvz 55.3/41.6/45.2; gdn_out 29.3/21.0/21.3; attn_qkv 52.3/38.8/38.9; attn_o 29.7/21.0/21.3; down 61.6/45.0/56.7; gate_up microbench unrepresentative (in-situ 84.3 below it).
Cold->warm gap is mostly the first ~16 MB (latency/startup), not bandwidth. GDN replay warm 23.4 vs in-situ 37.5, but state prefetch did not reproduce warm (unreliable).
=> ceiling of the direction ~1.6 ms/decode step (~+3.5%) < bar. Not proposed. E2: in-situ ABBA of down 16 MB prefix prefetch at the AR before gate_up (worker).
