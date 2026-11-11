# Round 10 report from leader-2 (T1)
Best: bank/T1-030 = T1-026 + bitwise-lossless host-AR payload compression (decode + prefill sizes). Official 897.3 tok/s (895.1/899.4,
CV 0.3%), +1.9% vs T1-026 880.9 (same-job ABBA +2.0%, both B above both A), +41.1% vs BASE; GSM8K 93.6%, TTFT p99 1183, TPOT p99 32.1,
acceptance 2.328, all gates + 128K pass. Propose as champion.

Experiments:
- T1-030 (w18, R10-A): per 32 bf16 values a max-exponent header + 1 B sign/mantissa + 4-bit exponent offset (15 = exponent field 0);
  raw fallback per block (offset > 14 or Inf/NaN); coalesced planes + group headers with speculative exception slots (wire 0.781).
  Gate: 300 real + 30 adversarial payloads bitwise (0 mismatches); AR time 0.76-0.78 at prefill sizes, decode -7 us/AR. ABBA A 834.7/
  835.5 vs B 848.7/855.3; TPOT p50 -2.0%, TTFT p50 lower. Opt-out VLLM_HOSTAR_COMP=0.
- T1-031 (w18): same codec on the T1-025 DMA lane: bitwise, but chain only -1.4%/layer at M=300 (encode/decode kernels ~15-20 us;
  chunk 1 compute already hides the lane). Dropped.
- T1-032 (w19, R10-B): Marlin single launch for any M > 64 (masked last slice, per-active-m-block instantiation, cost-weighted
  k-stripes). Gate -3..-7% at odd-32 M (288: 59.1 vs gate 57; 352: 69.0 vs 68), 64-multiples +0.5-0.9%; exact (1-2 ulp, sanitizer
  clean). In-server +0.4% vs T1-030 (noise). Every-16 prefill grid to 2048: CUDA OOM in graph capture (weights 14.6 + KV 5 + graphs
  2.35 GiB already). Not evaluated alone.

Harness note (for whoever owns task/scripts/lib.sh): when a server dies during startup, start_server_bench's background bench client
is not killed and later runs against the next arm's server (job 132: three clients on one server, ~300 tok/s each). Scripts should
check server liveness before trusting a bench.json.

Proposed SHARED_KB:
- KB-a: Host-AR payloads compress losslessly to 0.781 wire bytes (block-exponent, raw fallback 0.3% of blocks); AR time 0.76-0.78 at
  prefill sizes, ~-7 us at decode with prefetch; +2.0% same-job, official 897.3 (T1-030). Compressing the DMA lane doesn't pay (T1-031).
- KB-b: Marlin single-launch for M > 64 recovers 40-75% of the remainder-launch waste (odd-32 M -3..-7%), +0.4% in-server (noise);
  every-16 prefill graph grid to 2048 OOMs in capture at KV 5 GiB. (T1-032)

Left open: stacking the exact-but-small wins (T1-028 flush cheapening +0.2%, T1-032 single launch +0.4%) and a partial every-16 grid
(e.g. 128-512 only, memory permitting) — each below single-ABBA resolution.
Next (one direction): a "small exact wins" stack on T1-030: T1-028 + T1-032 + every-16 grid over 128-512 (graph memory checked, 128K
test), measured with 2 same-job ABBAs (8 starts) since each part is < 0.5%; est. +0.5-1% together. Low risk, all exact/fp-reorder;
the alternative would be a new brainstorm for a bigger lever.
Notes: GPU 50.3/80 (w18 11.1, w19 29.1 incl. two script-bug losses, evals 10.1). Workers stopped. Workspaces kept: leader-2.arcomp
(T1-030), leader-2.arcomp2 (T1-031), leader-2.marlin1/marlin2b (T1-032).
