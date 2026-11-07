team:         T1
performance:  "unofficial: 666.8 tok/s (bench 400 c=16) vs T1-002 665.0 unofficial / 662.8 official: +0% within noise"
summary:
  hypothesis: logits all-gathers (~1.1 ms/decode step via NCCL) can go through the T1-002 host-staged path, saving ~0.4-0.65 ms.
  change:     builds on T1-002: host-staged all_gather (any dim/dtype, exact copy; input viewed [outer, inner], each rank
              publishes its shard to shm and places it, then places the peer's) + hook at the top of
              CudaCommunicator.all_gather. Fallback to NCCL: non-contiguous, shard <1 KB or >24 MB, sizes not 16 B multiples.
              Worker leader-1.w2, ws leader-1.hostar2.
  results:    Observed: unit_ag.py 60 cases (bf16/fp32, dims 0,1,2,-1, 2-4D, incl. [48,124160]) bitwise == NCCL; 1500 random
              eager stress 0 bad; mixed AR+AG graph x200 replays 0 bad. Microbench: 4 MB shard NCCL 335 vs 274 us; 11.9 MB
              shard 977 vs 737-773 us -> est. 0.36 ms/step (~1%). Server (out/srvA): 666.8 tok/s, TTFT p99 1236, TPOT p50 22.0 /
              p99 51.5, accept 2.35, gsm8k50 88%. nsys B (out/nsysB): NCCL AllGather count 0, hostar kernels 140/step.
  conclusion: Interpretation: works and exact, but a ~1% effect, not measurable against 3% noise. Kept as a building block (T1-005).
validity:     leader-1 review: copy only (exact), index math out[(o*2+rank)*inner + j] = concatenation along dim; same flag/epoch
              protocol as T1-002 (monotonic wait); only on the TP group of a 2-rank no-P2P box. OK. No official eval (noise).
