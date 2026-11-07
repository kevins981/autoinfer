team:         T1
performance:  "official #1: 662.8 tok/s (+5.0% vs BASE), GSM8K 94.0%, gates pass (unofficial pre-fix: 665.0)"
summary:
  hypothesis: NCCL (RING_LL, SHM via host) reaches only 7.7 GB/s at decode size on this no-P2P box; a one-shot
              all-reduce through shared pinned host memory can approach the ~25 GB/s PCIe rate and cut the ~21% AR share.
  change:     new backend vllm/distributed/device_communicators/host_staged_ar.{py,cu} (JIT via torch cpp_extension,
              no csrc build) + hook in cuda_communicator.py. Auto-enabled for the TP group when world_size==2, CUDA, and
              no P2P (_can_p2p false); opt-out VLLM_DISABLE_HOST_STAGED_AR=1. Each rank writes its bf16 partial to its
              slot of a /dev/shm buffer (cudaHostRegister mapped, 2 parity slots), per-chunk epoch flags, reads the peer's
              partial, out = bf16_rn(fp32(a)+fp32(b)). Device-side epoch counter -> CUDA-graph capturable. NCCL fallback:
              non-bf16, non-contiguous, <1 KB, >24 MB, size % 16 != 0. From BASE. Worker leader-1.w2, ws leader-1.hostar.
  results:    Observed:
              - microbench (CUDA graph, 50 calls/graph, same job as NCCL), us: M=16 20 vs 39; M=48 48.6 vs 67; M=128 112 vs 138;
                M=450 331 vs 352; M=2048 1399 vs 1483 (bf16 x 5120). Go/no-go target (<=40 us at M=48) missed.
                Phase timing at M=48: write ~24 us + read ~25 us, mostly serial; M=1 ~10 us fixed.
              - correctness: bitwise == NCCL and rank0 == rank1 on 3000 random-size eager calls (M 1..3000, scales 1e-2..1e2),
                mixed-size CUDA graph x300 replays, race test (3000 back-to-back small ARs with random per-rank sleeps, eager
                + graph): 0 mismatches.
              - server (lib.sh bench 400 16, pre-fix build): 665.0 tok/s, TTFT p99 1382 ms, TPOT p50 21.8 / p99 39.9 ms,
                MTP acceptance 2.34; gsm8k 50: 88.0%. Log confirms backend enabled on both ranks. Ready in 52 s (incl. JIT).
              - napkin before: ~10-15 us x 135 ARs = 1.4-2 ms of 38.5 ms decode step (~4-5%) + small prefill gain; measured +5.3%.
  conclusion: Interpretation: transport-level gain is real but bounded by the serial write->flag->read latency chain;
              copy-engine variants were judged unlikely to help at M=48 (not built). Largest remaining AR lever would be
              overlapping the AR with compute, or fewer ARs.
validity:     leader-1 review of the raw diff: same math (fp32 add of exactly the two partials, round-to-nearest bf16; verified
              bitwise == NCCL), baseline precision comms, generic enable condition (topology, not benchmark), NCCL fallback.
              Review found a hang hazard in the first version (reader waited flag == epoch; a peer already at e+1 could
              overwrite it) -> fixed to a wrap-safe monotonic wait (flag >= epoch); the old check survives only behind a
              test-only mode bit never passed in production. Pure communication-kernel change: no memory-picture change
              beyond a 96 MB pinned host buffer + 64 B device state, so no 128K test required. Caveats: spinning kernel needs
              all its 8-64 blocks resident (same class of assumption as NCCL); both ranks must issue the same AR sequence
              (true for TP). OK.

official #1 (2026-11-07T00:06): 662.8 tok/s at c=16 (+5.0% vs BASE), GSM8K 94.0%, MTP acceptance 2.337, TTFT p50 238 ms, TPOT p50 22.2 ms; gates: GSM8K pass, latency pass (official/1/results.json)
