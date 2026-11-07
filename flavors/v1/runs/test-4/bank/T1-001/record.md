team:         T1
performance:  "official #1: 672.5 tok/s (669.1/675.8), +5.7% vs BASE; all gates pass (unofficial: +4.1%/+5.2% same-job)"
summary:
  hypothesis: NCCL RING_LL (no P2P) pays ~35 us fixed + half link rate per all-reduce. A one-shot
              2-rank AR through pinned shared host memory (/dev/shm + cudaHostRegister Mapped|Portable)
              should cut decode-size ARs from ~43/68 us (T=16/48) to ~20-45 us.
  change:     new vllm/distributed/device_communicators/host_ar/{host_ar.cu,__init__.py} (JIT via
              torch cpp_extension, sm_89); cuda_communicator.py: HostAllReduce for 2-rank "tp" group,
              dispatched before all other backends for bf16 contiguous tensors <= 2 MiB, else
              unchanged path (PyNCCL). Kernel: fixed grid 64 writer + 64 reader blocks, 256 thr;
              writers stage local chunk to host slot (epoch&1), fence.sys + flag; readers poll the peer's
              flag (ld.acquire.sys), read the peer chunk (ld.volatile), out = bf16(float(a)+float(b)).
              Per-block epoch counters in device memory (graph-replayable). 5-min spin timeout -> trap.
              Env VLLM_USE_HOST_AR=0 disables. From BASE.
  results:    Observed:
              - microbench (worker jobs 5/8/10, CUDA graph, 128 ARs/graph, min): host-AR vs NCCL
                T=16 19.6-24.9 vs 43.3 us; T=48 45-47 vs 68.2; T=96 80-84 vs 99.9; T>=256 ties NCCL.
                Napkin (write/25GB/s + read/23.4GB/s + 5us) T=48: 46 us -> matches.
              - single-GPU concurrent write+read of 491 KB: 29-33 us (vs 18.7 write / 21.5 read alone):
                the PCIe link is not full-duplex here; floor ~34 us at T=48.
              - bitwise: 0 mismatches vs NCCL (42 cases, T 16..2048, scales 1e-20..3e4), ranks identical;
                1500 alternating-size eager calls + 40 replays of a 129-AR graph: 0 mismatches.
              - NCCL_PROTO=Simple at T=256..2048 eager: 202/381/752/1477 us vs default 216/382/751/1474: no gain.
              - in-server A/B (bench 100, c=16): pair1 (job 11) OFF 601.3 / ON 626.0 tok/s (+4.1%),
                TPOT med 24.04 -> 23.28 ms; pair2 (job 12, final code) OFF 600.9 / ON 632.3 (+5.2%),
                TPOT med 23.82 -> 23.36. MTP AL 2.355/2.350, 2.366/2.350. gsm8k50 44/50 (= BASE).
                long_request 128000 passed twice. Logs: agents/leader-1.w1/ab/.
  conclusion: Interpretation: ~20 us saved per decode AR (~2.5 ms/step) -> ~+4-5%, consistent with
              napkin. Remaining gap to link floor is the serial write-then-read + flag latency.
validity:     leader-1 review of the diff: OK. Same math: 2-rank bf16 sum computed as fp32 add rounded
              to bf16, commutative -> both ranks identical, measured bitwise equal to NCCL. Comms stay
              bf16, nothing lossy. No benchmark-keyed behaviour (dispatch on dtype/size only). GPU memory
              unchanged (counters only); 8 MiB pinned host memory; 128K request tested. Robustness note:
              if construction failed on one rank only, ranks would disagree on the backend (not seen).

official #1 (2026-11-07T05:20): 672.5 tok/s at c=16 (mean of 2 trials: 669.1 / 675.8, CV 0.7%) (+5.7% vs BASE), GSM8K 93.6%, MTP acceptance 2.344, TTFT p50 232 ms, TPOT p50 21.8 ms; gates: GSM8K pass, latency pass, 128K request pass (official/1/results.json)
