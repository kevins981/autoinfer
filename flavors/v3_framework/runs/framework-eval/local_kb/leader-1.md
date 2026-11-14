# leader-1 (T1) notes
- Sweep files reports/T1-sweep/1-6 done; map proposal T1-map-1 applied.
- Split with T2: T1 = host-mem AR (+ AR/GEMM overlap W1), draft flags (fp8 draft head, local argmax), depth k=3/4, target-side local argmax. T2 = W8A8 (decode+prefill), prefill lockstep, L2 prefetch, GDN DeltaLog, Marlin eff.
- Infra: export VLLM_RPC_BASE_PATH=/scratch/kevinsong/learning/autoinfer/.cache/tmp in every server job (socket path limit).
- Jobs: 1,3,4,6 worker AR microbenches. Host-mem AR: 43 us vs NCCL 53-60 at n=42; neutral at n>=260. PCIe: store+load concurrently 15-17 GB/s/dir (T2 job 8).
- job 5 failed (socket path). flags1 A/B rerun pending.
- Workspaces: t1-base (BASE, my A/Bs), t1-ar (worker: host AR).
