team:         T3
builds_on:    T2-007
performance:  +3.7% vs the champion (T2-007, 787.6): 816.6 tok/s, starts 811.3 / 821.8 (eval/1, GPU job 49); both starts above 787.6.
summary:
  hypothesis: HBM sits idle while each decode all-reduce (host-mapped hmar, ~46 us, latency-bound on PCIe) runs; a small side-stream
              kernel can pull the head of the NEXT GEMM's FP8 weight into L2 during that window, so the GEMM after the AR reads part of
              its weight from L2 (Place knob, L2 tier, in a link window; T3 map 4-knobs K5-a). Napkin ~1.1-1.5 ms of a ~36 ms step (~+3%).
  change:     by worker leader-3.w1. NEW vllm/distributed/device_communicators/l2_prefetch.py: persistent Triton read kernel (32 programs,
              8 warps, evict_last loads of the first VLLM_T3_L2_PREFETCH_MB=16 MiB of the weight; result folded into a private dummy),
              set_target/take/clear, one side stream per device. host_mapped_all_reduce.py all_reduce: if a target is pending,
              side.wait_stream(cur), launch prefetch on side, hmar on cur, cur.wait_stream(side). models/qwen3_5/nvidia/model.py
              Qwen3_5DecoderLayer.forward: set_target(in_proj qkvz/qkv weight) around input_layernorm.forward_fused (which absorbs the
              previous MLP's all-reduce), set_target(gate_up weight) around post_attention_layernorm.forward_fused; clear() after each.
              NCCL path never prefetches. Env VLLM_T3_L2_PREFETCH=0 disables. serve_args.txt = T2-007's. change.diff = ws diff t3-pf
              (contains T2-007's changes) = the measured workspace.
  results:    Observed: harness eval/1 (job 49): 816.6 tok/s (811.3, 821.8), acceptance 2.333, TTFT p99 1047/999 ms, TPOT p99 35.5/31.7 ms,
              gate_latency pass, 128K pass. Quick bench (job 48, 1 start each, same job): on 818.6 vs off 779.7 (+5.0%), TPOT p50
              17.96 vs 18.67 ms. Microbench (job 47, real 2-GPU hmar + cold Triton W8A8 GEMM in graphs, agents/leader-3.w1/hmar_pf.log):
              AR+GEMM gate_up 148.7 -> 138.8 us, qkvz 91.5 -> 82.1 us at 16 MiB; 32 MiB+ is a net loss. Earlier spin-surrogate microbench
              (job 45/46, agents/leader-3.w1/mb.log) consistent.
  conclusion: Interpretation: a real gain, consistent with hiding ~10 us of weight fetch per AR+GEMM pair under the latency-bound AR
              (~128 pairs/step -> ~1.2 ms, ~3.5%); the cache mechanism is not isolated by counters (review T3-2). Tuning per target
              (gate_up/qkvz/qkv, MTP) is open.
validity:     Exact relative to T2-007: the T3 code only reads weights (and writes a private dummy scalar never read by the model);
              arithmetic, communication precision, weights, KV unchanged. c=1 greedy 4 prompts x 200 tokens identical on/off (worker,
              job 48); acceptance unchanged. GSM8K not needed (no math change; T2-007's own GSM8K 93.6% and its scale-floor disclosure
              carry over). No memory change beyond a 4-byte dummy and one stream.
              Review reviews/T3-2 (codex): capture safety and no target leakage confirmed; open, non-blocking: (a) the dummy store is a
              conditional write/write race on a private scalar (no model effect); (b) no try/finally around set_target/clear (only
              matters after exception recovery); (c) device guard assumes current device == input device (true for 1 GPU per rank).
              Left as measured (fixing them would change the filed code: ruling on filed vs measured).
