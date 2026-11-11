team:         T28
builds_on:    T27-002 + T29-001 (T29_W8A8 all-M Triton W8A8 path, ported) + T25-003 (two-slot GDN state, ported, env VLLM_GDN_TWO_SLOT)
performance:  "unofficial same-job ABBA (job agents/leader-28.w2/job1/out4, 300 prompts): k=3 vs k=2, both W8A8 + two-slot: ALL B > ALL A, +2.93% (B 983.0/989.1 vs A 960.1/955.8). Decode step +2.65 ms (+10.2%) for E +14.4%; TPOT p50 -2.1%. Earlier unpaired pair: 991.1 vs 974.0 (+1.8%). Not yet compared with the champion in the same job."
summary:
  hypothesis: with W8A8 (FP8 MMA) decode GEMMs the M=64 cliff disappears, so k=3's +15% E costs only the 3rd draft step + AR + glue (~+1.4 ms) instead of +4.3 ms (T5-004); two-slot GDN state (T5-005/T25-003, bitwise) makes 16 requests at k=3 fit in the 5.0 GiB KV budget (KB13: stock 13 blocks/req x 16 = 5.8 GB).
  change:     workspace t28-k3 = champion + T29's W8A8 dispatch (envs.py, kernels/linear/__init__.py forcing TritonFp8BlockScaledMMKernel under T29_W8A8=1, fp8_utils.py assert, 5 RTX_4090 config JSONs) + T25-003's two-slot GDN pieces (csrc/gdn/fused_gdn_decode_kernel.cu, _custom_ops.py, v1/attention/backends/gdn_attn.py, mamba/gdn/replay.py, qwen_gdn_linear_attn.py, envs.py; default OFF here). Serve args: champion's + num_speculative_tokens 3; KV budget harness default (5.0 GiB).
  results:    Observed (worker leader-28.w2; A = job 194 start 1, B = job 201; 300 prompts c=16, one fresh start each; agents/leader-28.w2/job1/outB3/):
    | | A: W8A8 k=2 | B: W8A8 k=3 two-slot | B/A |
    | tok/s | 974.0 | 991.1 | +1.8% |
    | E (acceptance) | 2.335 | 2.671 | +14.4% |
    | TPOT p50/p99 ms | 14.74/28.06 | 14.47/32.05 | -1.8% |
    | ITL median ms (decode step) | 25.89 | 28.48 | +2.59 (+10.0%) |
    | TTFT p50/p99 ms | 170/832 | 174/734 | |
    | KV pool tokens | 267,721 | 276,547 | +3.3% |
    | peak KV usage | 83.8% | 58.6% | |
    Memory: model 14.27 GiB both ranks (same as A), graphs 2.64 GiB (A 2.63), nvidia-smi 23,812 MiB both GPUs after bench and after 128K. 0 OOM/preempt. 128K request PASS (51.7 s), health 200. gsm8k50 90.0%. W8A8 log line present; two-slot assert silent; gdn_replay_op_available True.
    Earlier attempts: k=3 at KV 5.75 GiB OOMs in prefill-graph capture; at 5.5 GiB runtime OOM in the GDN forward on the first request (stock GDN slots): memory is ~a few hundred MiB tighter than in T5-004's era (screen + W8A8 path).
  conclusion: Interpretation: the Marlin 3->4 m-block cliff is mostly gone (+2.6 vs +4.3 ms/step); ~1.2 ms above the forecast's +1.4 remains, consistent with Triton's own M=48->64 warm growth in the microbench (T28-002: Triton M=48 13.56 -> M=64 14.56 warm, +1.0 ms; cold +0.3): a byte-floor kernel at M=64 (Marlin-e4m3, in progress) is worth ~1 ms/step more. E +14.4% vs step +10% => decode ~+4%; the measured +1.8% is one unpaired pair (noise ~+-1.5%). Capacity: two-slot GDN raises the KV pool 3.3% in tokens and drops peak usage 84% -> 59%, so 16 requests admit with room (max context unchanged; 128K passes).
ABBA (worker leader-28.w2, order ABBA, 300 prompts, both arms T29_W8A8=1 VLLM_GDN_TWO_SLOT=1 KV 5.0 GiB; A k=2, B k=3):
    | start | tok/s | E | TPOT p50/p99 | ITL med | TTFT p50/p99 | peak KV |
    | 1 A | 960.1 | 2.328 | 14.97/23.6 | 25.89 | 162/930 | 58.6% |
    | 2 B | 983.0 | 2.663 | 14.70/32.8 | 28.59 | 167/904 | 58.6% |
    | 3 B | 989.1 | 2.665 | 14.54/32.8 | 28.52 | 174/942 | 58.6% |
    | 4 A | 955.8 | 2.326 | 14.91/29.7 | 25.93 | 157/890 | 58.6% |
    Verdict: all B > all A, +2.93%. KV pool 276,547 tokens in all starts; 0 OOM/preempt. Interpretation: k=3 on an FP8-MMA decode path is a consistent ~+3% over k=2 (noise ~+-1%); B's TPOT p99 32.8 is above A's but far below the gate (BASE 46.1 x 1.2).
ABBA vs the pristine champion (worker leader-28.w2, agents/leader-28.w2/job1/out5, order ABBA, 300 prompts; A = t28-champ (T27-002 as is), B = t28-k3 T29_W8A8=1 VLLM_GDN_TWO_SLOT=1 k=3):
    | start | tok/s | E | TPOT p50/p99 | ITL med | TTFT p50/p99 | peak KV | KV pool |
    | 1 A | 942.7 | 2.336 | 15.31/31.1 | 24.91 | 182/1205 | 83.8% | 267,721 |
    | 2 B | 981.9 | 2.665 | 14.54/32.4 | 28.59 | 169/869 | 58.6% | 276,547 |
    | 3 B | 981.7 | 2.662 | 14.31/35.0 | 28.59 | 168/784 | 58.6% | 276,547 |
    | 4 A | 942.9 | 2.339 | 15.18/33.1 | 24.98 | 165/1009 | 83.8% | 267,721 |
    Verdict: all B > all A, +4.13% (B 981.8 vs A 942.8). Interpretation: vs the MARLIN champion the decode step grows +3.64 ms (24.95 -> 28.59) for E +14.0%, i.e. decode tokens/ms is neutral (0.0932 vs 0.0936): on the Triton standard-layout path, k=3's M=64 in situ costs about what Marlin's cliff cost (T29-001's +1.05 ms/step k=2 penalty + the +2.6 ms growth). The +4.1% therefore comes from the mixed steps (W8A8 at M>96), which T29-003 captures better (+9.2%). For the combined stack the decode step must come from the e4m3 kernel (T28-004: M=64 warm 13.39 vs Marlin M=48 13.75) and from removing the quant launches. Superseded as a champion candidate by T29-003/004; kept as the measured k=3 datapoint.
validity:     (leader-28) Speculation exact: greedy verification unchanged, k is a config value. Two-slot GDN: bitwise (T5-005, T25-003). W8A8 path is T29's (FP8 activations per the ruling; a record for official eval will need the GSM8K-1319 same-job A/B). Memory change stated: KV pool 267,721 -> 276,547 tokens, 128K pass. Not an official candidate yet (single unpaired pair; W8A8 path will move to the Marlin layout).
