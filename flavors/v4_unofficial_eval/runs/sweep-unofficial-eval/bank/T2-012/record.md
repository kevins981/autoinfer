team:         T2
builds_on:    T2-011
performance:  unofficial: PASS +0.85% vs T2-011 (736.9/740.4 vs 732.4)
summary:
  hypothesis: Marlin FP8 decode GEMMs at M<=48 (thread_m_blocks<=3): the dispatcher picks the first valid config {64,256,256} (large_batch list), which is 5% slower than {128,64,128} on the biggest GEMM (gate_up N=17408), and ties on the rest (task 6 sweep, agents/t2-w1/m6/out.log).
  change:     csrc/quantization/marlin/marlin.cu: new mid_batch_thread_configs = {128,64,128},{64,128,128},{64,256,256}, used for thread_m_blocks 2..3 (M 17..48); m_blocks>=4 (M>48, prefill) keeps large_batch list, m_blocks==1 keeps small list. Fixed rule, no env knobs, fp32 reduce unchanged (ON). The "N/tn*m*4<=sms -> {128,64,128}" heuristic is unchanged. C++ rebuild required. (ws diff also contains T2-011's change since the workspace is built on it.)
  results:    Observed: microbench (agents/t2-w1/m6/bench7.py, job 56, L2 flushed, CUDA graph x20, flush subtracted, 2 rounds each, head=T1-005 marlin == T2-011 marlin): us head -> new
    qkvz  M16/32/48: 48.5/48.6/49.6 -> 48.4/48.7/49.5
    attn_qkv: 42.8/44.3/45.8 -> 42.6/44.2/45.8
    out: 19.4/20.7/21.7 -> 19.2/20.6/21.6
    gate_up: 102.3/106.3/111.7 -> 102.2/103.5/106.1 (M=32 -2.6%, M=48 -5.0%)
    down: 51.6/53.2/55.3 -> 51.5/52.9/55.2
    Max |diff| of outputs new vs head 0.0156 (max |out| 9.3, bf16 ulp scale; head vs head repeat = 0). No regressions at any M (all new <= head within 0.1 us noise).
    Napkin: gate_up -5.6 us per layer at M=48 x 64 layers (+1 MTP layer) ~ 0.36 ms/step; at M=32 ~0.2 ms; other shapes ~0.1 ms/step total. If a decode step is ~15-20 ms, ~2% at best, probably 1-2%.
  conclusion: Interpretation: small, safe, exact-up-to-reordering win; only gate_up benefits (two-thirds of the gain at M=48). Marlin kernels at M<=48 are otherwise at ~0.75-0.84 of HBM BW; remaining gap needs a new kernel (not tried).
validity:     Only changes which Marlin thread tile config is chosen (same kernel, same math, same fp32 reduce); no benchmark-specific logic; accumulation order differs so outputs differ at fp reordering level (0.0156 max, bf16 ulp). Needs unofficial_eval --gsm8k for the usual accuracy gate.
leader review (leader-2): valid. Only changes which compiled tile config of the same Marlin kernel is picked for thread_m_blocks 2-3 (M 17-48), for every Marlin GEMM (generic, not keyed to benchmark). fp32 reduce unchanged; outputs differ by accumulation order only (max |diff| 0.0156 at |out| 9.3; head-vs-head 0). No memory change -> no --long. Exact up to FP reordering -> no --gsm8k (official confirmation runs it).

unofficial #1 (2026-11-15T07:34): PASS: +0.85% vs HEAD T2-011 (B 736.9 / 740.4 vs reference mean 732.4 over 2 starts, 0 run in this job; 200 prompts); p99 TTFT 1248 ms / TPOT 43.8 ms, MTP acceptance 2.340 (unofficial/1/unofficial.json)
