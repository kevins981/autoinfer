team:         T30
builds_on:    T29-006
performance:  "unofficial: two-wave eager prototype 900.1 tok/s vs champion 972.9 in the same job (-7.5%); host-bound"
summary:
  hypothesis: Removing the per-layer join of the token-split chain (two waves A/B that each run all layers, B's core waits only for A's core of
              the same layer) lets each wave's all-reduces hide behind the other wave's compute, cutting big mixed steps (exposed AR ~31-43 ms).
  change:     Workspace t30-wave (from T29-006). T30_WAVE=1: for steps with >=384 tokens and a 64-aligned split (all spec/decode rows in wave A) the forward runs
              EAGER (no breakable graph) as two waves: per layer and wave: input norm -> qkvz/ba or qkv projection -> [B waits event of A's core] -> core with
              per-wave attention metadata (existing ubatch-style slicing via build_attn_metadata; mamba aligned state rows per wave; continuation wave has
              has_initial_state) -> out_proj -> DMA AR lane c -> post norm -> MLP -> DMA AR lane c into a shared out buffer; one join after the last layer.
              Wave A on the main stream + lane 0, wave B on the aux stream + lane 1. T30_WAVE=2 is the control (eager + the champion's joined chain).
              Files: vllm/t30_wave.py (new), v1/worker/gpu/model_runner.py (V2 runner), v1/worker/gpu/model_states/mamba_hybrid.py, models/qwen3_5/nvidia/model.py,
              v1/worker/ubatch_utils.py, v1/worker/t29_hostprobe.py (copied from t30-probe).
  results:    Observed (job 289, 120 prompts/arm, c=16, probe on; one run per arm):
              tok/s (acc len): W0 champion 972.9 (2.346); W2 eager+joined 958.4 (2.338); W1 two waves 900.1 (2.349).
              41 of 955 steps eligible; all 41 took the wave path in W1.
              Eligible steps by tokens, median GPU ms e0->e2 / host enqueue until forward returned / period:
                384-600: W0 104.0/69.4/72.0   W2 121.5/142.3/145.0   W1 178.5/198.7/201.7
                600-900: W0 144.0/71.3/76.2   W2 140.4/133.8/136.7   W1 167.1/183.2/186.2
                900-1200 (n=4): W0 179.6/81.1/88.8   W2 181.2/131.8/134.5   W1 169.5/186.5/189.5
              W1 step time is flat in size (178/167/170 ms) and host enqueue >= forward GPU span: host-bound. Eager alone (W2) already raises host enqueue from ~70 to ~135 ms.
              Correctness: layer 0 stages vs full-row reference: qkvz GEMM bitwise; ba rel 2.3e-3 (M-dependent GEMM); core 2.3e-3; attn AR out 1.3e-3; MLP 1e-2. Live generation test
              (one real wave step, n=1873, split 943) gave coherent outputs. No GSM8K run. NOT bitwise vs the chain (GEMM row-count effects).
  conclusion: Interpretation: the schedule runs and looks functionally right, but with eager cores the host (two eager GDN/attention cores per layer, ~+50 ms over eager-joined)
              cannot keep the GPU fed; the measurement says nothing yet about whether the schedule hides the AR on the GPU (see T30-005).
validity:     no change to model math beyond GEMM row-count (M) effects the champion's own split already has; eager forward only for eligible steps; default off.
