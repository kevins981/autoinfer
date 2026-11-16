team:         T2
builds_on:    T2-004
performance:  "+1.2% vs the champion (T2-004, 743.9): 752.6 tok/s (starts 754.5, 750.7), eval/1 (GPU job 32); both starts above 743.9. Measured on the pre-guard revision (workspace t2-fq, change_measured.diff); the guards listed under 'change' were added afterwards (workspace t2-fq.b, change.diff) and job 32 is not an exact measurement of t2-fq.b"
summary:
  hypothesis: In T2-004 every body linear still runs a separate activation-quant op (~3 us/GEMM in graph; 4 per layer,
              256 per target step) because Fp8LinearMethod never advertises its input quant key, so Qwen's existing
              producer machinery (QuantizedActivation) never fires for the Triton W8A8 path. Fusing the quant into the
              producers (Gemma add-RMSNorm -> in_proj/qkv and gate_up; SiLU-mul -> down_proj) removes 3 of 4 per layer
              (192 launches per target step + 6 for the MTP layer's two passes; o_proj/out_proj stay unfused).
              Napkin (plan.md): ~0.5 ms of a ~46 ms decode step, ~+1%. The record also rides the extra tuned Triton
              configs for the other 23 graph sizes (5 JSONs now 31 M keys each, +~0.5% napkin).
  change:     (1) kernels/linear/scaled_mm/BlockScaledMMLinearKernel.py: input_quant_key() returns kFp8Dynamic128Sym
              (opt-in class flag accepts_quantized_input, set only on TritonFp8BlockScaledMMKernel) and apply_weights
              accepts a QuantizedActivation (data/scale views, same layout QuantFP8 produces). (2) quantization/fp8.py:
              expose_input_quant_key(layer, self.fp8_linear). Effect: the existing SiluAndMul producer
              silu_and_mul_per_block_quant now feeds down_proj. (3) models/qwen3_5/nvidia/ops/fused_norm_quant.py: new
              Triton kernel gemma_add_rms_norm_dyn128 (Gemma (1+w) add-RMSNorm, residual updated in place from the fp32
              sum, y rounded to bf16, per-group-128 amax -> scale = max(amax,1e-10)/448, q = y*(1/scale) clamped, fp32
              row-major scales; plain fp32 division = the fast-math division of the CUDA quantizer). (4)
              models/qwen3_5/nvidia/model.py: _norm_quant_key accepts the dyn128 key; forward_fused branch (explicit TP
              all-reduce first when the norm absorbs the reduce, then the fused norm+quant). (5) 5 config JSONs extended
              to 31 graph sizes (agents/leader-2.w1/mb/tuned_configs2, same cold-L2 search, job t2:tune2).
              serve_args.txt = T2-004's (no new flags).
              change.diff = workspace t2-fq.b = the measured code (t2-fq, change_measured.diff) PLUS guards added after
              job 32 from reviews/T2-5: (a) BlockScaledMMLinearKernel: explicit scale column count instead of view(M, -1)
              (M=0), (b) fused_act_quant SiLU producer: skip the CUDA launch for 0 tokens, (c) model.py _dyn128_layout_ok:
              unfused fallback unless x is 2-D contiguous bf16/fp16 (and residual same shape, contiguous). Guards are
              Python-side checks; they don't change the arithmetic. Updated unit_B.py (final API, same-y parity asserted,
              M=0 and layout cases) passes on t2-fq.b (GPU job 35).
  results:    Observed (harness, 2 fresh starts x 400 prompts, GATES=long,gsm8k, GPU job 32): 754.5 / 750.7 tok/s, mean
              752.6 (+1.2% vs T2-004 743.9, whose starts were 741.1/746.8; both starts above 743.9; +18.4% vs BASE).
              TTFT p50/p99 180/1033 ms (T2-004 183/918), TPOT p50/p99 19.6/33.6 ms (T2-004 19.8/35.9; BASE 23.0/46.1);
              gate_latency pass; MTP acceptance 2.335 (T2-004 2.348). 128K request pass (48.9 s, server answers after).
              GSM8K-250: 92.8% (5 no-answer; T2-004 92.8%, BASE 94.4%, gate 89.6%).
              Server log: Selected TritonFp8BlockScaledMMKernel; 10 "Using configuration from ..." lines, no
              "Using default W8A8 Block FP8 kernel config" warning.
              Unit tests on GPU (agents/leader-2.w1/fq/unit_test.py, unit_B.py; jobs 29, 31): consumer GEMM with a
              QuantizedActivation is bit-identical to the bf16-input path (M=1,16,48,320); norm+quant is bit-identical
              to flashinfer gemma_fused_add_rmsnorm + QuantFP8 (0.0000% scale/code mismatch averaged over M=1..2048; the
              normed y itself differs from flashinfer's in 0.0004-0.001% of elements = reduction order); SiLU producer vs
              unfused SiluAndMul + QuantFP8: not bit-exact (codes differ 3.0-3.8%, scales within ~1%, because the producer
              keeps silu*up in fp32 and divides x/scale).
  conclusion: Interpretation: +1.2% (two starts, consistent) is above the ~+1% napkin for fusion and ~0.5% for configs: the
              measurement does not separate the two. TTFT p99/TPOT p99 are lower than T2-004's, consistent with the
              fused kernels also helping mixed steps. Acceptance 2.335 vs 2.348 is within the usual start-to-start spread.
validity:     (leader to review) Same W8A8 scheme as T2-004: dynamic per-token-group-128 e4m3 activations, fp32 scales,
              no UE8M0 rounding, 128x128 block-scaled checkpoint weights, fp32 K-block accumulation; weights unchanged.
              Norm producer: bit-identical quantization of the bf16-rounded normed activation (unit test). SiLU producer
              (existing csrc op): numerically different from the unfused chain in two documented ways, neither a change of
              scheme: (a) silu*up kept in fp32 rather than rounded to bf16 twice (more precise), (b) scale floor
              max(absmax/448, 1/(448*512)=4.36e-6) vs the unfused absmax floor 1e-10: groups with absmax < ~1.95e-3 get a
              coarser quantization (absolute step 4.4e-6, i.e. error <= 2.2e-6 per element before the GEMM accumulation).
              Disclosed as an implementation detail of the existing CUDA producer; its occurrence on real down_proj inputs
              was not measured and no claim is made about its size relative to bf16 GEMM rounding after accumulation. GSM8K-250 92.8% >= 89.6 gate. lm_head, attention/GDN kernels, other norms, KV,
              comms unchanged. The MTP drafter's decoder layer uses the same fused path (same decoder class).
review:       reviews/T2-5 (code review, 2 rounds): TP all-reduce/residual semantics, GDN in_proj_ba bf16 input,
              call sites (decode, mixed, MTP decoder) and W8A8 validity confirmed. Issues raised: M=0 handling,
              strided-input assert, stale unit_B test -> agreed by the worker and fixed in t2-fq.b (unit_B incl.
              edge cases PASS, GPU job 35). SiLU producer scale floor (4.36e-6 scale vs 1e-10 amax floor): accepted
              by the reviewer as an implementation detail of the same scheme, disclosed above. Filing t2-fq.b
              with job 32 (pre-guard revision t2-fq) accepted with explicit provenance; change_measured.diff kept.
              leader-2 judgment: adopted.
note (leader-2):  a third start of the filed change.diff (GPU job 37, eval/2) was cancelled after 1.4 min,
              before any measurement, when T1-007 (771.3) was promoted: T2-006 alone (752.6) cannot
              beat it. Its code continues in the stacked record (T1-007 + T2-006), which was measured on
              the filed code.
correction (leader-2, after reviews/T2-6): the scale-floor error bound above ("absolute step 4.4e-6, error <= 2.2e-6")
              is wrong because FP8 code spacing varies; e.g. at the floor scale s, x = 399*s -> code 384, error ~6.5e-5.
              No numeric bound is claimed; disclosure = scale = max(amax/448, 1/(448*512)) and fp32 silu*up. See T2-007.
