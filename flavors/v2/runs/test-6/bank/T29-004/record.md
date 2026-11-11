team:         T29
builds_on:    T29-003 (T27-002 + T29_W8A8=2; leader-29.w2's Marlin-layout W8A8 GEMM bs_gemm_marlin + quant_perm)
performance:  "OFFICIAL #1: 1045.8 tok/s (1043.9/1047.7, CV 0.3%), +8.1% vs champion T27-002 967.3; unofficial same-job ABBA +9.2% (T29-003, job 207)"
summary:
  hypothesis: Ship T29-003's tested mode as the default so official_eval (which can't set env vars) measures it.
  change:     T29-003 with T29_W8A8 defaulting to "2" in all its reads (envs.py compile_factors, marlin.py, kernels/linear/__init__.py, triton.py, t29_marlin_w8a8.py). Compile-cache key: default/"2" adds factor T29_W8A8="2" (same key as job 207's mode-2 servers); "0" adds no factor (champion hash), "0" = the exact champion path = the bf16-activation reference. Kernel/config code frozen as T29-003 (split-1 table, bs_gemm.cu sha1 c0e17e8682). Two non-math changes vs T29-003: (1) the bs_gemm extension builds from a copy of the source at a content-hash path inside $TORCH_EXTENSIONS_DIR, so any workspace copy (the official eval's) reuses the prebuilt .so instead of rebuilding (~86 s) in the GPU job (build.ninja holds no workspace path; reload checked: no rebuild); (2) the unwired t29_act_quant.py removed from the diff (kept in agents/leader-29.w1/fq/).
  results:    Observed:
    GSM8K-1319 same-job A/B (job 214; harness gsm8k.py --n 1319, conc 16, baseline client tree as eval.sh; champion serve_args):
      B = default (T29_W8A8 unset -> mode 2, FP8 activations at M > 96): 96.82% (+-0.5), 12 without 'Answer:', 343 s
      A = same code, T29_W8A8=0 (champion path, bf16 activations):       96.21% (+-0.5), 17 without 'Answer:', 363 s
      difference B - A = +0.61 points -> within 2.0 (PASS). Identical outputs on 471/1319 questions (temperature 0).
    128K: long_request 128000 on the default (B) server in job 212: pass (status 200, 19 tokens, 49.6 s; health after: 200).
      Job 212 is INCOMPLETE (its client shell hit a 30-min limit and killed it during side A; no reference there): its B side gave
      GSM8K-1319 96.7% (12 no-Answer), consistent with job 214. notes/job212/.
    Startup/memory (job 214): B ready 75 s, all 5 FP8 shapes warmed at load; model load 14.45 GiB (B) vs 14.44 GiB (A) per rank; KV 267,721 tokens.
    Throughput: not remeasured (code path = T29-003's mode 2: +9.2% abba, job 207).
  conclusion: Interpretation: the FP8-activation accuracy gate passes with margin (+0.6, i.e. no measurable loss; the sign is noise, SE ~0.5 each). The default now runs the tested mode, so official_eval measures T29-003's path. Ready for official_eval.
validity:     (leader-29's own review of the diff vs task_definition.md, Activations ruling 2026-11-09):
- Scope: only the target's (and MTP layer's) FP8 block linears with M > 96 change; M <= 96 calls the champion's exact marlin_gemm (precheck bitwise at M=48/96, job 207); decode median ITL unchanged. lm_head (target), embeddings, bf16 in_proj_a/b, norms, residual stream, attention/GDN, KV cache, communication (host AR/DMA, codec), draft lm_head + T27 screen: untouched.
- Activation quant (t29_marlin_w8a8.py quant_perm): dynamic, per token per 128-group, computed at run time from the live tensor: amax = max(max|x|, 1e-10), s = div_rn(amax, 448), q = clamp(x * (1/s), +-448) -> e4m3 RNE (rounded on the e4m3 grid in fp32 first: PI_SPEC s7 fix for Triton's f16 double rounding). No static/calibrated scale, no per-tensor or larger group, no data-tuned floor beyond the 1e-10 epsilon (same as vLLM's per_token_group_quant_fp8; w2's job 203 bitwise check vs it). The PI column permutation only reorders k inside each 32-block (group boundaries and scales unchanged) and is matched by the Marlin weight layout (PI_SPEC; repack emulation bitwise vs gptq_marlin_repack, job 202; confirmed independently by T28's derivation).
- Weights: the checkpoint's FP8 bytes in the champion's Marlin layout (unchanged); scales = an exact fp32 copy of the checkpoint's bf16 weight_scale_inv [N/128,K/128]. +~10 MB/rank.
- GEMM (t29_bs_gemm/bs_gemm.cu): SPK = 128/BKS pipeline steps per 128-K block (line 57); mma.sync m16n8k32 e4m3 accumulates into a per-block register tile `tmp` only; at it % SPK == SPK-1 (end of each 128-K block) tmp is scaled by As[row,kb]*Bs[nblk,kb] and fmaf-added into the fp32 accumulator, tmp reset (lines 205-216). Scales loaded per K-block (line 158). static_assert BN divides 128 (one weight scale per tile per K-block). K splits (unused: split-1 table) fall on K-block boundaries (kb0/kb1). So the FP8 tensor-core accumulator never spans more than one 128-K block; across blocks fp32 -- as the ruling requires. Output bf16; relF 1.67e-3 vs fp32 reference of the same W8A8 math (bf16 rounding level).
- Accuracy: GSM8K-1319 same job vs the same code with T29_W8A8=0 (bf16 activations): see results (must be within 2.0 points). GSM8K-250 gate by official eval.
- Benchmark-neutral: dispatch only on M (> 96) and shapes; configs from microbench of random/real weights; nothing keyed to prompts, seed or concurrency.
- Note: job 207's A (942-944 tok/s) is below the champion's official 967.3 because abba.sh uses 300 prompts (not the official 400); compare within job only.
notes:        notes/job214/ (A/B output, both gsm8k logs, job.sh), notes/job212/ (INCOMPLETE job: B GSM8K + 128K), notes/job.sh (job 212's script).

official #1 (2026-11-09T23:27): 1045.8 tok/s at c=16 (mean of 2 trials: 1043.9 / 1047.7, CV 0.3%) (+64.5% vs BASE), GSM8K 94.4%, MTP acceptance 2.325, TTFT p50 152 ms, TPOT p50 13.9 ms; gates: GSM8K pass, latency pass, 128K request pass (official/1/results.json)
