team:         T1
builds_on:    T1-026
performance:  "official #1: 897.3 tok/s (895.1/899.4, CV 0.3%; +1.9% vs T1-026 880.9; same-job +2.0%), +41.1% vs BASE, GSM8K 93.6%, TTFT p99 1183, TPOT p99 32.1, accept 2.328, all gates + 128K pass"
summary:
  hypothesis: AR payloads (bf16 partials) compress losslessly to ~0.77 of their bytes (block-exponent scheme, measured on real AR
              inputs); the host AR is PCIe-bound at decode and prefill sizes, so fewer bytes = shorter AR (R10-A).
  change:     host_staged_ar.cu/.py: mode bit 16 = compressed payload. Per 32 bf16 values: header = max exponent field; per value
              1 B sign+7 mantissa + 4-bit exponent offset (0..14; 15 = exponent field 0, so zeros/denormals exact); a block with any
              offset > 14 or max exponent 255 (Inf/NaN) is sent raw (header 0xFF). Layout: 3 coalesced planes + 64 B header per
              32-block group with 2 speculative exception slots (warp-ballot order) + sparse overflow; wire ratio 0.781. Decoded
              peer value bit-identical -> same fp32 add and bf16 rounding. Raw fallback if size%64 != 0 or layout doesn't fit the
              slot. Opt-out VLLM_HOSTAR_COMP=0; VLLM_HOSTAR_COMP_MIN_BYTES (0). Allgather, L2 prefetch blocks, DMA lane unchanged.
              Worker leader-2.w18, ws leader-2.arcomp (agents/leader-2.w18/comp/NOTES.md, gate.py, g2/, abba1/).
  results:    Observed: gate job 120 (2-proc, graphs, same job): 300 real payload pairs comp AR == raw AR == fp32 ref (0 mismatches);
              decode round-trip == input; 30 adversarial cases (NaN payloads, Inf, -0, denormals, offset 14/15, exp 254/255,
              odd sizes) 0 mismatches; raw blocks 0.31% (max 1.25%). us/AR raw -> comp: M=48 with in_proj+out_proj prefetch
              46.5 -> 39.5, M=150 147 -> 115, M=300 292 -> 222 (0.76). ABBA job 121 (200 prompts c=16, node-0 slots in all arms):
              A 834.7, B 848.7, B 855.3, A 835.5 (+2.0%); TPOT p50 16.76 -> 16.41; ITL p50 27.38 -> 27.0; TTFT p50 191 -> 184;
              acceptance same; "lossless payload compression on" logged in B only. Estimate +2-3.5%; measured +2.0%.
  conclusion: Interpretation: gain in both decode and mixed steps; codec costs ~2 us/AR over the pure byte saving at decode.
validity:     leader-2 review of the raw diff (enc_block/dec_block): exponent offsets encoded only for 0..14 with nonzero exponent,
              exponent-field-0 values keep their mantissa (code 15), any other case and max exponent 255 -> raw block; decode
              reconstructs the exact bits. Pure re-encoding = baseline-precision communication (round-3 ruling); bitwise shown on real
              and adversarial inputs. No math change. OK.

official #1 (2026-11-08T09:25): 897.3 tok/s at c=16 (mean of 2 trials: 895.1 / 899.4, CV 0.3%) (+41.1% vs BASE), GSM8K 93.6%, MTP acceptance 2.328, TTFT p50 166 ms, TPOT p50 16.2 ms; gates: GSM8K pass, latency pass, 128K request pass (official/1/results.json)
