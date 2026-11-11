# AR payload lossless compressibility (bs-r7-systematic_engineer, round 7, job 82)
Dump: BASE + a test-only hook in CudaCommunicator.all_reduce (ws bs-r7-se.dump), --enforce-eager, 48 ShareGPT prompts c=16;
every 37th AR input after call 3000, 300 per rank (agents/bs-r7-systematic_engineer/ardump/out, analysis ana.py).
Same math as the champion's AR inputs (bf16 partial sums of row-parallel GEMMs).
Results (decode M=48 and prefill M>=300 tensors):
- bf16 exponent-field entropy ~2.6-2.7 bits -> order-0 bound ratio ~0.73.
- fixed-rate scheme: per block of 32 values, one max-exponent byte + 4-bit exponent offset per value (code 15 = exponent 0,
  so zeros/denormals stay exact) + sign + 7 mantissa = 12 bits/value; a block with any offset >14 is sent raw (+1 flag bit):
  ratio 0.768 (23% fewer bytes), raw blocks 0.3%. Block 16: 0.785; 64: 0.760 (0.7% raw); 3-bit offsets: worse (0.85).
- Same ratio for decode and prefill payloads; exact by construction (pure re-encoding).
