team:         T3
builds_on:    T3-002
performance:  n/a (offline training experiment; no serving change)
summary:
  hypothesis: Restricting what moves (dedicated step-2 layer, LoRA, very low lr) plus a step-2 loss masked to step-1-correct positions turns the pilot's loss into a held-out gain.
  change:     agents/leader-3.w1/train2.py: A = frozen stock step 1 + trainable copy of the decoder layer for step 2 (fc/norms shared, frozen), lr 3e-6/1e-5; B = LoRA r32 on fc + layer linears, lr 1e-4/3e-4; C = full FT lr 2e-6; AL = A with LoRA. 3 epochs, 4096 tok/step, hard labels.
  results:    Observed (job t3-sweep3, 2.9 GPU-min): held-out a1/a2cum stock .7820/.5617; best A lr3e-6 .7820/.5630 (paired bootstrap over 40 seqs: a2 +0.13 pt, 95% CI [-0.11, +0.35]); C lr2e-6 .7817/.5622; B best .7814/.5608; AL .7820/.5618. After FP8 re-quantization all within +-0.1 pt of stock (A lr3e-6 a2 -0.04 pt). Held-out a2 resolves about +-0.25 pt.
  conclusion: Interpretation: restricted updates stop the degradation of T3-002 but give no gain: no learnable headroom with 130K hard-label tokens. Next: soft-label KL distillation from target logits computed from stored hidden states, plus a data-scaling slope (T3-004). Caveat: the A simulation uses stock step-1 KV for earlier positions, more favorable than a real 2-layer vLLM drafter.
validity:     No serving change; data provenance as T3-001.
