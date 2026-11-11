team:         T3
builds_on:    T3-001
performance:  n/a (offline training experiment; no serving change)
summary:
  hypothesis: Fine-tuning the MTP head on target self-generated data with a 2-step unrolled CE loss raises held-out acceptance.
  change:     Full fine-tune of fc, norms, decoder layer (agents/leader-3.w1/train_mtp.py), 360 train prompts (~130K response tokens), 4096 tok/step, 3 epochs, lr 1e-5 (GPU 0) and 5e-5 (GPU 1); unmasked step-2 loss.
  results:    Observed (GPU job 9, 1.3 GPU-min): held-out a1/a2cum stock .782/.562; lr1e-5 best (ep3) .778/.549; lr5e-5 best (ep3) .768/.534; held-out already lower after 1/3 epoch (.773/.534, .762/.509). Train loss collapses at 5e-5 (step-2 2.57 -> ~0.2-0.4). FP8 re-quantization of the trained weights: a2 +0.7-1.2 pts vs their bf16 (lr1e-5 FP8 .782/.561 ~= stock).
  conclusion: Interpretation: with 130K tokens, full-weight Adam updates hurt held-out from the first steps (perturbation faster than signal); 5e-5 overfits. Unmasked step-2 loss is dominated by positions that never matter. Not evidence against the direction yet; next: restrict trainable params (dedicated step-2 layer, LoRA) + masked step-2 loss (T3-003).
validity:     No serving change; data provenance as T3-001.
