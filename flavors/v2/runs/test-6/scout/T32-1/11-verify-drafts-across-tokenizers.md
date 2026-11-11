# Verify draft text across different tokenizers

**What it is.** Ordinary speculative decoding assumes draft and target models share token IDs. SLEM converts an off-the-shelf drafter's proposed text into target tokens and verifies several positions in one target pass. At the first disagreement it emits the target's token and discards the remaining proposals, then realigns the drafter's context.

**What it exploits.** A cheap drafter predicting several consecutive target tokens. Poor acceptance, conversion overhead, or already efficient large-batch target execution can erase gains.

**What it requires.** Two unchanged models, tokenizer alignment, and block verification support. This is a decoding-runtime change, without training a new draft head.

**Exactness.** Use SLEM's temperature-zero path with target-token comparison. Only matching target argmax tokens are accepted; the correction also comes from the target. Non-injective tokenizer conversions require the paper's explicit handling. Distribution preservation alone would be insufficient evidence of greedy equivalence.

**Reported results.** Table 1 reports 59.7 tokens/s versus 21.4 for autoregressive CodeLlama-13b-Instruct on HumanEval, one A6000, using tiny_starcoder_py as drafter; reported speedup 2.79×. This is request generation, not a loaded-server benchmark.

**Sources.** Nadav Timor et al., [*Accelerating LLM Inference with Lossless Speculative Decoding Algorithms for Heterogeneous Vocabularies*](https://arxiv.org/html/2502.05202v3), first published January 31, 2025; [ICML publication](https://proceedings.mlr.press/v267/timor25a.html), July 13–19, 2025, Algorithm 2 and Table 1.
