# Retrieve draft continuations from a suffix automaton

**What it is.** Ordinary decoding runs the target once per generated token. SAM-Decoding matches the current suffix against an automaton of prior text and retrieves a possible continuation. A target-model pass verifies multiple proposed tokens together; the model-free variant can fall back to Token Recycling.

**What it exploits.** Repeated phrases and copying provide cheap proposals without another neural model. Unpredictable continuations reduce acceptance and can make drafting overhead unhelpful.

**What it requires.** Static/dynamic suffix automata, a tree-verification runtime, and speculative cache rollback. No target or drafter training is needed for the model-free variant.

**Exactness.** Under greedy verification, accept only the contiguous proposal prefix matching target argmax tokens, then use the target's correction. Retrieved text never bypasses verification; learned EAGLE variants are outside this card.

**Reported results.** Version 1 reports 2.27× speedup over autoregressive decoding for SAM-Decoding[T] on Spec-Bench: Vicuna-7B-v1.3, FP16, batch 1, greedy decoding, one RTX A6000-48GB. Token Recycling alone reports 1.84×.

**Sources.** Yuxuan Hu et al., [SAM Decoding: Speculative Decoding via Suffix Automaton](https://arxiv.org/html/2411.10666v1), November 16, 2024, §§3–4.
