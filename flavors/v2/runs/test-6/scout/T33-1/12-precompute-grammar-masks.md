# Precompute most grammar checks and share parser stacks

**What it is.** Grammar-constrained decoding can interpret every vocabulary token against a parser stack each step. XGrammar prechecks tokens whose validity is independent of deeper stack context. It shares stack branches for remaining checks and overlaps mask construction with GPU inference.

**What it exploits.** Most token checks repeat across parser states, and speculative parser branches share structure. It helps structured generation; unconstrained decoding has no grammar overhead to remove.

**What it requires.** A grammar compiler, persistent-stack matcher, and engine integration; the paper's implementation contains 12,000 lines of core C++ with Python bindings.

**Exactness.** The comparison is to decoding with the SAME grammar, tokenizer, and masking semantics. Faster computation of the identical allowed-token mask preserves greedy output; adding a grammar to unconstrained inference would change output and is excluded.

**Reported results.** The source reports token-mask generation below 40 μs and over 100× JSON-grammar speedup against prior engines. Tests use Llama-3.1-8B-Instruct vocabulary, JSON-mode-eval, Ryzen 9 7950X and RTX 4090. These are grammar-engine results, not unconstrained generation speedups.

**Sources.** Yixin Dong et al., [XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models](https://arxiv.org/html/2411.15100v1), November 22, 2024.
