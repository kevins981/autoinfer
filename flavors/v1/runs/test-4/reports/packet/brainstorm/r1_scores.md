# Round 1 scoring (manager). Novelty / upside, 1-5. Sources: systematic_engineer.md (SE), adjacent_field.md (AF).
| id | idea | proposed by | N | U | notes |
|---|---|---|---|---|---|
| R1-A | Host-staged one-shot all-reduce via pinned shared host memory (no P2P) | SE#2, AF#1 (both top bet) | 4 | 5 | AR 23% of wall (both measured); NCCL 65 us vs ~20 us wire at T=48 (AF measured); est +10-15% (assumed floor ~25-30 us). Custom kernel + memory ordering = risk. NCCL env knobs measured useless at decode sizes; Simple at prefill sizes unmeasured (SE job hung). |
| R1-B | Vocab-parallel head diet: FP8 draft lm_head + use_local_argmax_reduction (flags) + greedy target distributed argmax (code), then re-tune num_speculative_tokens | SE#1+#3, AF#2+#3 | 2 | 4 | drafter head + logits gathers ~10% of decode step (both measured); flags est +4%, target argmax +2-3%, depth 3 +4-10% (acceptance at pos 3 assumed). Low risk, explicitly allowed. |
| R1-C | Marlin tile configs / split-K / fused GEMMs for M<=64 | SE#4 | 3 | 2 | 64-84% of DRAM peak (ncu, measured), but AF measured L2-hot Marlin near its own floor -> ideas.md |
| R1-D | L2 prefetch of next GEMM's weights during AR | AF#4 (self-ruled out) | 4 | 1 | measured ~+3% max -> ideas.md |
| R1-E | NCCL env tuning | SE, AF (negative) | 1 | 1 | measured no gain at T<=96; prefill-size Simple unmeasured -> folded into R1-A's first check |
Selected: R1-A -> T1 (novel slot + top upside), R1-B -> T2 (high-confidence upside).
