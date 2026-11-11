# LOCAL_KB leader-49 (T49 research)
## Cards (first set)
- Excess guarantees / unused results -> lm_head argmax (T42 has it). Boundaries -> TP AR (exhausted, T45 min-cut).
  Representation -> made me ask whether the HARDWARE can hold weights in a smaller form (Ada generic compression), untested in the run.
- Machine cards (IPC, PCIe lanes, green contexts) -> all measured zero-sum here; PCIe H2D lane idle ~85% of decode = a small extra weight channel (<=~2-5%).
## Spar T49-1 (ended): beliefs table in sparring/. PP-with-prefill-in-bubbles simulated -> T49-001 (+2..+10% at best, closed-loop latency eats bubble).
## Probe 1 (job pending): compressible memory (CU_MEM_ALLOCATION_COMP_GENERIC) read BW on zeros/const/real FP8 weights/nibble planes/random/fp32/bf16. agents/leader-49/comp/
- Probe 1 done: job 418 -> T49-002: HW compression only zeros/const (8.7x/2.9x); our data -4..-5%. Closed.
- 2026-11-11: manager recorded KB123/KB124. I replied 'close' (no idea clears the bar; 19.6 GPU-min returned).
