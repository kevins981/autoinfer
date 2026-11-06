# ASSUMPTION LEDGER
Frozen: model/weights, FP8 weights + FP8 KV, sampling, benchmark/harness/gate, hardware, exactness of outputs, max context >=128K with KV capacity >= baseline.
Open: FFN kernels/layout/fusion, comm overlap, engine code, serving flags incl. max-num-seqs, spec-decode config (exact), TP/PP.
Focus: goal is a faster FFN layer; changes anywhere in the engine are fine if they serve that goal.
Assumptions: Marlin is ~80% of DRAM peak so FFN gains likely come from fewer bytes read, better amortization, fusion, or hiding the all-reduce; noise ~3%.
