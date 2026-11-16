# T8 sweep 4: knob by knob (gains vs the champion's measured 29.0 ms decode step; overall ~= 0.71 x decode gain unless noted)

## Knob 1 Keep/Recompute
| site | variant | napkin gain | verdict |
|---|---|---|---|
| GDN state S[r,l] (in-place keep) | DeltaLog (champion: 1 write + log) | done | built (T5) |
| S[r,l] | multi-step log: write S only every 2nd step, replay up to 2 steps of logs | write 1.2 GB -> 0.6 GB/step: -0.6 ms + spill effect ~0.4 ms -> ~-3.4% decode, ~+2.4% overall | candidate (large build, complex replay; not doable before 08:15) |
| S in bf16 | precision change | — | illegal (GDN math stays) |
| conv state | 3 rows, tiny | <0.1% | drop |
| KV cache (fp8 append) | already minimal; 16 layers only | ~0.2 ms | drop |
## Knob 2 Rewrite
| site | variant | gain | verdict |
|---|---|---|---|
| AR -> residual add -> RMSNorm -> quant | fuse into the AR epilogue (dedup passes over x) | norms 0.79 ms -> ~0.4 | ~+1% overall; medium build | candidate, low |
| small bf16 in_proj_ba (0.5 ms, latency-bound) | merge into qkvz GEMM (concatenate weights along N; same input) | -0.4 ms -> +1% overall; but ba is bf16 and qkvz FP8 W8A8: different dtype paths -> needs a mixed kernel | candidate, low |
| dense model, no MLA / LoRA / MoE | — | — | no Principle 1-3 sites |
## Knob 3 Speculate
| site | variant | napkin | verdict |
|---|---|---|---|
| next token, chain depth k | k=3 (M=64) | tokens/step 2.35 -> ~2.75 (pos2 ~0.567 x 0.70); step +~3.4 ms (draft step +1.34, GDN +0.6, AR payload +~1.0, GEMM/attn +0.5) -> 29.0 -> 32.4 (+12%); net decode +4.5%, overall ~+3% (mixed steps cost a bit more) | **candidate #1: flag only, DL kernel supports up to 3 log rows (kMaxLogRows=3)** |
| | k=1 | tokens 2.35 -> 1.78 (-24%), step -~3 ms (-10%) | loses |
| | tree (top-2 at pos0) | +~0.1-0.15 tokens; GDN state per branch (no shared S across branches), tree attn in Triton + GDN kernels | loses on build cost now |
| draft lm_head | reduced vocab (ruling 2, generic corpus freq) | draft head 1.75 ms -> ~0.3 ms: -1.45 ms (-5% decode, ~+3.5% overall) minus acceptance loss (tokens outside top-V') | candidate #3 (needs generic token-frequency data and a build; not before 08:15) |
| layer outputs, router | dense, no lookup index | — | Gate 2 fails |
## Knob 4 Split/Merge
| site | variant | napkin | verdict |
|---|---|---|---|
| d/m over TP=2 (champion) | — | — | baseline |
| l: PP=2 with 2 micro-batches | removes 128 ARs (-6 ms) but each GPU re-reads its 32 layers per micro-batch: +12 ms weights | loses (dominance: AR saved < weight re-read; breaks only if AR > weight time) |
| r: dual-batch overlap of AR with the other half-batch's GEMM | hides up to ~6 ms AR but weights read twice per layer (89 MB gate_up > 72 MB L2) -> +12 ms | loses |
| vocab (lm_head) | already split over TP, local argmax | — | done |
## Knob 5 Place
| weights/KV on CPU or peer | link 25 GB/s << 1 TB/s DRAM | loses |
## Knob 6 Window
| site | variant | napkin | verdict |
|---|---|---|---|
| AR || next-GEMM weight prefetch into L2 (champion: first 16 MiB) | larger prefetch (24-32 MiB) or more programs: AR ~46 us could stream up to ~40 MB at DRAM rate; measured prefetch sum 4.69 ms for ~2.1 GB (~0.45 TB/s) | each extra 8 MiB fully hidden saves ~8 us/GEMM x ~128 -> up to -1 ms/step (~+2.5% overall); risk: L2 (72 MB) eviction, prefetch overrunning the AR | **candidate #2: env-only (VLLM_T3_L2_PREFETCH_MB / _PROGS)**, cheap test |
| draft step || target lm_head | serial (draft needs the sampled token) | — | Gate fails |
| sample SendRecv 0.52 ms/step | replace NCCL send/recv with hmar path | -0.5 ms -> +1.2% overall | candidate #4, small build |
