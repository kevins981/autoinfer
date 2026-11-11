# leader-2.w12 R7-A in-server hybrid token split (ws leader-2.dmaar2, from T1-023)
Gate microbench: agents/leader-2.w12/dmaar/ (NOTES.md there).
Change: dma_staged_ar.{cu,py} (copy-engine AR, own slots/flags), cuda_communicator (creates it; VLLM_DMA_SPLIT=0 off),
GDN/attention defer_out_proj, model.py _qwen3_5_split_chain (M >= VLLM_DMA_SPLIT_MIN_TOKENS=256; c0 = 64*ceil(M/128) DMA on
current stream, c1 hostar on aux stream; layer returns reduced output, next norm skips AR). Test-only VLLM_DMA_SPLIT_CHECK=1.
job 93 (smoke1/): eager check: DMA vs hostar AR 0 neq in all 256 checks; split vs unsplit bitwise at 1024/1024, 1024/1017;
  max diff 0.5 out / 0.0625 resid at 896/868 and 192/161 (3.5-5.6% elems). Graph server: 100p 792.8 tok/s, gsm8k50 88%, 128K pass.
job 94 (abba1/): 200p c=16: A1 817.5, B1 833.5, B2 817.7, A2 821.9 tok/s; TTFT p50 A 197.0/197.9, B 193.1/191.0;
  TPOT p50 A 16.95/17.21, B 16.85/17.00; acceptance A 2.335/2.345, B 2.325/2.329. B-A mean +0.7% (B per-start spread 16 tok/s).
job 95 (abba2/, order B A A B): B1 822.7, A1 816.5, A2 817.2, B2 822.1 tok/s; TTFT p50 B 189.7/198.0, A 185.5/198.1;
  TPOT p50 B 17.03/17.19, A 17.22/17.07; acc B 2.332/2.335, A 2.334/2.349.
Pooled (jobs 94+95): A 818.3 (816.5-821.9), B 824.0 (817.7-833.5) = +0.7%; TTFT p50 A 194.6, B 193.0 (job 95 alone: B not lower).
