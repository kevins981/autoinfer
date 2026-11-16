# T9 sweep, 6: ranking (vs champion T5-001, 895.4 tok/s; decode step 29.0 ms, draft lm_head 876 us x2 per step, job 118)
1. **Reduced-vocab fp8 draft head, K=131072 (ids [0,131039) by BPE merge rank + 33 added tokens), evenly re-sharded (65536 rows/rank).**
   Napkin: head 876 -> ~470 us per draft step: -0.8 ms/step (-2.8% decode, ~+1.9% overall) minus acceptance loss (each -1% acceptance ~ -1% throughput).
   Wins if acceptance falls < ~1.5%. Riskiest assumption: tokens >= 131K are rare in drafts' accepted positions. Cheapest test: one harness start (acceptance + tok/s), STOP_BELOW=895.4.
2. Mixed-step DeltaLog (T5 writeup #1): +0.5-1%, kernel routing build; not in the time left.
3. k=3 with freed GDN spec slots (T5 #2 + T7): needs the external-log allocator change first; multi-hour.
