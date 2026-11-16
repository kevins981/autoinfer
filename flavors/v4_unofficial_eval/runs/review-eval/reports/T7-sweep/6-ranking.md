# T7 sweep, step 6: ranking (vs champion 895.4 tok/s)
1. **k=3 drafts + KV 5.5 GiB on the champion** (flags only). Napkin +2-4%: acceptance 2.34 -> ~2.70 (+15%, T1-009), step cost +~10%
   (extra fp8 draft step ~0.8 ms, verify 64 rows, AR bytes +33%). Wins if all 16 requests stay running and the step-cost increase < ~13%.
   Riskiest assumption: memory (0.7 GiB headroom) for the 128K request; and step cost (T1-009 was neutral on older code).
   Cheapest test: the harness itself (GATES=long, STOP_BELOW=895), job 123.
2. **k=3 at a KV budget between 4.4 and 5.5 GiB** if 5.5 OOMs on 128K: capacity-limited, fewer running requests; likely loses (T1-004).
3. **Free spec slots under DeltaLog** (only checkpoint + log needed in decode): structural; unblocks k=3 without extra memory; >1 h of work.
4. **DeltaLog for mixed-step spec rows** (T5 #1): +0.5-1%; code change in the GDN path; >1 h with review.
5. PP=2: large, out of time.
