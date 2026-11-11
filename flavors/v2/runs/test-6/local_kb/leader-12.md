# leader-12 (T12 explore) notes
- Seed: 0 GPU used. T12-001 (CPU): cross-request n-gram drafting is dead (oracle at k=2 +0.7% E; free-row upper bound +2.7%).
- Costed below the bar: cheaper drafter + budgeted K3 (~+4.3%); duplex DMA AR (~+2.5%, KB55 bound); 32+16 microbatch (~0); GDN state prefetch (<=1%); replicated drafter (no gain).
- Anomaly worth checking: drafter ARs 110-183 us per draft step (T5-003) vs ~50 expected.
- Write-up: agents/leader-12/T12-writeup.md. Recommended close.
- T12-1 killed at G1 (T12-002): 32K draft vocab -8.3% acc, -7.3% tok/s; FP8 fc -1.3% acc for ~10us. Package below bar. Recommend close; 21.5 min left.
