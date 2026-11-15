# Champion
T3-009 (team T3: champion T2-011 + fused norm+quant run per row half, restoring the mixed-step row-split AR overlap that T2-009 had silently disabled): official #1: 951.5 tok/s at c=16 (trials 955.0 / 947.9), +2.11% over T2-011, +49.6% over BASE, GSM8K 93.2%, TTFT p99 ~805 ms, TPOT p99 ~29 ms, 128K pass (bank/T3-009/official/1/results.json). Same-job ABBA vs T2-011: +1.40%, all B > all A (job 123). How: bank/T3-009/.

## History
- 2026-11-15: T3-009 (951.5 tok/s, +2.11% over T2-011) replaces T2-011.
- 2026-11-15: T2-011 (931.8 tok/s, +1.39% over T3-006) replaces T3-006.
- 2026-11-15: T3-006 (919.0 tok/s, +1.33% over T2-009) replaces T2-009.
- 2026-11-14: T2-009 (906.9 tok/s, +1.06% over T3-003) replaces T3-003.
- 2026-11-14: T3-003 (897.4 tok/s, +1.97% over T2-008) replaces T2-008.
- 2026-11-14: T2-008 (880.1 tok/s, +4.4% over T1-014) replaces T1-014.
- 2026-11-14: T1-014 (842.7 tok/s, +2.8% over T2-006) replaces T2-006.
- 2026-11-14: T2-006 (819.9 tok/s, +4.3% over T1-009; credit T2 + T1) replaces T1-009.
- 2026-11-14: T1-009 (786.4 tok/s, +3.8% over T2-002) replaces T2-002.
- 2026-11-14: T2-002 (757.6 tok/s, +7.1% over T1-004; credit T1 + T2) replaces T1-004.
- 2026-11-14: T1-004 (707.6 tok/s, +11.3%) replaces BASE.
- start: BASE (the task baseline): official #1 (shipped with the task, task/baseline/official/): 635.9 tok/s at c=16, GSM8K 94.4%, MTP acceptance 2.345, TTFT p50/p99 235/1341 ms, TPOT p50/p99 23.0/46.1 ms (official/1/results.json)
