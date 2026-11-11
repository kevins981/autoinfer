# T45 write-up (research, leader-45)
Material: proposals/T45-1.md, bank/T45-001..005, local_kb/leader-45.md. Win: T45-005 champion 1173.4.
- Row-pricing rule (KB118): extra row ~0.07 ms below 64 rows, ~0.16 ms above; accepted token ~0.61 ms (KB111).
- gate_up band (KB120) and dispatch lesson (KB122): re-check GEMM dispatch whenever decode width changes.
- Memory clock P2 cap 10251 vs 10501 MHz (admin-only, KB119).
