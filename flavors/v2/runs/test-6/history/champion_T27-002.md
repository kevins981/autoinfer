# CHAMPION (run test-6)
record: T27-002 (team T27, leader-27) = T14-004 + a PCA-screened FP8 draft lm_head inside the MTP drafter (ruling R1, task_definition.md 2026-11-09): approximate logits from a rank-1024 PCA projection of the drafter hidden state -> top-64 candidates -> exact FP8 rescoring. Draft step lm_head 336 -> 115 us. Exact: target verification unchanged (full vocab, greedy-ids path); only draft proposals can differ (acceptance 2.328 vs 2.33). Artifact: PCA fitted only on T3's target-self-generated data (160 train / 40 held-out sequences; r/k chosen on held-out); the file must stay at bank/T27-002/pca_train_n160.pt. Switch: T27_SCREEN=0 restores T14-004. +48 MiB/rank; 128K pass.
serve_args.txt: bank/T27-002/serve_args.txt (identical to T14-004's)
official #1: 967.3 tok/s at c=16 (+1.1% vs T14-004 956.6, +53.2% vs BASE 631.4; trials 966.3/968.3, CV 0.15%); TTFT p99 1141 ms; TPOT p99 33.0 ms; GSM8K 93.6%; MTP acceptance 2.328; 128K request pass. bank/T27-002/official/1/results.json
Evidence vs T14-004: official both trials above both of T14-004's (955.8/957.5); same-job ABBAs job 183 (A B B A, +0.8%, one B below one A) and job 186 (B A A B, +0.7%, all B > all A); pooled 8 fresh starts +0.76%.
Promoted 2026-11-09 by the manager.
History: BASE 631.4 -> ... -> T1-042 942.2 (test-5) -> T9-005 948.9 (test-6, T9) -> T14-004 956.6 (test-6, T14 + T6) -> T27-002 967.3 (test-6, T27).
A performance reference, not a required starting point.
