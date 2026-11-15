# leader-2 (T2) notes
- Map in reports/T2-sweep/. Scripts: agents/leader-2/step_dump.py, step_seq.py; base_steps.csv.
- BASE unofficial: 634.5 / 643.4 (bank/BASE/unofficial/1).
- Job 4: unofficial_eval --ws t2-cg (BASE code) --args agents/leader-2/args_cg2048.txt --long (label t2-cg2048).
- Worker leader-2.w1 in workspaces/t2-w1: AR microbench (agents/leader-2.w1/allreduce_results.md). Allowance 8 GPU-min.
- Split with T1 proposed: T1 = AR + distributed argmax; T2 = W1 graphs, draft head, W8A8 prefill, GDN commit.
