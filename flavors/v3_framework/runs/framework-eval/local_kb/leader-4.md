# leader-4 (T4) local KB
- Started 2026-11-15. Grant 100 GPU-min. Champion T3-009 951.5 tok/s.
- ws t4-champ = T3-009. Worker leader-4.w1 profiling (jobs 139 graph, 140 node) -> agents/leader-4/prof_*.
- Phase 1 files: reports/T4-sweep/1-inputs.md, 2-graph.md done.
- Early candidate ideas: GDN state write-back via ReplaySSM depth L (checkpoint + logs); GEMM+AR column-chunk overlap in decode;
  draft count 3; tree spec w/ GDN; norm-distributivity to start next GEMM before AR.
- Phase 1 done, files 1-6 + summary written. E1 (capture sizes 576..1024 + steplog in ws t4-steplog) = job 145 by worker.
- Others: decode is at achievable floors (DRAM ~870 GB/s). k=3 dead, hostmem big msgs dead, CE AR dead, prefetch tuning null.
- T2 claims mixed-step idle only 2.5 ms; my graph trace says ~22 ms on small steps. Steplog settles.
- 02:33 Run PAUSED by human (via manager). E1 job 145 cancelled before running (no GPU used). Resume: resubmit E1 (w1_task2.md + steplog switch msg).
- Run ended by human. Wrote T4-writeup.md + T4-sweep/update-1.md; fixed e1_job.sh sed. Told manager; awaiting close.
