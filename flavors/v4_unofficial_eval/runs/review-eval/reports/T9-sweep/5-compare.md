# T9 sweep, 5: compare with history (compressed)
| my candidate | status in the run | on what code |
|---|---|---|
| A k=3 drafts | tested twice: T1-009 (on T1-007, 1 start, ~neutral), T7-001 (on T5-001 = champion, 1 start 888.9, -0.7%: acceptance 2.70 but ITL +14%) | champion: settled enough; not repeated. Missing in my napkin (T7's finding, adopted): GDN spec-state slots count against the KV budget (k=3 needs 5.5 GiB), and the measured per-row cost (+4 ms/step) exceeds the weight-bound napkin (+2.6 ms). |
| B tree drafts | dropped by T1-T3 maps (GDN branch states) | agree |
| C AR merging / AR+norm fusion | T4-002 (fused AR+norm: +0.2%, noise) | AR transport itself done (HMAR, T1-007) |
| D AR/GEMM overlap | T3-001 promoted (L2 prefetch); T5-002 (bigger prefetch, noise); T6-003/T6-004 mixed-step row split (+0.3%, failed 3rd start) | done |
| F reduced-vocab drafter | T1-002 on BASE: K=32768 bf16 replicated: step -8.5% but acceptance -4.6%, lost to the fp8 head. Never tried on the fp8 head with a larger K and kept sharded | **untried in this form**: my pick |
Differences: earlier maps priced reduced vocab as replicated (no gather) with K=32K; at K=131K replication would read more bytes than today's sharded fp8 head, so it must stay sharded and be re-sharded evenly (a prefix of each rank's own shard would leave rank 0 unchanged). T1-002's acceptance loss came from ids >= 32K (Latin long tail + all CJK); K=2^17 covers ids up to the end of the CJK block, only the newer multilingual merges (>=131K) are dropped. Special tokens (<|im_end|>, think tags) are added explicitly.
Disagreement: none with verdicts; T5's DeltaLog follow-ups (mixed-step DL, external log) are larger builds than this run's remaining hour allows.
