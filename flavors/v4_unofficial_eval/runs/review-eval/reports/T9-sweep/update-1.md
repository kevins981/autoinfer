# T9 map update 1 (07:25 UTC)
Tested: #1 reduced-vocab fp8 draft head K=2^17, evenly re-sharded: T9-001, 908.0 tok/s (3 starts 912.7/902.5/908.8, jobs 126+127), +1.41% vs T5-001,
acceptance -0.3%, 128K pass. Matches the napkin (+1.9% gross minus acceptance). Champion candidate.
Not done on purpose: a K sweep (choosing K by benchmark tok/s would be fitting to the benchmark; K was fixed a priori).
New dimension adopted from T8 (bank/T8-002, its graph-trace profile of T5-001) and my count in its sqlite: mixed (prefill+decode)
steps are host-launch-bound. Per GDN layer per prefill step, ~26 eager torch elementwise/gather/index launches + ~10 Triton
launches (chunk_* delta-rule kernels, conv1d, materialize, spec update): ~1,700 launches per mixed step; GPU busy ~50 ms of a
~110 ms period at 60-130 prefill tokens (profiled; host time inflated). Candidate (now #1 for future work): capture the GDN
prefill/mixed path in graphs or fuse the per-layer torch state gathers/scatters (ssm_state[idx] read/write) into one kernel;
napkin +10-20% if the idle survives at harness speed. Not attempted: too large for the time left.
Ranking now: 1. mixed-step GDN launch overhead (T8 + T9); 2. mixed-step DeltaLog (T5 #1); 3. k=3 with freed spec slots (needs 2 + external log).
