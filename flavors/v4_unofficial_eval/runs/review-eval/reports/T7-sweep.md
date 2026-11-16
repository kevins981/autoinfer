# T7 sweep (leader-7): summary
Map: [1-inputs](T7-sweep/1-inputs.md), [2-graph](T7-sweep/2-graph.md), [3-calibration](T7-sweep/3-calibration.md),
[4-knobs](T7-sweep/4-knobs.md), [5-compare](T7-sweep/5-compare.md), [6-ranking](T7-sweep/6-ranking.md).
Compact map (team opened 76 min before the run's end). Main point: Speculate along the serial edge is the cheapest lever left
(per-step cost ~75% fixed). k=3 was judged "memory-blocked" by earlier maps; it is a capacity trade (GDN spec slots are counted
in the KV budget), and its step cost has dropped since its last test (T1-009, older code). Testing k=3 + 5.5 GiB KV on the champion.
