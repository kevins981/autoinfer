team:         T22
builds_on:    T14-004 (measurement only, no change)
performance:  "measurement: decode-kernel L2-warm floors vs in-situ"
summary:
  hypothesis: the champion's decode step leaves HBM idle ~35% (kernel chain ~360 us/GDN layer vs ~230 us of HBM bytes; leader-22 CPU analysis of T14's in-situ nsys, agents/leader-22/onestep.py); kernels running cold could be warmed in those idle slots.
  change:     none (microbench, workspace t22-a = T14-004 built). Scripts + data: agents/leader-22.w1/E1/.
  results:    Observed (jobs 148-151, read flush, M=48, us cold/warm/in-situ): qkvz 55.3/41.6/45.2; gdn_out 29.3/21.0/21.3; attn_qkv 52.3/38.8/38.9; attn_o 29.7/21.0/21.3; down 61.6/45.0/56.7; gate_up microbench unrepresentative (in-situ 84.3 below it). Prefetching the first 16 MB recovers 50-85% of each cold->warm gap; more MB add ~0.2-0.4 us/MB. GDN replay fully warm 23.4 vs in-situ 37.5, but state prefetch did not reproduce warm (harness flags=0; unreliable).
  conclusion: Interpretation: in situ every decode GEMM except down (+11.7) and qkvz (+3.6) is already at its L2-warm floor; the cold->warm gap is a first-tiles latency effect, not bandwidth. Ceiling of the whole "warm everything" direction ~1.6 ms/decode step (~+3.5%), below the 5% bar.
validity:     measurement only.
