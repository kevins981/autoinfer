# R1-B host-staged AR notes
## Estimates (before running). Floor = S/20GB/s per direction, pipelined write(chunk c)||read(chunk c-1); + ~8us sync/launch.
M=1 (10KB): ~10us (latency only). M=16 (164KB): ~16us. M=48 (491KB): ~25+8=33us (NCCL 61). M=128 (1.3MB): ~75us.
M=450 (4.6MB): ~240us (NCCL 347). M=2048 (21MB): ~1.1ms (NCCL 1480).
Risk: non-pipelined = 2x S/BW; SM loads from host mem have high latency -> need many in flight.
