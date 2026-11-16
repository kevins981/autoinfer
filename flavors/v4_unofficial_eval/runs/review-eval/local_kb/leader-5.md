# leader-5 (T5) notes
- 04:58 start. Run ends 08:15, no new grants after 07:45. Grant 45 min.
- Champion T3-007 853.6. ws t5-champ = T3-007. worker leader-5.w1 profiling (job 77).
- Finding: csrc/gdn/fused_gdn_decode_kernel.cu gdn_decode_post_conv_mtp_kernel reads 1 fp32 state (from slot accepted-1) and WRITES num_tokens (=3) states per (req, v-head) -> per step/rank: read 1.21 GB + write 3.63 GB. DeltaLog candidate: write 1 state (+tiny inputs), replay accepted rows next step. ~2.4 GB/step saved ~2.6 ms of ~31 ms.
- Cheap screens: VLLM_T3_L2_PREFETCH_MB (16 default), async scheduling, num_spec 3.
- 06:35 T5-001 DeltaLog promoted, 895.4 (+4.9%). Mechanism job 118: GDN 72.7->35.6 us/call, next kernels -0.8 ms/step (L2 write-back).
