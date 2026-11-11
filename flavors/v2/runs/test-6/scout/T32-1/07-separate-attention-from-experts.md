# Give attention and experts separate device pools

**What it is.** Placing each whole MoE layer on one device group ties attention capacity to expert capacity. MegaScale-Infer separates the two into independently sized pools. Microbatches shuttle between them, overlapping one microbatch's communication with another's computation and aggregating enough expert work for efficient execution.

**What it exploits.** Attention and sparsely activated expert networks have different batching and hardware needs. Small deployments or slow inter-pool links can make the extra per-layer communication dominant.

**What it requires.** A distributed serving runtime, deployment planner, and GPU-to-GPU M2N communication library; demonstrated on Ampere and heterogeneous H20/L40S clusters.

**Exactness.** The existing router still selects the same experts, and every selected expert executes. Disaggregation changes placement and timing, not model architecture or expert selection; reduction rounding may differ.

**Reported results.** The source reports 1.90× per-GPU decoding throughput over TensorRT-LLM for Scaled-MoE on its Ampere cluster. Evaluation also includes Mixtral-8x22B and DBRX, BF16, production prompts with median input/output lengths 571/159. This headline excludes prefill; end-to-end gains are smaller, and microbatch pipelining can add latency.

**Sources.** Ruidong Zhu et al., [*MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism*](https://arxiv.org/html/2504.02263v4), first published April 3, 2025; revision July 26, 2025, §§4–7.
