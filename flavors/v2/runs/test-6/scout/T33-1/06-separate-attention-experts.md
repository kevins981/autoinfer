# Serve attention and experts from separate device pools

**What it is.** Conventional MoE serving places attention and expert feed-forward work within one parallel deployment. MegaScale-Infer separates them into independently sized pools. Microbatches shuttle between pools in a ping-pong pipeline, allowing attention replicas to feed larger expert batches.

**What it exploits.** Expert routing fragments batches and underuses expert devices. Pooling helps when enough requests aggregate; weak interconnects or low load can erase gains.

**What it requires.** A substantially redesigned distributed runtime, RDMA/GPUDirect communication, deployment planning, and possibly heterogeneous GPUs.

**Exactness.** The existing model's router still chooses the same experts. All selected expert computations execute with unchanged weights and BF16 state; placement and batching change, allowing rounding differences.

**Reported results.** The source reports 1.90× per-GPU decoding throughput over TensorRT-LLM for its 317B Scaled-MoE on an Ampere cluster. Evaluation includes Mixtral-8×22B and DBRX, production requests with median input/output lengths 571/159, and a 150 ms per-output-token limit; heterogeneous tests use H20/L40S.

**Sources.** Ruidong Zhu et al., [MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism](https://arxiv.org/html/2504.02263v1), April 3, 2025, §§4–7.
