# No direct path between the two GPUs

**What it is.** On this machine one GPU cannot read or write the other's memory (peer-to-peer access is reported as not supported), and there is no dedicated GPU-to-GPU link. Data going from one GPU to the other travels up one link into host memory and down the other link. Libraries that depend on direct access between GPUs do not apply here.

**What it makes possible.** Every exchange between the GPUs has two hops and passes through memory the host processors can also see. The host is on the path whether it takes part or not.

**How this was checked.** `nvidia-smi topo -p2p r` reports peer-to-peer as not supported.

**Where to read more.** CUDA Programming Guide: Programming Systems with Multiple GPUs.
