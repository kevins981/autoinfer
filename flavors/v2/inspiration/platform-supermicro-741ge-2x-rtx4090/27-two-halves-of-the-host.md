# Two halves of the host

**What it is.** The host is two halves (NUMA nodes), each a processor with its own memory. A core reaches the other half's memory more slowly than its own. Both GPUs attach to the first half (cores 0-11 and 24-35). A thread on the second half, or a buffer placed in the second half's memory, is one step further from both GPUs.

**What it makes possible.** Where a thread runs and where a buffer lives changes the cost of the same operation.

**How this was checked.** Read from `lscpu`, `nvidia-smi topo -m` and the devices' reported NUMA node.

**Where to read more.** `lscpu`; `/sys/devices/system/node/`.
