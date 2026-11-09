# More than one kernel at a time

**What it is.** A GPU can have up to 128 kernels resident at once. Kernels issued on different streams of one process may run at the same time when the first leaves SMs, registers or scratch memory free; otherwise they run one after another. Streams can be given priorities. The two GPUs are fully independent of each other.

**What it makes possible.** A small kernel does not have to wait behind a large one, and a kernel that cannot use the whole chip need not leave the rest idle.

**How this was checked.** Queried from the device on this machine (CUDA driver API device attributes).

**Where to read more.** CUDA Programming Guide: Asynchronous Execution.
