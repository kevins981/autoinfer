# The communication library, beyond one operation

**What it is.** The library that moves data between GPUs (NCCL 2.30) offers more than the sum-across-GPUs operation: broadcast, reduce to one GPU, gather from all, reduce-then-scatter, and plain send and receive between two GPUs. Several operations can be grouped and issued together. Buffers can be registered ahead of time. It has a Python interface. Its newest feature, communication started from inside a kernel, needs direct GPU-to-GPU access and so does not apply on this machine.

**What it makes possible.** The exchange pattern between the GPUs is a choice from a larger set than the one in use, and the library is not the only way to move data.

**How this was checked.** The package and its Python interface are installed. Which transport it selects on this machine was not checked here.

**Where to read more.** NCCL user guide and release notes.
