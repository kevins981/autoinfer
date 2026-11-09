# Fetching without waiting, inside a kernel

**What it is.** A kernel can start a copy from main memory into its scratch memory and continue working while the copy is in flight. The data does not pass through registers. Barrier objects tell the kernel when a batch of copies has landed, and several batches can be in flight at once (a pipeline).

**What it makes possible.** Fetching the next piece of data can overlap with computing on the current piece, inside a single kernel, instead of the two alternating.

**How this was checked.** Each instruction was assembled for this chip (`ptxas -arch=sm_89`); not run. The bulk form of this copy is not available on this chip.

**Where to read more.** CUDA Programming Guide: Asynchronous Data Copies; Pipelines; Asynchronous Barriers.
