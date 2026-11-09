# Every transfer has a fixed cost

**What it is.** Moving data across a link costs a fixed amount of time that does not depend on the size: a call into the driver, setting up the transfer, and learning that it has finished. Only above some size does the time grow with the number of bytes.

**What it makes possible.** For small transfers the count matters more than the bytes. The size at which the fixed part stops dominating can be measured, and it tells you which regime a given exchange is in.

**How this was checked.** A general property of this kind of link; the fixed cost on this machine is not stated here. Measure it.

**Where to read more.** CUDA Programming Guide: Asynchronous Execution.
