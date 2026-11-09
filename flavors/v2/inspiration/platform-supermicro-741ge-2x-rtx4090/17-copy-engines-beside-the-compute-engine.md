# Copy engines beside the compute engine

**What it is.** Each GPU reports two copy engines, separate from the hardware that runs kernels. A copy between host and GPU can proceed while a kernel runs, and a copy in each direction can proceed at the same time.

**What it makes possible.** Moving data on or off a GPU does not have to stop computation on it, and sending does not have to wait for receiving.

**How this was checked.** Queried from the device on this machine (CUDA driver API device attributes).

**Where to read more.** CUDA Programming Guide: Asynchronous Execution.
