# Four one-way lanes

**What it is.** Each PCIe link carries data in both directions at once at full rate, and each GPU has its own link and two copy engines. So there are four independent one-way paths: host to GPU and GPU to host, for each of the two GPUs.

**What it makes possible.** An exchange between the GPUs does not have to be a sequence of turns. Both GPUs can send and receive at the same moment.

**How this was checked.** Follows from the link type and the device's reported copy engines; simultaneous transfers were not measured here.

**Where to read more.** PCI Express specification summaries; CUDA Programming Guide: Asynchronous Execution.
