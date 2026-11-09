# Work described ahead of time as a graph

**What it is.** A sequence of kernels and copies, with their dependencies, can be recorded once as a graph and then launched as a unit, any number of times. Parameters of individual nodes can be updated without rebuilding the graph. Graphs can contain conditional nodes (if, while, switch) whose condition is written by a kernel on the GPU, so a decision does not have to go back to the host. A kernel can itself launch a graph.

**What it makes possible.** The cost of issuing many small pieces of work is paid once, and some control flow can stay on the GPU.

**How this was checked.** From the documentation; conditional nodes and device-side launch were not tried on this machine.

**Where to read more.** CUDA Programming Guide: CUDA Graphs.
