# Ways to observe

**What it is.** Tools that work here without administrator rights: a per-kernel profiler with hardware counters, down to source lines and instructions (Nsight Compute); a timeline tracer for host and GPU activity (Nsight Systems); the framework's own profiler; a GPU debugger; sampling of GPU state (utilization, clocks, power, link speed, memory) through `nvidia-smi` or its library; and the Linux `perf` tool for a user's own processes. On this machine GPU hardware counters may be usable on the first GPU only.

**What it makes possible.** Anything that can be observed can be checked instead of assumed, including what the hardware does underneath the software.

**How this was checked.** The tools are installed and have been used on this machine.

**Where to read more.** Nsight Compute and Nsight Systems documentation; `man perf`; NVML reference.
