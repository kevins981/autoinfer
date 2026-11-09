# Main memory and how fast it can be read

**What it is.** Each GPU has 24 GB of GDDR6X memory on a 384-bit bus clocked at 10,501 MHz. At 21 Gbit/s per pin that is a theoretical peak of 1,008 GB/s. There is no error correction. The memory is not shared between the two GPUs.

**What it makes possible.** The peak rate puts a floor under any step that must read a given number of bytes: bytes divided by rate. Comparing that floor with a step's measured time shows whether the step is limited by reading memory or by something else.

**How this was checked.** Queried from the device on this machine (CUDA driver API device attributes). The peak rate is arithmetic from the bus width and data rate, not a measurement.

**Where to read more.** Ada Tuning Guide; the Nsight Compute memory metrics.
