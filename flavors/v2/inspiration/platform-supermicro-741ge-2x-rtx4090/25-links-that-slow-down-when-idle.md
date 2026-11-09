# Links that slow down when idle

**What it is.** A PCIe link can drop to a lower speed or a low-power state when idle and return to full speed when traffic resumes. On this machine an idle GPU's link was seen reporting generation 1 instead of 4. Coming back to full speed takes time.

**What it makes possible.** The first transfer after a quiet period may not see the same link as a transfer in a steady stream.

**How this was checked.** The idle link speed was read from `nvidia-smi`. The time to return to full speed was not measured.

**Where to read more.** `nvidia-smi -q` (GPU link info); PCI Express power management summaries.
