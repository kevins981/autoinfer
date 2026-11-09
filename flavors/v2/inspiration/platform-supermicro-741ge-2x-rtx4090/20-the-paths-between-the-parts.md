# The paths between the parts

**What it is.** Each GPU connects to the host by its own PCIe link (generation 4, 16 lanes) on its own port. Both links attach to the first processor socket. A link carries about 32 GB/s in each direction at most. A GPU reads its own memory about 30 times faster than that.

**What it makes possible.** Anything that crosses a link is far more expensive per byte than anything that stays on a GPU. How many bytes cross, and how often, is set by how the work is divided.

**How this was checked.** Topology read from `nvidia-smi` and `lspci`. The link rate is the standard's figure, not a measurement.

**Where to read more.** PCI Express specification summaries; `nvidia-smi topo -m`.
