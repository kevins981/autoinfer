# Compiling a whole sequence of operations as one

**What it is.** The framework can trace a model's sequence of operations and hand it to a compiler as one unit, which may fuse operations, drop intermediate results and generate code specialized to the shapes seen. Separately, a sequence of GPU operations can be captured once and replayed as a graph.

**What it makes possible.** Boundaries between operations are a choice of the code as written, not of the computation.

**How this was checked.** Part of the installed framework (torch 2.14); not exercised here.

**Where to read more.** PyTorch documentation: `torch.compile`, `torch.fx`, CUDA graphs.
