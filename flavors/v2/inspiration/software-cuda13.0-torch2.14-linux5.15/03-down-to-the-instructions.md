# Down to the instructions

**What it is.** The compiler's intermediate language (PTX) can be read, written by hand, and placed inline inside C++ code. The assembler reports registers used and spilled per kernel. The final machine instructions can be disassembled from any compiled kernel, including ones from libraries. The profiler can attribute time to single source lines and instructions.

**What it makes possible.** What a kernel really does is visible, and what the higher-level languages do not expose can still be written.

**How this was checked.** The tools are installed (`ptxas`, `nvdisasm`, `cuobjdump`) and the assembler was used to check the hardware cards.

**Where to read more.** PTX ISA reference; CUDA binary utilities documentation.
