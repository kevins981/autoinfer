# Single instructions that do a loop's work

**What it is.** The instruction set has single instructions for jobs that are usually written as loops or several steps: count the set bits in a word, find the highest set bit, reverse bits, extract or insert a bit field; any three-input boolean function; a four-way 8-bit dot product into a 32-bit sum; reduce a value across all 32 threads of a warp (sum, min, max, and, or, xor); find which threads of a warp hold the same value; packed pairs of 16-bit floats in one register, with a fused multiply-add that can clamp at zero; an approximate tanh.

**What it makes possible.** Work on bits, small integers and whole warps can cost one instruction where a straightforward version costs many, and can stay in registers without touching memory.

**How this was checked.** Each instruction was assembled for this chip (`ptxas -arch=sm_89`); not run.

**Where to read more.** PTX ISA reference: integer, logic and warp-level instructions; CUDA Programming Guide: Device-Callable APIs and Intrinsics.
