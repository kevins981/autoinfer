# What this chip does not have

**What it is.** Several features of newer NVIDIA chips are absent on this one: thread-block clusters and scratch memory shared across SMs; the bulk copy unit (TMA) that loads whole tiles with one instruction; moving registers between warps while a kernel runs; add, subtract and multiply directly on bfloat16 values (fused multiply-add on bfloat16 does exist); the block-scaled and 4-bit float formats. Scratch memory per SM is 100 KB, less than on the older data-center chip of the same family (164 KB) and the newer one (228 KB).

**What it makes possible.** Code written for newer chips often assumes these features. Here the same job has to be done another way, or the code falls back to a slower path without saying so. Knowing what is missing tells you which published designs carry over unchanged and which need rethinking.

**How this was checked.** The compiler rejects cluster launch for this chip; the assembler rejects the bulk copy, register reallocation and bfloat16 add; the device reports no cluster launch and no tensor-map access.

**Where to read more.** CUDA Programming Guide: Compute Capabilities; PTX ISA reference (the target notes under each instruction).
