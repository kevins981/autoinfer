# Other engines, power and clocks

**What it is.** Besides compute, each GPU has video encoders (two), a video decoder, a JPEG engine and an optical-flow engine; all are idle in a compute program. The compute clock is not fixed: it ranges up to 3,105 MHz and moves with load, temperature and the power limit. The limit is set to 450 W per GPU, and an ordinary user cannot change it.

**What it makes possible.** The same code can run at different speeds at different moments, for reasons outside the code. The engines for video are separate hardware that compute work does not slow down.

**How this was checked.** Read from `nvidia-smi`.

**Where to read more.** `nvidia-smi -q`; NVIDIA Video Codec SDK documentation.
