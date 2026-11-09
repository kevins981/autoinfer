# Compiling while the program runs

**What it is.** GPU code does not have to be compiled before the program starts. A program can generate source, compile it in memory (NVRTC), link it with other compiled pieces while still optimizing across them (nvJitLink), and load the result, all while running. The Python kernel languages here work this way already.

**What it makes possible.** At run time the program knows things that were unknown beforehand: sizes, shapes, constants, which case actually occurs. Code compiled then can have them built in.

**How this was checked.** The libraries are installed (NVRTC 13.0, nvJitLink 13.4); not exercised here.

**Where to read more.** NVRTC and nvJitLink documentation; `cuda.core` documentation.
