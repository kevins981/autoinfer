# How Python runs

**What it is.** The Python here is 3.12. One thread at a time executes Python code within an interpreter; the others wait for a lock. Native code called from Python can release the lock while it works, and most heavy library calls do. Separate processes each have their own lock, and so can separate interpreters within one process (in this version they can only be created from native code). Starting a thread is cheap and it shares all memory; a process is isolated and shares only what is arranged.

**What it makes possible.** Whether two pieces of host work truly overlap depends on where the lock is held, not on how many threads exist.

**How this was checked.** Version read from the environment. Whether a given library call releases the lock has to be checked case by case.

**Where to read more.** Python documentation: the global interpreter lock, `threading`, `multiprocessing`.
