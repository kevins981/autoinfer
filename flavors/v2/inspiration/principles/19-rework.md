# Rework

**The principle.** Some work is done and then thrown away: an attempt fails, a guess is wrong, a result arrives too late or is made invalid, something is retried from the start. Its cost was paid and it contributed nothing. This is often accepted as rare, and rarely measured.

**The waste.** All work whose result is discarded.

**General directions.**
- **Measure it:** how much work is thrown away, and why?
- **Detect the failure earlier:** before more work is built on top of it.
- **Keep what is still valid:** discard only the part that was wrong.
- **Fail less often:** remove the cause of the discarded attempts.
- **Make a failed attempt cheaper:** so that being wrong costs little.

**Illustrations** (examples only; the best use of this principle may look like none of them). A carpenter measures twice and cuts once. A long calculation saves its progress so a failure doesn't restart it. A factory inspects at each station, not only at the end of the line.
