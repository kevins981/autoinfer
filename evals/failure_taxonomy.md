# Failure taxonomy: why agents miss, kill or under-realize good changes

For judges. When grading a run against the answer key (`evals/EVAL.md`), every missed win, wrong
cause or partial gain has a reason. Use these types to name it. A miss can have more than one type.
If none fits, describe it plainly in the run's evaluation; add a type here once a pattern repeats.

The types are about **judgment and process**, not luck. A change that was never reached because the
budget ran out is not a failure of judgment, but say so explicitly ("ran out of budget"), with the
evidence (where the run was heading when it stopped).

## 1. Measurement methodology: the experiment couldn't answer the question
The agent ran an experiment whose number can't support the conclusion drawn from it. The number may
be noisy, measured on the wrong thing, or from a setup that doesn't behave like the real engine.

Common forms:
- **Noisy or unstable measurement.** Repeating the same measurement gives very different numbers,
  and the agent picks one or averages them without finding out why.
  - Example: a custom all-reduce kernel is timed at 52 µs in one job and 92 µs in another, same
    code, same size. The timing loop recorded each sample on one GPU only, without first lining the
    two GPUs up, so it also measured one GPU waiting for the other. The agent noted "skew,
    undiagnosed" and dropped the change based on these numbers.
- **A proxy instead of the real thing.** A microbenchmark or a standalone test decides a change
  whose real effect only shows inside the inference server. Standalone tests miss what the engine
  does around the operation: other kernels, CUDA graphs, timing differences between the GPUs.
  - Example: a change is dropped because a standalone benchmark shows "at best a tie", when one
    in-server measurement (~5 GPU-min) would have shown +5%.
- **Wrong setup.** The measurement didn't run the code it was meant to measure: an unbuilt
  workspace, a flag that silently didn't apply, a code path not taken.
  - Example: a prefetch that "gave no gain" never ran, because a later change moved the operation it
    was attached to.
- **Too little data for the size of the effect.** One start per arm for a 1% effect when starts
  differ by ~1-2%.

What a judge checks: Would repeating the measurement give the same answer? Does it measure what runs
in the engine? Is the sample big enough for the effect claimed?

## 2. Interpretation: good numbers, wrong conclusion
The measurement is fine, but the agent compares it with the wrong thing, prices the wrong cost, or
treats noise as signal.

Common forms:
- **Wrong reference point.** The number is compared with the wrong baseline, or with a "floor" or
  "best possible" that was assumed rather than derived.
  - Example: a new all-reduce takes 50 µs. The agent compares it with NCCL forced to a faster
    protocol in a standalone test (57 µs) and calls it a tie. But the engine runs NCCL's default,
    which takes ~72 µs there, so the real saving is ~20 µs per call (~7%). The agent also calls
    57 µs "the floor", though its own bandwidth figure gives a floor of ~40 µs.
- **Wrong cost.** The agent prices what is visible (a kernel's run time) instead of what the
  component actually costs the step (e.g. the GPU sitting idle while the CPU prepares it).
  - Example: attention is ranked "1% of a step, nothing to gain" from kernel times. Its real cost
    is the attention backend's per-step CPU planning, during which the GPU idles ~1-1.4 ms per step.
    Switching backends gives +4-7%.
- **Noise read as signal.** A gain or loss smaller than the measurement's own spread is treated as
  real.
  - Example: a setting is promoted at +1.0% when repeated measurements of the same code vary by
    ~1.6 points. It turns out to have no effect at all.
- **Irrelevant comparison.** A loss is counted that doesn't matter for how the change would be used.
  - Example: a small-message all-reduce is marked "loses for large messages", though the design
    only sends small messages to it.
- **Wrong mechanism.** The conclusion is right or wrong for a reason the evidence doesn't support.
  - Example: "the setting didn't apply, because the kernel names didn't change". The names don't
    reflect that setting; the library's debug log shows it did apply.

What a judge checks: Compared with what, and is that what the engine actually runs? Is the "floor"
derived or assumed? Is the effect bigger than the noise? Does the cost counted match where time
actually goes (including GPU idle)?

## 3. Path dependence: not revisiting an earlier choice
The diagnosis is right, but the fix stays local. The agent works around a constraint that an
earlier choice created, instead of asking whether that earlier choice should change.

- Example: a cheaper draft head needs ~0.6 GB of extra GPU memory. With it, only 15 of 16 requests
  fit, and throughput drops. The agent diagnoses this correctly, then tries smaller heads. It never
  revisits its own earlier change, CUDA graphs captured for steps up to 2048 tokens, which is what
  used up the memory. A smaller capture size (e.g. 512) would have kept most of that gain and made
  room for the head.
- Signs: "doesn't fit", "OOM", "not enough X" followed by shrinking or dropping the new change,
  with no look at what else holds X.

What a judge checks: When something didn't fit or conflicted, did the agent ask which earlier
choice created the constraint, and whether trading it off would pay more?

## 4. Never considered: the direction never came up
The change never appears in the run's maps, rankings, ideas or records.

- Example: the attention backend is never listed as a knob, so switching it (+4-7%) is never tried.
- Not the same as type 2: if the agent looked at it and wrongly dismissed it, it's interpretation.
- Distinguish from budget: if the run stopped early and the direction was plausibly next, say so.

What a judge checks: search the run's reports, knob tables, ideas and records for the change by
what it does (not by name), and say what was searched.

## Process costs (record them, but they aren't judgment failures)
- **Duplicate work:** two teams or agents test the same change.
- **Repeated attempts at a known dead end** without a new reason to expect a different result.

## Good process (note it, for contrast)
- Profiling before spending more GPU time when a result is surprising.
- Correcting one's own earlier explanation when new evidence arrives, and saying so in the record.
- Running the cheap real test (one in-server measurement) before a proxy decides anything.

## What could catch these (open, not adopted)
- Types 1 and 2: a fresh-context reviewer, called before a conclusion or claim, who asks: "what is
  this compared against, and is it what the engine runs?", "is the measurement repeatable?", "is
  this the real cost?".
- Type 3: a different question: "which earlier choice is now the constraint?".
