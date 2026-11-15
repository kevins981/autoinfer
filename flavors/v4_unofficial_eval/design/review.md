# Reviewer (replaces the sparring partner, 2026-11-15)

Agreed with the human after sweep-simple-eval. Like a colleague reviewing a pull request.
- **Tool:** `tools/review` (start / send / list), adapted from the old `spar`. Codex only (model/effort
  from config.md `- review:`), multi-turn via `codex exec resume`, run from the run folder in the
  read-only sandbox with `CUDA_VISIBLE_DEVICES=""`. Transcripts in `reviews/<team>-<n>.md`, including
  the commands the reviewer ran (for auditing).
- **What it sees:** fresh context; may read the run (records, workspaces, GPU job outputs, task,
  shared KB, reports) and the engine's baseline code, and run CPU commands. Never GPUs, never writes.
  Told to judge the experiment on its own evidence (earlier records are context, not proof).
- **Prompt:** agents/reviewer.md: the project in a few lines, where things are, what it may do. No
  output format, no failure taxonomy (its examples come from the answer key), no "senior" persona.
- **When:** the leader's call, and its responsibility. The leader prompt suggests reviewing each
  record before filing it, positive and negative results alike, and workers' claims that shape the
  next step. No programmatic enforcement, no review line in records, no manager gate on candidates.
- **Known gap:** codex's read-only sandbox can still read the whole disk; staying out of the
  project's other folders (e.g. evals/) is prompt-only. The logged commands let us check.
- Why: in sweep-simple-eval a negative conclusion (host-memory all-reduce "at best a tie") was drawn
  from a flawed microbenchmark and a wrong baseline and never re-checked; the sparring partner, which
  saw no files, was rarely used.
