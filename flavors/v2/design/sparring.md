# v2 sparring partner (spec, draft)

(Named "questioner" in early drafts; renamed by the human's request to something less adversarial: it is on the leader's side.)

**Status:** agreed with the human (2026-11-10). Implemented (`tools/spar`, leader prompts, manager
may ask); not yet used in a run.

**Mechanics (as built):** synchronous, message-like CLI: `spar start "<msg>"`, `spar send <id> "<msg>"`,
`spar end <id>` (asks for the belief list), `spar list`. Each call blocks and prints the reply.
Backends: `claude -p` with `--resume` (default; file and shell tools disallowed, only WebSearch /
WebFetch, empty cwd under .cache/sparring_work: isolation enforced) or `codex exec resume` (more
model diversity; read-only sandbox, but Codex can still read files, so isolation is prompt-only).
config.md: `- sparring: claude leader / medium` (leader = the caller's model) or `codex gpt-6.1-sol / high`
(the human's choice for Codex).

**How the leader learns it:** the leader prompts have a short paragraph (what it is, when, "run
`spar guide` first"). The full guidance is agents/sparring_guide.md, printed by `spar guide` and
automatically on an agent's first `spar start`/`send` (which then sends nothing). It includes how to
open: the goal and where the time goes, the few biggest bottlenecks and why each seems hard, or the
conclusion about to be drawn. The "Leader-side guidance" text below is now in that file. Transcripts:
sparring/<team>-<n>.md (+ .json state).

## Why
- Leaders take earlier conclusions as given. In test-6, T32 and T33 both rejected most external
  cards by citing knowledge-base "limits" (GEMMs at the HBM floor, comm/compute overlap is
  zero-sum, the MTP head is near what it can learn) without re-examining them.
- Re-examining premises has paid off before, by accident: T31's direction came partly from
  finding that KB83's "L2 reads only 1.29 TB/s" was one team's kernel, not the cache; KB94
  corrected KB63's explanation.
- Cards (principles, hardware, external techniques) add ideas but don't make a leader question
  its own premises. That needs a conversation.

## What it is
- **A sparring-partner agent the leader can start at any time** (a Claude Code agent, same model as the
  leader). A multi-turn conversation by message, with no round limit; the leader ends it.
- **A fresh sparring partner per conversation**, so it never accumulates the run's history.
- **No history, enforced by settings:** an empty working folder; reads of the run's files and the
  code are denied. Everything it knows comes from the leader, who must produce the evidence (the
  number, the job output) rather than point to a KB entry.
- **Web only for general facts** (hardware specs, physical limits), not technique research.
- **Its prompt:** agents/sparring.md (persona: a principled researcher who wants the leader to
  win and so doesn't let hand-waving pass; every step of an argument must be justified, keep
  asking "why?" until evidence or an admitted assumption; being convinced is fine; load-bearing beliefs; evidence, necessity,
  counterfactual, decomposition, alternative, falsification; measurement vs conclusion; one to
  three questions per message; accept good answers; the leader decides; say when nothing
  load-bearing is left; keep a list of beliefs and their status).
- **Each conversation is saved** (sparring/<team>-<n>.md: the transcript plus the sparring partner's
  final list), so we can count later which premises were challenged and which then failed a test.

## Leader-side guidance (to add to the leader prompts)
Draft text:

> ### Talking with a sparring partner
> You can start a sparring partner whenever you like (`agent start ... --role sparring`; see the
> tools README): a sharp, principled researcher with no knowledge of this run. **Its goal is to
> help you**: to help you think critically about your problem and find what you might be
> missing. It does that by being skeptical on purpose: it asks how you know things, breaks your
> claims down and calls out steps that aren't justified. Teams before you often took earlier
> conclusions as settled; several of those turned out to be wrong, and finding that out led to
> wins.
>
> - **It is for inspiration and critical thinking, not instructions.** You don't have to adopt
>   anything it says or suggests. Use what helps.
> - **Good moments:** when you are about to conclude that something is impossible, at a floor,
>   or not worth doing; before you write a proposal or a forecast; before you recommend closing;
>   when you're stuck. Any other time is fine too.
> - **Explain your reasoning in plain words,** with the numbers. Don't send it to the knowledge
>   base: it can't read it, and restating the evidence yourself is part of the exercise.
> - **Take its questions seriously.** It is not an expert in this system, but it is smart, and
>   most of its questions will be worth an honest answer. A question you can't answer with a
>   measurement shows you something: the belief is assumed, not established. If you notice
>   yourself answering "because an earlier result says so", look at what that result actually
>   measured.
> - **You decide what to pursue.** You don't have to convince it. When you are confident a
>   question doesn't apply, say so with your reason in a line and move on. End the conversation
>   when you've got what you need.
> - **Afterwards:** note in your LOCAL_KB which beliefs turned out supported, which are assumed,
>   and any cheap tests worth running. A test that overturns an earlier conclusion is a result:
>   file it like any other.

## How we'll judge it (no judge model)
- How often leaders start a sparring partner unprompted, and at which moments.
- Per conversation: beliefs discussed; how many the leader marked assumed; how many led to a test;
  how many tests overturned an earlier conclusion.
- Leader behaviour: does it engage (answers with evidence) and still use its autonomy (skips with
  a reason, ends the conversation itself)? Read a sample of transcripts.

## Open
- Mechanics: the leader starts it with `agent start` (like a worker, not counted against the
  worker cap); its settings variant denies the run and code paths; the transcript is copied to
  questions/ when it stops.
- Encouragement level: start with the leader text above plus the manager asking at proposals and
  closures whether the leader talked to a sparring partner (not required). Require it only if usage
  stays near zero.
