# How to use the sparring partner (read once)

A sparring partner is a sharp, principled researcher with no knowledge of this run. **Its goal is
to help you**: to help you think critically about your problem and find what you might be
missing. It does that by being skeptical on purpose: it asks how you know things, breaks your
claims down and calls out steps that aren't justified. Teams before you often took earlier
conclusions as settled; several of those turned out to be wrong, and finding that out led to
wins.

- **It is for inspiration and critical thinking, not instructions.** You don't have to adopt
  anything it says or suggests. Use what helps.
- **Good moments:** when you are about to conclude that something is impossible, at a floor, or
  not worth doing; before you write a proposal or a forecast; before you recommend closing; when
  you're stuck. Any other time is fine too.
- **Explain your reasoning in plain words,** with the numbers. Don't send it to the knowledge
  base: it can't read it, and restating the evidence yourself is part of the exercise.
- **Take its questions seriously.** It is not an expert in this system, but it is smart, and most
  of its questions will be worth an honest answer. A question you can't answer with a measurement
  shows you something: the belief is assumed, not established. If you notice yourself answering
  "because an earlier result says so", look at what that result actually measured.
- **You decide what to pursue.** You don't have to convince it. When you are confident a question
  doesn't apply, say so with your reason in a line and move on. `spar end <id>` when you've got
  what you need (it gives you its list of beliefs and where each stands).
- **Afterwards:** note in your LOCAL_KB which beliefs turned out supported, which are assumed, and
  any cheap tests worth running. A test that overturns an earlier conclusion is a result: file it
  like any other. Conversations are saved in sparring/.

## Starting a conversation
Open with the problem as you see it, in your own words and with numbers. For example:
- **the goal and where the time goes** (the main parts of a step and what each costs);
- **the few biggest bottlenecks you think stand between the champion and much more speed, and
  why you believe each one is hard or at a limit**; or
- **the specific conclusion you're about to draw** (a direction is dead, a forecast is capped,
  nothing in a deck can work) and the reasoning behind it.

Then answer its questions. A conversation usually runs several turns; end it when you've got
what you need.

## Commands
- `spar start "<message>"` (or `spar start -f <file>`): a fresh partner; prints its first reply.
- `spar send <id> "<message>"` (or `-f <file>`): continue; prints the reply.
- `spar end <id>`: its final list of beliefs and where each stands.
- `spar list`: this run's conversations (transcripts in sparring/).
- Each call waits for the reply (seconds to a few minutes): give the Bash call a long timeout.
- `spar guide`: this text again.
