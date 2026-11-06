# Search policy (v1 default)

How the manager decides what to work on next, given everything so far. The manager reads this at
every `brainstorm()`. It is an isolated variable: a run can override it, and the human can edit it
mid-run. The run keeps a copy of the text used in each round.

1. If a leader's work this round improved the champion, at least one of next round's directions
   comes from that leader's "what I'd do next" (keep pushing what works).
2. Fill the remaining slots by upside and novelty: most by upside, at least one slot for a
   high-novelty idea (protected exploration; don't reject it on napkin math alone).
3. If nothing improved this round, select purely by upside and novelty.
