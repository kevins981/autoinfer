# Observations (flavor v1)

Behaviors worth noting across runs, with the decision taken. Not rules.

- **Cross-team merge mid-round** (test-4, round 1). The manager told leader-1 that T2-003 (flags) had passed official, was orthogonal to T1's all-reduce, and could optionally be checked on top. Leader-1 A/B'd the stack and filed T1-003 = T1-001 diff + T2-003 serve args, which became the champion (+8.7%).
  Decision: fine. Teams aren't isolated, and what isn't disallowed is allowed. No new rule.
- **Brainstorm skipped when continuing leaders fill every slot** (test-4, round 2). Both leaders beat BASE, so search policy rule 1 gave both slots to their own next steps, and the manager skipped the brainstorm.
  Decision: correct. Made explicit in search_policy.md rule 1, manager.md and DESIGN.md.
