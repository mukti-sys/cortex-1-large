---
name: gsd
description: >-
  Get Shit Done (GSD): Spec-driven development framework designed to eliminate agent context rot.
  Enforces structured phase execution (Spec -> Plan -> Execute -> Verify) with human-in-the-loop
  checkpoints and concrete verification criteria before proceeding.
---

# Get Shit Done (GSD): Spec-Driven Execution Protocol

GSD is a rigorous execution discipline designed to prevent agent hallucination, rabbit holes, and context degradation during complex coding tasks.

## The Core Problem: Context Rot
When an agent works continuously in a long conversation, early constraints and architectural decisions get diluted by accumulated tool output. GSD solves this by breaking work into isolated, verifiable phases.

---

## The 4-Phase GSD Lifecycle

```
[ Phase 1: SPECIFICATION ]
       │
       ▼
[ Phase 2: ATOMIC PLANNING ]
       │
       ▼
[ Phase 3: EXECUTION ]
       │
       ▼
[ Phase 4: EMPIRICAL VERIFICATION ] ────► [ COMMIT / DONE ]
```

### Phase 1: Specification (`spec.md`)
- Define the exact acceptance criteria before touching any code.
- Declare non-goals: explicitly list what is **out of scope**.
- Identify critical edge cases and failure modes.

### Phase 2: Atomic Planning
- Break the task into discrete, sequentially independent sub-tasks.
- Each sub-task must be executable in a single tool turn.
- Define the exact verification command (test suite, type check, lint) for each step.

### Phase 3: Execution with Fresh Context
- Execute one atomic step at a time.
- Stop immediately if an unexpected error occurs; do not attempt multiple speculative fixes in a single turn.

### Phase 4: Empirical Verification
- Never declare a task complete based on assumption.
- Run the real test suite or reproduction script.
- Verify exit code `0` and inspect stdout/stderr.
- Commit working changes immediately to Git with a clean, conventional commit message (`feat:`, `fix:`, `refactor:`).
