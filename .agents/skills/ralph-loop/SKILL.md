---
name: ralph-loop
description: >-
  Autonomous execution loop based on the Ralph Wiggum technique by Geoffrey Huntley. Operates
  relentlessly across complex, multi-step backlogs by restarting with fresh context on every
  iteration and tracking persistent state via prd.json and progress.txt.
---

# Ralph Loop: Relentless Autonomous Execution Protocol

The Ralph Loop is an autonomous agent workflow designed for overnight execution of large backlogs. Instead of keeping a single long conversation open, the Ralph Loop runs in fresh context iterations, using the filesystem as the sole source of persistent state.

## Core Architecture

```
               ┌──────────────────────────────┐
               │    Persistent State (Disk)   │
               │   • prd.json                 │
               │   • progress.txt             │
               │   • Git History              │
               └──────────────┬───────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  New Agent Turn   │
                    │  (Fresh Context)  │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │ Execute Next Task │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │ Test & Git Commit │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │ Update prd.json   │
                    └───────────────────┘
```

## 1. Key State Files

### `prd.json` (Product Requirements Backlog)
```json
{
  "project": "Cortex-1 Release",
  "tasks": [
    {
      "id": "TASK-001",
      "title": "Add option ranking tests",
      "status": "done",
      "verification": "pytest tests/test_ranking.py"
    },
    {
      "id": "TASK-002",
      "title": "Package wheel distribution",
      "status": "pending",
      "verification": "python -m build"
    }
  ]
}
```

### `progress.txt` (Append-Only Log)
A chronological record of lessons learned, gotchas, and architectural decisions made by prior iterations to prevent looping.

## 2. The Ralph Iteration Rules
1. **Read State First:** On wakeup, read `prd.json` and `progress.txt`. Find the first `pending` task.
2. **Execute In Isolation:** Focus strictly on the single active task. Do not touch unrelated code.
3. **Verify Empirically:** Run the verification command. If it fails, fix it within the turn or record the blocker in `progress.txt`.
4. **Commit & Advance:** Mark the task `done` in `prd.json`, commit to Git (`git commit -m "task(TASK-XXX): ..."`), and terminate the turn cleanly.
