---
name: coderabbit
description: >-
  Automated code review, security static analysis, and PR triage engine. Identifies potential
  regressions, performance bottlenecks, race conditions, AppSec vulnerabilities (OWASP/CWE), and
  test coverage gaps before code is merged or published.
---

# CodeRabbit: Autonomous Code Review & Quality Gating

CodeRabbit provides senior-level code review and automated pull-request triage directly within the agent workflow.

## 1. Review Checklist & Triage Protocol

Every code review runs through 5 distinct analysis gates:

### Gate 1: Security & Vulnerabilities (AppSec)
- [ ] **Data Injection:** SQL injection, command execution, prototype pollution.
- [ ] **Secret Leaks:** Hardcoded API keys, private tokens, passwords, database URIs.
- [ ] **Auth & Access:** Missing authorization checks, CSRF, insecure direct object references (IDOR).
- [ ] **Cryptographic Weaknesses:** Weak hashing algorithms (MD5/SHA1 for passwords), insecure random numbers.

### Gate 2: Correctness & Race Conditions
- [ ] **Async / Concurrency:** Unhandled promise rejections, race conditions in shared mutable state.
- [ ] **Resource Leaks:** Unclosed file descriptors, database connections, uncollected event listeners.
- [ ] **Boundary Conditions:** Off-by-one errors, empty lists, null/undefined pointer dereferences.

### Gate 3: Performance & Scalability
- [ ] **Database Queries:** N+1 query patterns, missing indices on filtered foreign keys.
- [ ] **Memory Footprint:** Unbounded buffer growth, retaining large objects in closures.
- [ ] **Algorithmic Complexity:** Accidental $O(N^2)$ loops inside hot paths.

### Gate 4: Test Coverage & Regression Safety
- [ ] Are new codepaths covered by explicit unit tests?
- [ ] Are failure paths and error handling tested, not just the happy path?

### Gate 5: Architecture & Maintainability
- [ ] Does the change respect established repository patterns?
- [ ] Are public interfaces properly typed and documented?

## 2. Review Output Format

Review findings must be formatted as actionable, severity-ranked comments:

```markdown
### CodeRabbit Review Summary

| Severity | Count | Status |
| :--- | :--- | :--- |
| 🚨 Critical | 0 | PASSED |
| ⚠️ Warning | 1 | ACTION REQUIRED |
| 💡 Suggestion | 2 | OPTIONAL |

#### ⚠️ Warning: Potential Connection Pool Leak
- **Location:** `database/connection.py:45-52`
- **Issue:** Connection acquired from pool is not wrapped in `try/finally` or context manager.
- **Fix:** Use `with pool.acquire() as conn:` to guarantee release upon exceptions.
```
