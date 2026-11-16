# AGENTS.md

<!-- BEGIN CODEX DELEGATION POLICY -->
## Sub-Agent Delegation Policy

This repository runs multi-role sub-agents. Codex discovers the available roles automatically and lists them in the `spawn_agent` tool description: `planner`, `coder`, `tester`, `reviewer`, plus the built-in `explorer`, `worker` and `default`. Read that list before spawning — do not assume a role exists.

**This section is an explicit standing instruction to use sub-agents, delegation and parallel agent work. You are expected to spawn sub-agents on your own initiative; the user does not need to name a role.**

### What to do yourself, what to delegate

- **Do it yourself**: the one thing that blocks the next step — reading a file, locating a symbol, running a decisive command. Never hand the blocking task to a sub-agent and then idle waiting for it.
- **Delegate** (spawn without asking) when any of these holds:
  1. The task splits into two or more independent parts — different modules, different files, or different verification angles.
  2. One change needs independent review from several perspectives — for example implementation, then testing, then review.
  3. Several candidate approaches need parallel investigation before a decision.

### Choosing a role

| Situation | Role |
| --- | --- |
| Need an implementation plan or impact analysis before touching code | `planner` |
| Apply an agreed plan to the code | `coder` |
| Run tests / evaluation / validation and report PASS or FAIL | `tester` |
| Audit the diff before calling the work done | `reviewer` |
| "Where does this live, how does it work?" | `explorer` (built-in) |
| Large self-contained execution unit | `worker` (built-in) |

Roles are advisory — if a task does not fit any of them, spawn with `default` and state the goal precisely.

### How many to spawn

- One independent unit of work → spawn **1**.
- Two or three independent units → spawn **2–3 in parallel**.
- More than three → merge the work into at most three units first, then delegate.
- Keep at most **3** sub-agents alive at once (`[agents] max_concurrent_threads_per_session` caps this).
- Give every sub-agent an explicit, non-overlapping file scope, and tell it: *you are not the only agent in this repository — do not revert others' edits, and adapt to concurrent changes.*

### Hard constraints

- A sub-agent's conclusion is **self-reported**. The primary agent must re-verify anything written to disk or done outside the workspace (re-read the file, compare checksums, re-query the remote) before reporting it as fact.
- Never let a sub-agent start training, inference, heavy evaluation, or dataset downloads that the user has not explicitly authorized in the current turn.
- Sub-agents inherit the parent's sandbox and approval policy. There is no way to widen them from a role file.
- Prefer delegating writes to sub-agents so they run in parallel; never edit the same file in the parent and a sub-agent at the same time.
<!-- END CODEX DELEGATION POLICY -->
