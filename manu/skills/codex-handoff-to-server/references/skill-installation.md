## Where this skill must live to be discoverable

Codex does not scan every directory of the workspace. Measured 2026-11-03 with
`codex debug prompt-input hi`:

| Location | Discovered? |
|:--|:--|
| `$CODEX_HOME/skills/<name>/SKILL.md` (personal skill) | yes |
| `<cwd>/.codex/skills/<name>/SKILL.md` | yes (shows up in the Skill roots table) |
| `<cwd>/.agents/skills/<name>/SKILL.md` | yes (same) |
| `<cwd>/skills/<name>/SKILL.md` | **no** |
| `<sub>/.codex/skills/` inside a nested git repo | **no** - only the cwd's own copy counts |

Verification (copy-paste; run it on the host, from any cwd):

```bash
CODEX_HOME=/tmp/ch codex debug prompt-input hi | grep -oE '### Skill roots[^#]*'
```

Measured output for a cwd holding `.codex/skills/probe-cwd-codex/SKILL.md`,
`.agents/skills/probe-cwd-agents/SKILL.md` and `skills/probe-cwd-plain/SKILL.md`:

```
### Skill roots
- `r0` = `/tmp/skillprobe2/.codex/skills`
- `r1` = `/tmp/ch/skills`
- `r2` = `/tmp/ch/skills/.system`
- `r3` = `/tmp/skillprobe2/.agents/skills`
```

The available-skills list then contains `r3/probe-cwd-agents/SKILL.md` and
`r0/probe-cwd-codex/SKILL.md` - and **not** `skills/probe-cwd-plain/SKILL.md`, which is the proof
that `<cwd>/skills/` is invisible.

Note: `codex debug prompt-input` writes state into `CODEX_HOME`, so a read-only `~/.codex` makes it
fail with `Error: Read-only file system (os error 30)`. Point `CODEX_HOME` at a writable scratch
directory (e.g. `/tmp/ch`) instead.

For this skill: install it to `$CODEX_HOME/skills/codex-handoff-to-server/SKILL.md` (personal) or
`<cwd>/.codex/skills/codex-handoff-to-server/SKILL.md` (project). `<cwd>/skills/` does **not**
work.
