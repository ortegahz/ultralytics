---
name: codex-handoff-to-server
description: |
  Hand a live Codex conversation over from this host to the training server so the server can
  continue the same thread after the host is powered off. Load this skill whenever the user says
  "把对话交接到服务器", "让服务器接着干", "host 马上要关机", "服务器上的对话是空的", "在服务器上 resume
  这个会话", "把当前对话同步到服务器", or asks why a thread is invisible in the server's thread
  picker. It covers the cwd-keyed visibility rule, the adopt/import/resume mechanism, the
  one-command host-side transfer, the network-free read-only verification over the SSHFS mirror,
  the agent-side git-hook trigger for when the agent itself has no outbound network, and the
  TOML trap that silently keeps sandbox network access disabled. Do NOT use it to run training or
  evaluation, to judge a candidate's release verdict, or to curate datasets.
descriptions:
  zh-Hans: "把当前 Codex 对话交接到训练服务器：cwd 匹配可见性规则、adopt/import/resume 三步机制、host 一行命令传输、无网只读验收、agent 无网时的 git 钩子触发器、sandbox 网络开关的 TOML 陷阱。"
displayNames:
  zh-Hans: "对话跨机交接操作规程"
---

# Codex Handoff To Server

Move the **live** Codex thread of this workspace to the training server so the server can keep
working on it after this host is powered off. The scripts live in `handoff/` at the **workspace
root** (`/media/manu/1TB-Volume/workspace/handoff/`), i.e. a sibling of this repo, because they
touch the host-side Codex store, not the detector code.

Facts (what happened, with numbers) live in `manu/memory/codex_handoff_to_server_20261103.md`;
this file is the **method**.

## The invariant: a thread is visible only if its cwd matches

The server's Codex keeps threads in `$CODEX_HOME/state_5.sqlite` -> table `threads`. `thread/list`
filters by **exact `cwd` match**. It does **not** filter by model provider (that belief was
refuted by measurement), and it does not filter by "thread id known".

So a conversation that ran on the host is *present but invisible* on the server: its `cwd` is the
host workspace, the server's project directory is a different path. Every other thread already on
the server has the same problem, which is why the server's picker looked empty.

=> **The whole handoff reduces to making one thing true**: the server's copy of the thread has
`cwd` = the server's project directory. Sources:
`manu/memory/codex_handoff_to_server_20261103.md:31` (root cause) and `:166` (measured
reconciliation table).

## Mechanism: adopt -> import -> resume

| Step | What it must achieve | Implementation |
|:--|:--|:--|
| 1. adopt | rewrite the rollout's `session_meta.cwd` and `runtime_workspace_roots` to the target cwd | `handoff/handoff_helper.py adopt --rollout <f> --cwd <target>` |
| 2. import | place the rollout under `$CODEX_HOME/sessions/<Y>/<M>/<D>/`, point `threads.rollout_path` at it, repoint `threads.cwd`, clear the stale history projection, WAL-checkpoint | `handoff/server_takeover.sh` (runs on the server) |
| 3. resume | `thread/resume` materialises the paginated history index (`thread_history_1.sqlite` -> `thread_turns` + `thread_items`) | `handoff/handoff_helper.py resume --codex-home <h> --thread <id> --cwd <target>` |

Each of these silently breaks the chain when skipped - these are the measured failure modes:

- `history_mode` **must** be `paginated`. If the target row already exists with `legacy`,
  `thread/turns/list` answers "not supported" and the conversation looks empty even though the
  rollout file is right there.
- `title` / `preview` / `first_user_message` must be back-filled, or the picker shows a blank
  entry and the operator cannot tell which thread is which.
- The target `config.toml` must trust the target cwd and already have the model provider, or
  resume fails on the provider before it ever reads history.
- A stale history projection must be cleared, otherwise resume projects only the tail of the
  rollout.
- **Do not use `thread/fork`.** A forked thread is not visible in `thread/list`
  (`thread_source=NULL`); it can only work as a side effect of building the index. The handover
  object is the *source thread itself*, id unchanged.

Sources: `manu/memory/codex_handoff_to_server_20261103.md:52` (mechanism), `:142` (the fork
trap), `:186` (six simulated scenarios against the server's real state DB).

## Inputs to collect

| Item | Value for this project | Where it comes from |
|:--|:--|:--|
| Thread id | the live session id, e.g. `01a19eda-...` | the thread's own rollout file name under `~/.codex/sessions/<Y>/<M>/<D>/` |
| Server | `huangzhe@192.168.99.40` port `32222` | `manu/memory/rules.md` section 2.1 |
| Target cwd | `/tmp/pycharm_project_10ae9e2e` | the server's project root |
| Server `CODEX_HOME` | `/mnt/data/.codex` (**not** `/mnt/data/siping/.codex`) | `manu/memory/rules.md` section 2.1 |
| Server codex | `0.166.0` at `/home/huangzhe/.local/bin/codex` | measured on the server |

The thread id is **never hardcoded** in the scripts: `handoff_to_server.sh` exits 2 without
`--session`/`--latest`, because a stale id would silently ship the wrong conversation.

## Procedure

### 1. Pick the route

| Route | Who runs what | Use when |
|:--|:--|:--|
| **A - one command** | a human runs one line in a normal host terminal | one-off handover; zero prerequisites, nothing left behind |
| **B - agent-driven** | human runs `prepare_host_repo.sh` once, then the agent triggers the rest | repeated handovers, or when you want the agent to drive it end to end |

### 2. Route A - the one command

```bash
bash /media/manu/1TB-Volume/workspace/handoff/handoff_to_server.sh --session <thread-id>
```

It runs snapshot -> adopt -> local rehearsal -> ship -> server takeover -> verify, and prints the
server-side resume instructions. The rehearsal step prints how many history items the server
should end up with; that number is the acceptance value for step 4.

### 3. Route B - agent-driven trigger (agent has no outbound network)

Only needed when the agent cannot reach the network itself. The bridge is a git hook: the Codex
app-server creates its managed worktree with `git worktree add --no-checkout`, which **does not
fire `post-checkout`** but **does** fire `reference-transaction` (phase `committed`). Measured on
git 2.53.0 - see `manu/memory/codex_handoff_to_server_20261103.md:332` and `:394`.

```bash
# human, once, in a normal host terminal:
bash /media/manu/1TB-Volume/workspace/handoff/prepare_host_repo.sh
# the agent then calls codex_tui.create_worktree once; the hook fires and runs the same transfer
```

Two facts decide whether this route can work at all:

- The app-server runs `git rev-parse HEAD` **in the task's source directory** first, so that
  directory must be inside a git repository. Git discovery walks upward while `.git` is an empty
  or invalid *directory*, so the repository may live one level above the workspace.
- The workspace root here can never become a repository: `/`, `/media`, `/media/manu` and
  `/media/manu/1TB-Volume` are read-only for the agent, and `workspace/.git` is a `ro` tmpfs
  mount (`mode=555`; even `chmod` fails with `EROFS`). **Creating the trigger repository is
  therefore a human action** - `prepare_host_repo.sh` does it.

`prepare_host_repo.sh` reads `handoff/.handoff_session` (or `HANDOFF_SESSION=`) to arm the
trigger file with the session id in the same run, so the human only runs it once.

### 4. Verify without any network (read-only, over the SSHFS mirror)

The server's `CODEX_HOME` is reachable read-only at `/home/manu/mnt/data/.codex`, so the agent
can confirm the handover landed without opening a socket. Copy the DBs out first - SQLite needs
the `-wal` sidecar to see recent writes, and `immutable=1` silently ignores it:

```bash
M=/home/manu/mnt/data/.codex
ls -l "$M/sessions/2026/11/03/" | grep <thread-id>
sha256sum "$M/sessions/2026/11/03/rollout-...<thread-id>.jsonl"
grep -c "/media/manu/1TB-Volume/workspace" "$M/sessions/2026/11/03/rollout-...<thread-id>.jsonl"   # expect 0
grep -o "/tmp/pycharm_project_10ae9e2e" "$M/sessions/2026/11/03/rollout-...<thread-id>.jsonl" | wc -l

mkdir -p /tmp/threadcheck && cp "$M"/state_5.sqlite* "$M"/thread_history_1.sqlite* /tmp/threadcheck/
python3 - <<'PY'
import sqlite3
con = sqlite3.connect("/tmp/threadcheck/state_5.sqlite")
for r in con.execute("select id, cwd, history_mode, model, archived from threads where id=?", ("<thread-id>",)):
    print(r)
con.close()
con = sqlite3.connect("/tmp/threadcheck/thread_history_1.sqlite")
print("turns:", con.execute("select count(*) from thread_turns where thread_id=?", ("<thread-id>",)).fetchone()[0])
print("items:", con.execute("select count(*) from thread_items where thread_id=?", ("<thread-id>",)).fetchone()[0])
PY
```

Acceptance: exactly one `threads` row with `cwd` = the server project dir, `history_mode=paginated`,
`archived=0`; the old host cwd appears **0** times inside the rollout; `thread_turns`/`thread_items`
counts equal the rehearsal numbers printed by `handoff_to_server.sh`.

### 5. Verify with network

```bash
bash /media/manu/1TB-Volume/workspace/handoff/server_continue.sh                # read-only proof
bash /media/manu/1TB-Volume/workspace/handoff/server_continue.sh --probe-turn   # + one real turn
bash /media/manu/1TB-Volume/workspace/handoff/server_continue.sh --resync       # + re-ship the snapshot
```

`--probe-turn` spends one turn of the server's model quota and is the only way to prove the
server can genuinely *continue* the work rather than merely list it.

The script already exports `PATH="$HOME/.local/bin:$PATH"` and passes `--skip-git-repo-check`
to `codex exec resume`, because a non-interactive `ssh` does not source the profile (so `codex`
is not on `PATH`) and the server's project directory is not a git repository. If you run the
resume by hand over `ssh`, do the same:

```bash
ssh -p 32222 huangzhe@192.168.99.40 \
  "export PATH=\"\$HOME/.local/bin:\$PATH\" && export CODEX_HOME=/mnt/data/.codex && \
   cd /tmp/pycharm_project_10ae9e2e && \
   codex exec resume --skip-git-repo-check <thread-id> '继续之前的工作'"
```

### 6. Continue by hand on the server

```bash
export CODEX_HOME=/mnt/data/.codex
cd /tmp/pycharm_project_10ae9e2e
codex resume <thread-id>
# headless, one turn:
codex exec resume <thread-id> "继续之前的工作"
```

`CODEX_HOME` must point at the store above; without it `codex` reads the wrong (empty) store.

For an **unattended** continuation (host off, nobody logged in), start it detached instead of
interactively - `setsid nohup` plus a wall-clock cut-off, the same discipline as an overnight
training chain (rules 10 and 11):

```bash
ssh -p 32222 huangzhe@192.168.99.40 \
  "export PATH=\"\$HOME/.local/bin:\$PATH\" && export CODEX_HOME=/mnt/data/.codex && cd /tmp/pycharm_project_10ae9e2e && \
   setsid nohup codex exec resume <thread-id> '继续之前的工作' > /tmp/codex_resume.log 2>&1 < /dev/null &"
```

### 7. Re-sync later, without losing server-side work

The import is a **snapshot**. Turns the host adds afterwards are not on the server, and re-running
the transfer overwrites the server's rollout with the host's copy.

- `handoff/server_takeover.sh` refuses to overwrite when the server's copy is **larger** unless
  `--force` is given, and it now keeps the previous file as `<rollout>.bak.<ts>` before every
  overwrite. Treat that backup as the undo.
- After the server takes over, **the server is the owner of the conversation**. Re-sync
  deliberately, and back up `$CODEX_HOME/sessions/<Y>/<M>/<D>/` first.
- The one-shot trigger file `/tmp/codex-handoff-go` must be disarmed once the transfer is done
  (move it aside), otherwise a stray worktree creation re-imports and clobbers the live thread.

### 8. Pull it back - server to host after the host reboots

Pushing is one-way. After the host powers off and the server keeps working, the newer
conversation exists only on the server. Bringing it back is a second, reverse transfer.

```bash
# read-only comparison, writes nothing
bash /media/manu/1TB-Volume/workspace/handoff/handoff_from_server.sh \
  --session 01a19eda-6f3b-7552-928d-b16b7e377893 --check

# actually pull the server's copy back into this host
bash /media/manu/1TB-Volume/workspace/handoff/handoff_from_server.sh \
  --session 01a19eda-6f3b-7552-928d-b16b7e377893 --adopt
```

Both run in a **normal terminal on the host** - the agent sandbox blocks outbound sockets.

What it does, and what it refuses to do:

- The server is **read-only**: `ssh` runs `find`/`stat`, `scp` reads the file. Nothing is ever
  written to the server. The fetched copy lands in `/tmp/codex-pullback-XXXXXX/` and is removed
  on exit.
- `handoff_helper.py adopt` rewrites the cwd back to the host's
  (`/tmp/pycharm_project_10ae9e2e` -> `/media/manu/1TB-Volume/workspace`). Measured on the real
  server copy: 5640 lines, **zero** unparseable JSON, `session_meta.cwd` and all 17
  `turn_context.cwd` values are the host cwd, and the old server cwd occurs **0** times.
- **liveness guard** - if the host's own copy was written within `--liveness-seconds`
  (default 120), a local session is still writing this thread; refuse rather than let two
  writers corrupt one rollout.
- **ahead guard** - if the host holds more `turn_context` records than the server, pulling would
  drop those turns; refuse. `--force` overrides both guards.
- The import reuses `handoff/server_takeover.sh`, so host and server stay on one code path: copy
  into `<codex home>/sessions/<Y>/<M>/<D>/`, repoint the `threads` row, clear the stale history
  projection, rebuild the paginated index, verify.

One trap worth knowing: `server_takeover.sh` refuses when the target file is **byte-wise larger**
than the incoming snapshot. On a pull-back the target is the host's own copy, which is often
larger for reasons that are not extra turns - host-side bookkeeping, and the host cwd string is
three characters longer than the server's in every occurrence (32 vs 29 bytes). `handoff_from_server.sh` therefore
passes `--force` to that script when the **turn count** says the server is ahead; the log line
`byte guard: skipped` records it. `--force` is honoured in exactly one place inside
`server_takeover.sh`, so nothing else is weakened.

Measured end to end on the host against the real server copy, into an isolated
`CODEX_HOME=/tmp/pb_home2` (the real `~/.codex` was not touched): `verify` returned
`ok: true`, `visible_in_picker: true`, `listed_threads: 1`, `turns: 11`, `items: 1704`,
`first_user_message` = the first prompt of the thread, `list_cwd` =
`/media/manu/1TB-Volume/workspace`.

The file name must carry the thread id: codex resolves a thread by the id embedded in the
rollout **file name** (the app-server scans `sessions/` for `rollout-*<id>.jsonl`). A staged file
without it imports without complaint and then fails on `thread/resume` with
`no rollout found for thread id ...`; the script checks for it.

After the pull-back the script prints the command to continue:

```bash
export CODEX_HOME=<host codex home>
cd /media/manu/1TB-Volume/workspace
codex resume 01a19eda-6f3b-7552-928d-b16b7e377893
```

Note the two-rows-one-id situation: after the push, host and server each hold **one row for the
same thread id** in their `state_5.sqlite`, but with different `cwd` values. The picker filters
on an exact cwd match, so each machine only shows the copy whose cwd is its own. The pull-back
collapses that fork back onto the host's row.

A pull-back is a **snapshot**, not a live sync: the server's copy is left untouched and stays
the server's, and from then on the two diverge independently.

## Enabling agent network access and write roots - the TOML placement trap

The agent sandbox blocks outbound sockets and writes to `~`. Both permissions live in the **same
top-level** `[sandbox_workspace_write]` table of `~/.codex/config.toml` - putting them under
`[projects."..."]` silently does nothing. The exact TOML, the warning codex prints when it is in
the wrong table, the proof (the startup banner), the `.git/`/`.codex/` read-only gotcha and the
"the host filesystem is NOT read-only" clarification are all in
[references/permissions.md](references/permissions.md).

## Where this skill must live to be discoverable

Codex does not scan every directory of the workspace - only a short list of roots. The measured
discovery rules, the verification command and the install commands are in
[references/skill-installation.md](references/skill-installation.md).


## Overnight托管：让服务器自己把这个对话跑一整夜

A handover is a **disk snapshot**, not a live process. The server only keeps working if something on
the server resumes the thread. Three facts, all measured on 2026-11-03:

| Fact | Consequence |
|:--|:--|
| `codex exec resume <id> "<prompt>"` runs **exactly one turn and exits** | one background invocation buys one turn; to work "all night" the server needs a **supervisor loop** that re-resumes the *same* thread id until a wall-clock deadline |
| the host powers off at 18:30 -> the ssh session dies -> SIGHUP | the worker must be launched with `setsid nohup ... < /dev/null &` so it is not in the ssh session's process group |
| two writers on one rollout file corrupt the thread | "watch" (`tail -f loop.log`) and "take over" (`codex resume`) are mutually exclusive - stop the loop first |

The shipped implementation is `handoff/overnight/server_overnight_loop.sh` (server copy:
`/mnt/data/codex-overnight/server_overnight_loop.sh`), with `handoff/overnight/prompt_first.txt` /
`prompt_next.txt` as the per-round prompts and `handoff/overnight/server_stop_session.sh` as the
stop switch. State directory: `/mnt/data/codex-overnight/` (`loop.log`, `STOP` flag, `last_turn.txt`).

```bash
# start (host -> server; survives the host powering off)
ssh -F /dev/null -p 32222 huangzhe@192.168.99.40 \
  'setsid nohup bash /mnt/data/codex-overnight/server_overnight_loop.sh \
      --session <id> --until "2026-11-04 08:40" --cwd /tmp/pycharm_project_10ae9e2e \
      > /mnt/data/codex-overnight/launcher.log 2>&1 < /dev/null &'

# watch (read-only, zero risk)
ssh -F /dev/null -p 32222 huangzhe@192.168.99.40 'tail -f /mnt/data/codex-overnight/loop.log'

# take over interactively - stop the loop first, or the thread gets two writers
ssh -F /dev/null -p 32222 huangzhe@192.168.99.40 \
  'bash /mnt/data/codex-overnight/server_stop_session.sh --session <id>'
ssh -F /dev/null -p 32222 huangzhe@192.168.99.40 \
  'export CODEX_HOME=/mnt/data/.codex PATH="$HOME/.local/bin:$PATH"; cd /tmp/pycharm_project_10ae9e2e && codex resume <id>'

# pull back to the host next morning
bash /media/manu/1TB-Volume/workspace/handoff/handoff_from_server.sh --session <id> --adopt --force
```

Three traps that cost a full debugging round on 2026-11-03:

1. **`--sandbox` and `--approve-for-me` are mutually exclusive** in codex 0.166.0
   (`the argument '--sandbox <SANDBOX_MODE>' cannot be used with '--approve-for-me'`).
   For an unattended run use `--approve-for-me` alone - it already implies the workspace-write
   sandbox and routes escalations through automatic review - plus `--add-dir <dir>`.
2. **Backgrounding a compound command keeps the ssh channel open.** `cd X && setsid ... &` leaves
   the background subshell holding stdout, so `ssh` never returns. Redirect the *whole* background
   command (`... > log 2>&1 < /dev/null &`) and do not `cd` inside the same list.
3. **`pkill -f` matches your own command line.** `pkill -f "server_overnight_loop.sh"` kills the
   shell that is running the pkill, because the pattern is in its own argv. Use the bracket trick
   (`"server_overnight_l[o]op.sh"`) and never pattern on the bare word `codex`, which also matches
   the `codex app-server` daemons that back the thread picker.

The agent can now do the whole transfer itself: with `sandbox_workspace_write.network_access = true`
the sandbox allows outbound sockets, so `ssh`/`scp` work from inside the agent session without
handing a command to the user. `handoff/enable_network.sh` remains for the case where the switch is
off.

## Goals do not travel with the handoff

A handoff moves the **conversation**, not the **goal**. The goal lives in its own store, on one machine:
`<CODEX_HOME>/goals_1.sqlite`, table `thread_goals`, whose primary key is `thread_id`. It is **not** in
the rollout jsonl, and it is **not** carried by `thread/fork`, `adopt` or `import`.

Consequence: on the server `codex resume <server-thread-id>` shows the conversation, but `/goal` is
**empty** - the objective has to be spelled out in the prompt, and the agent can easily drift.

Reconcile `thread_id` against `goals_1.sqlite` before and after the transfer. If the target machine has
no `active` row for that `thread_id`, write one:

```bash
# write the goal row on the server (INSERT OR REPLACE: the same thread_id may already hold a stale row)
ssh -F /dev/null -p 32222 huangzhe@192.168.99.40 'python3 - <<"PYEOF"
import sqlite3, time, uuid
db = "/mnt/data/.codex/goals_1.sqlite"
c = sqlite3.connect(db)
now = int(time.time() * 1000)
c.execute(
    "INSERT OR REPLACE INTO thread_goals "
    "(thread_id, goal_id, objective, status, token_budget, tokens_used, time_used_seconds, created_at_ms, updated_at_ms) "
    "VALUES (?,?,?,?,?,?,?,?,?)",
    ("<THREAD_ID>", str(uuid.uuid4()), "<OBJECTIVE>", "active", None, 0, 0, now, now),
)
c.commit()
c.close()
print("goal row written")
PYEOF'

# read-only verification
ssh -F /dev/null -p 32222 huangzhe@192.168.99.40 "python3 -c \"import sqlite3;[print(r[0],r[1]) for r in sqlite3.connect('file:/mnt/data/.codex/goals_1.sqlite?mode=ro',uri=True).execute('SELECT thread_id,status FROM thread_goals')]\""
```

Two things to keep in mind on the way back:

- **Pulling back to the host does not bring that sqlite file along.** Either `scp` the server's
  `goals_1.sqlite` back, or hand-write the row into the host's `goals_1.sqlite`. The host may already
  hold a **stale row for the same `thread_id`** (measured 2026-11-03: `status=complete`), so use
  `INSERT OR REPLACE` to overwrite it - never an insert that would collide with the primary key.
- **Sub-agent threads have their own goal row** (measured 2026-11-03:
  `01a1a03f-8b26-7612-ab8f-4fcb0fb37bc6`) - do not miss them when reconciling at the end of a run.

After the transfer, write one explicit line in the conversation - "goal synced to `<thread_id>`,
status=active" - as the evidence that the goal actually travelled.

## Failure modes

| Symptom | Cause | Fix |
|:--|:--|:--|
| Server picker still empty, thread file present | `threads.cwd` != server project dir | re-import; confirm `cwd` in `state_5.sqlite` |
| Conversation "looks empty" although the rollout is there | `history_mode=legacy` on an existing row | force `paginated` during takeover |
| `thread/turns/list` says "not supported" | same as above | same |
| Forked thread never appears | `thread/fork` products have `thread_source=NULL` | import the source thread instead of forking |
| `codex resume` on the server finds nothing | `CODEX_HOME` not exported, or points at `/mnt/data/siping/.codex` | `export CODEX_HOME=/mnt/data/.codex` |
| "Not inside a trusted directory" | target cwd not trusted | the takeover adds `[projects."<cwd>"] trust_level="trusted"`; otherwise add it by hand |
| `socket: Operation not permitted` from the agent | sandbox network off, or the TOML trap above | `handoff/enable_network.sh` + a new session |
| `create_worktree` fails with "not a git repository" | app-server resolved no repository in the task's source dir | human runs `handoff/prepare_host_repo.sh` once |
| Agent sees the thread in the DB but the picker hides it | cwd mismatch (the original root cause) | this is the thing the whole skill exists to fix |
| Pull-back refuses: "host copy was modified Ns ago" | a local session is still writing this thread | stop it, or raise `--liveness-seconds` / pass `--force` |
| Pull-back refuses: "the host copy holds N turns" | the host is ahead; pulling would drop those turns | pull only if that is intended (`--force`), else merge by hand |
| `no rollout found for thread id ...` after an import | the staged file name does not carry the thread id | keep the `rollout-*<id>.jsonl` name |
| `codex resume` on the server: "daemon executable not found at <CODEX_HOME>/packages/app-server-daemon/current/bin/codex" | the `current` symlink was copied from another host and points at a path that does not exist here | `ln -sfn <CODEX_HOME>/packages/app-server-daemon/releases/<version> <CODEX_HOME>/packages/app-server-daemon/current` |
| `codex resume` on the server: "failed to connect to <CODEX_HOME>/app-server-control/app-server-control.sock" | the control-socket symlink points at another user's stale daemon dir | `rm -f` that symlink, then `CODEX_HOME=<h> codex app-server daemon start`; or bypass the daemon with `codex resume --no-daemon <id>` |
| `codex exec` dies with "the argument '--sandbox <SANDBOX_MODE>' cannot be used with '--approve-for-me'" | the two flags are mutually exclusive in 0.166.0 | use `--approve-for-me` alone (it implies workspace-write) |
| `ssh` hangs and never returns after backgrounding the worker | the backgrounded compound still holds stdout | redirect the whole background command and drop the `cd` from that list |
| `pkill -f` kills the shell issuing the pkill | the pattern is in the pkill's own argv | bracket trick: `pkill -f "server_overnight_l[o]op.sh"` |
| Server does nothing at night although the snapshot is in place | `codex exec resume` runs one turn and exits | run the supervisor loop (`handoff/overnight/server_overnight_loop.sh`) |
| `FileNotFoundError: [Errno 2] No such file or directory: 'codex'` while driving the server | a non-interactive `ssh` does not source the profile, so `codex` is not on `PATH` (on that box it lives in `~/.local/bin`) | `handoff/handoff_helper.py` resolves it with `find_codex()`; by hand use `export PATH="$HOME/.local/bin:$PATH"` before `codex exec resume` |
| Server thread shows the conversation but `/goal` is empty | the goal lives in `<CODEX_HOME>/goals_1.sqlite` keyed by `thread_id`, and it is not carried by fork/adopt/import | INSERT OR REPLACE a row for the server-side `thread_id` into `goals_1.sqlite`; on pull-back, copy that row back to the host's `goals_1.sqlite` |

## Rules that apply

- **Local first** - edit scripts and docs in the local checkout; never write to the server
  directly; prove a sync by md5 on both sides (`manu/memory/rules.md` section 0).
- **Cross-machine state migration** (rule 12) - simulate against the target's real state DB,
  reconcile the fields written during adoption, pre-check and auto-add the target's trust entry
  and provider, and protect directionally before overwriting
  (`manu/memory/rules.md:403`).
- **Heavy work belongs on the server** (rule 13) - a handover is not heavy work, but anything
  that trains, evaluates or rebuilds caches still needs explicit user authorisation.
- **Change the server by `scp`-ing one file, never by re-running the whole handoff** -
  `handoff_to_server.sh` pushes the host's snapshot (25 turns) over the server's own copy
  (17 turns) and destroys the work the server already did. Ship only the single changed file; the
  server keeps the previous version as `<file>.bak.<ts>`.
- **Record the command, the machine and the artefact** for every conclusion (rule 9).

## Command appendix

| Purpose | Command | Machine | Artefact |
|:--|:--|:--|:--|
| transfer (route A) | `bash /media/manu/1TB-Volume/workspace/handoff/handoff_to_server.sh --session <id>` | host terminal | server `$CODEX_HOME/sessions/<Y>/<M>/<D>/rollout-....jsonl` |
| local rehearsal only | same command plus `--dry-run` | any | stdout: expected `items` / `turns` |
| server-side verify | `bash <remote>/verify_on_server.sh --thread <id>` | server | stdout JSON (`ok: true`) |
| trigger repo (route B) | `bash /media/manu/1TB-Volume/workspace/handoff/prepare_host_repo.sh` | host terminal | `handoff/.host_repo_state` |
| undo route B | `bash /media/manu/1TB-Volume/workspace/handoff/undo_host_repo.sh` | host terminal | - |
| hook regression | `bash /media/manu/1TB-Volume/workspace/handoff/test_hook_regression.sh` | any | stdout `PASS: 90    FAIL: 0` |
| read-only server forensics | `python3 /media/manu/1TB-Volume/workspace/handoff/server_state_check.py` | any (needs network) | stdout |
| enable agent network | `bash /media/manu/1TB-Volume/workspace/handoff/enable_network.sh` | host terminal | `~/.codex/config.toml` + backup |
| continue / re-sync | `bash /media/manu/1TB-Volume/workspace/handoff/server_continue.sh [--probe-turn|--resync]` | host terminal or networked agent | stdout |
| pull back (compare only) | `bash /media/manu/1TB-Volume/workspace/handoff/handoff_from_server.sh --session <id> --check` | host terminal | stdout: side-by-side turn/byte table |
| overnight start | `ssh -F /dev/null -p 32222 huangzhe@192.168.99.40 'setsid nohup bash /mnt/data/codex-overnight/server_overnight_loop.sh --session <id> --until "2026-11-04 08:40" --cwd /tmp/pycharm_project_10ae9e2e > /mnt/data/codex-overnight/launcher.log 2>&1 < /dev/null &'` | host -> server | server `/mnt/data/codex-overnight/loop.log` |
| overnight watch | `ssh -F /dev/null -p 32222 huangzhe@192.168.99.40 'tail -f /mnt/data/codex-overnight/loop.log'` | any | live stream of the server turns |
| overnight stop | `ssh -F /dev/null -p 32222 huangzhe@192.168.99.40 'bash /mnt/data/codex-overnight/server_stop_session.sh --session <id>'` | any | `STOP` flag + in-flight turn killed |
| pull back (install) | `bash /media/manu/1TB-Volume/workspace/handoff/handoff_from_server.sh --session <id> --adopt` | host terminal | host `<codex home>/sessions/<Y>/<M>/<D>/rollout-....jsonl` |

## Windows (win32) platform notes

Every script here is POSIX bash and assumes a Linux host plus `ssh`/`scp`. On win32, run them from
WSL (the paths under `/media/manu/...` must be mounted) or port the shell steps to PowerShell;
the mechanism itself (adopt -> import -> resume, and the cwd-matching rule) is platform
independent.
