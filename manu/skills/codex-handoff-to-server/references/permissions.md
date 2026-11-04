## Enabling agent network access and write roots - the TOML placement trap

The agent sandbox blocks `socket(AF_INET)` outright (`EPERM`), so it cannot scp/ssh. To let it
reach the server, `~/.codex/config.toml` must contain a **top-level** table:

```toml
[sandbox_workspace_write]
network_access = true
```

The trap: if that line is written **after** another table header, TOML reads it as a dotted key
*inside* that table and the setting silently does nothing. Measured on this host - the line sat
under `[projects."/media/manu/1TB-Volume/workspace"]`, and `tomllib` reported
`sandbox_workspace_write = None` while the real value hid inside the project entry. The sandbox
kept blocking sockets even though the file "looked" correct.

Codex itself names the mistake. Run a throwaway session and read its banner and warnings:

```
sandbox: workspace-write [workdir, /tmp, $TMPDIR] (network access enabled)     <- top-level table: honoured
warning: user (config.toml): `projects./media/.../workspace.network_access` is ignored.   <- misplaced: ignored
```

Two distinct shapes of the mistake - **both** have to go:

```toml
[projects."/media/manu/1TB-Volume/workspace"]
trust_level = "trusted"
sandbox_workspace_write.network_access = true   # shape 1: dotted key inside a project table
network_access = true                            # shape 2: bare key inside a project table
```

Fix and verify in one command (it backs the file up first, removes **both** shapes, adds the
real top-level table and re-parses the file with `tomllib`):

```bash
bash /media/manu/1TB-Volume/workspace/handoff/enable_network.sh
```

Measured on the real file: `misplaced lines removed: 2`,
`top-level 'sandbox_workspace_write' = {'network_access': True}`, `still nested under a project
table: none`, `RESULT: PASS`.

**No-file alternative** - the runtime override works on its own and needs no edit at all
(measured: a config.toml containing zero `network_access` lines still printed
`(network access enabled)`):

```bash
codex -c sandbox_workspace_write.network_access=true
```

Then probe: `python3 -c "import socket; socket.socket(); print('SOCKET_OK')"`. If it still fails,
the running sandbox was created with the old profile - start a **new** Codex session on this host
(the config is read when the sandbox is created).

### The same table also carries the write roots

**Write permission is the second field of the very same table.** `writable_roots` is a **sibling**
of `network_access` - never a child of a `[projects."..."]` table:

```toml
# ~/.codex/config.toml - correct: both fields in the TOP-LEVEL table
[sandbox_workspace_write]
network_access = true
writable_roots = ["/home/manu/.codex", "/home/manu/.minimax"]
```

```toml
# wrong: inside a project table -> codex ignores it and says so
[projects."/media/manu/1TB-Volume/workspace"]
trust_level = "trusted"
writable_roots = ["/home/manu/.codex"]
```

Measured on this host 2026-11-03 14:26-14:35: the misplaced form makes codex print

```
warning: user (config.toml): projects./media/.../workspace.network_access is ignored.
```

and the decisive evidence is the **startup banner of a child `codex exec`**, before vs after:

```
sandbox: workspace-write [workdir, /tmp, $TMPDIR] (network access enabled)
sandbox: workspace-write [workdir, /tmp, $TMPDIR, /home/manu/.codex, /home/manu/.minimax] (network access enabled)
```

Equivalent CLI switch for a single run (nothing written to disk):

```bash
codex --add-dir /home/manu/.codex --add-dir /home/manu/.minimax
```

**A new session is mandatory** - the config is frozen when the process starts, so the banner of a
running session never changes. That banner line is the only verdict that matters.

### Gotcha: `.git/`, `.codex/` and friends stay read-only even inside a writable root

Some paths are locked by the harness regardless of `writable_roots`. Measured inside the agent
sandbox on this host (`/proc/mounts`):

```bash
grep -E "workspace/(\.codex|\.agents|\.git|\.aws)" /proc/mounts
```

```
tmpfs /media/manu/1TB-Volume/workspace/.codex tmpfs ro,...,mode=555
tmpfs /media/manu/1TB-Volume/workspace/.agents tmpfs ro,...,mode=555
tmpfs /media/manu/1TB-Volume/workspace/.git   tmpfs ro,...,mode=555
tmpfs /media/manu/1TB-Volume/workspace/.aws   tmpfs ro,...,mode=555
```

Writing into any of them fails with `Read-only file system (os error 30)`.

### Clarification: the host filesystem is NOT read-only

The read-only view belongs to the **agent sandbox**, not to the host disk. Three pieces of
evidence, all measured on the host on 2026-11-03:

- the app-server keeps writing `~/.codex/sessions/2026/11/03/rollout-....jsonl` (its mtime
  advances in real time) - it runs outside the sandbox;
- the human edited `~/.codex/config.toml` by hand at 14:24 (mtime `2026-11-03 14:24:37`);
- `/proc/mounts` shows `/media/manu/1TB-Volume` mounted `ro` while the `<workspace>` sub-path is a
  **rw** bind - the `ro` flag is per mount point and bwrap can bind a subtree as `rw`:

  ```
  /dev/sdc1 /media/manu/1TB-Volume ext4 ro,nosuid,nodev,relatime 0 0
  /dev/sdc1 /media/manu/1TB-Volume/workspace ext4 rw,nosuid,nodev,relatime 0 0
  ```

=> **In the human's own terminal, `cp` into `~/.codex/skills/` always succeeds - no config change
is needed.** From inside the sandbox the agent cannot do it; the human can.

**Hard limits of the current harness** (measured 2026-11-03, so you do not have to re-derive them):

- The sandbox is **frozen at session start** - editing `config.toml` cannot change the running
  session; a new session is required.
- **Inside the sandbox** the root filesystem is mounted **`ro`** (the *host* disk is not - see the
  clarification above) and `socket(AF_INET)` fails with `EPERM` even to `127.0.0.1`. `seccomp`
  filters are inherited and cannot be lifted, so a nested `codex exec` / `codex sandbox` child
  process inherits the same wall even when its own banner says `(network access enabled)`.
- `/home/manu/.codex/app-server-control/app-server-control.sock` points into
  `/tmp/codex-daemon-1000/`, which is mode `000` on a `ro` mount on purpose - the harness closes
  the app-server-socket escape route explicitly.
- Consequence: **the agent cannot write `~` and cannot reach the network from inside a running
  session.** Both are one host-terminal command away, but the human has to run them.
