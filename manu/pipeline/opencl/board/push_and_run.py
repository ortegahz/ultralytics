#!/usr/bin/env python3
"""
push_and_run.py -- ship a file to the RK3588 board and run a command there.

Why this exists
---------------
Two board rules force the design, both recorded in manu/memory/rules.md sec.1b:

  1. The board disk has ~1.7 GiB free, so anything that is not required to be
     executable goes over NFS instead. This tool therefore pushes ONE file --
     the binary -- and leaves data and results on the share.
  2. Executables must NOT go over NFS: the board reports `Text file busy` when
     it runs a file the NFS server is still writing, and renaming does not help.
     Hence scp, not a copy into /mnt/manu.

Why scp driven by pexpect, and not base64 over an interactive shell
---------------------------------------------------------------------
An earlier attempt piped the payload through a heredoc on the interactive PTY.
The PTY echoed and re-chunked the base64 stream, and the board's `base64 -d`
ended up with a truncated file that still matched no checksum. Passing the same
bytes to scp as a real file transfer avoids PTY line discipline entirely.

The password is never written to disk: it comes from RK3588_PASSWORD or an
interactive prompt.

Usage:
    push_and_run.py push LOCAL REMOTE
    push_and_run.py run  REMOTE -- CMD [ARGS...]
    push_and_run.py pushrun LOCAL REMOTE -- CMD [ARGS...]
"""

import argparse
import hashlib
import os
import shlex
import sys

BOARD_HOST = os.environ.get("RK3588_HOST", "192.168.0.64")
BOARD_PORT = os.environ.get("RK3588_PORT", "22")
BOARD_USER = os.environ.get("RK3588_USER", "root")


def die(msg, code=1):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(code)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def password():
    pw = os.environ.get("RK3588_PASSWORD")
    if pw:
        return pw
    try:
        import getpass
        return getpass.getpass("board password: ")
    except Exception:
        die("no password: set RK3588_PASSWORD")


def expect_ok(ssh, pat, timeout=300, what=""):
    import pexpect

    idx = ssh.expect([pat, r"[Pp]assword:", pexpect.TIMEOUT], timeout=timeout)
    if idx == 0:
        return True
    if idx == 1:
        # Started from a clean prompt but asked for a password mid-transfer:
        # treat as failure rather than silently re-authenticating a half-scp.
        die(f"password requested during {what}")
    die(f"timeout during {what}")


def push(local, remote):
    import pexpect

    if not os.path.isfile(local):
        die(f"local file not found: {local}")
    digest = md5(local)
    print(f"[push] {local} ({os.path.getsize(local)} bytes, md5 {digest})")
    print(f"[push] -> root@{BOARD_HOST}:{remote}")

    ssh = pexpect.spawn(
        f"scp -o StrictHostKeyChecking=no -P {BOARD_PORT} "
        f"{shlex.quote(os.path.abspath(local))} "
        f"{BOARD_USER}@{BOARD_HOST}:{shlex.quote(remote)}",
        encoding="utf-8", timeout=300)
    # scp always prompts for the password first; anything else (host key
    # refusal, connection failure) must fail loudly instead of being answered
    # with a password.
    idx = ssh.expect([r"[Pp]assword:", pexpect.EOF, pexpect.TIMEOUT], timeout=60)
    if idx != 0:
        die(f"scp did not prompt for a password; output:\n{ssh.before}")
    ssh.sendline(password())
    idx = ssh.expect([pexpect.EOF, pexpect.TIMEOUT], timeout=300)
    tail = ssh.before or ""
    ssh.close()
    if idx != 0:
        die("scp timed out after authentication")
    # scp prints nothing on success and also exits quietly when the remote
    # directory is missing -- it only sets the exit status. Checking EOF alone
    # therefore reports "ok" for a transfer that never happened.
    if ssh.exitstatus is not None and ssh.exitstatus != 0:
        die(f"scp failed with exit status {ssh.exitstatus}; output:\n{tail}")
    print("[push] ok (scp exit 0)")
    return digest


def run(command, timeout=1800):
    import pexpect

    print(f"[run] {command}")
    ssh = pexpect.spawn(
        f"ssh -o StrictHostKeyChecking=no -p {BOARD_PORT} {BOARD_USER}@{BOARD_HOST}",
        encoding="utf-8", timeout=timeout)
    ssh.expect("[Pp]assword:")
    ssh.sendline(password())
    ssh.expect(r"[#$]\s*$", timeout=60)
    ssh.sendline(command)
    # The sentinel keeps the trailing shell prompt from being mistaken for
    # program output, and the capture group is what carries the exit status.
    ssh.expect(r"__PUSH_AND_RUN_DONE__=(\d+)", timeout=timeout)
    out = ssh.before
    rc = int(ssh.match.group(1))
    print("---- board output ----")
    print(out)
    print(f"---- remote exit code: {rc} ----")
    ssh.sendline("exit")
    ssh.close()
    return rc


def main():
    ap = argparse.ArgumentParser(
        description="push a file to the RK3588 board and/or run a command there")
    sub = ap.add_subparsers(dest="mode", required=True)

    p = sub.add_parser("push", help="scp LOCAL to REMOTE")
    p.add_argument("local")
    p.add_argument("remote")

    p = sub.add_parser("run", help="run a command on the board")
    p.add_argument("cmd", nargs=argparse.REMAINDER)

    p = sub.add_parser("pushrun", help="scp, then run")
    p.add_argument("local")
    p.add_argument("remote")
    p.add_argument("cmd", nargs=argparse.REMAINDER)

    args = ap.parse_args()

    if args.mode in ("push", "pushrun"):
        digest = push(args.local, args.remote)
        if args.mode == "push":
            print(f"[ok] local md5 {digest}")
            return 0

    if args.mode in ("run", "pushrun"):
        cmd = args.cmd
        if cmd and cmd[0] == "--":
            cmd = cmd[1:]
        if not cmd:
            die("run needs a command after --")
        # argv is already shell-free: argparse hands over separate words, and
        # re-quoting the joined string would send the quotes to the board
        # literally. Joining with spaces is only safe because every path and
        # argument here is a simple token.
        command = " ".join(cmd) + "; echo __PUSH_AND_RUN_DONE__=$?"
        return run(command)

    die("unreachable")


if __name__ == "__main__":
    sys.exit(main())
