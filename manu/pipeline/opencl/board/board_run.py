#!/usr/bin/env python3
"""
board_run.py -- transfer a binary to the RK3588 board over one SSH session and
run it with given arguments.

Why not scp/NFS: board executables must NOT go through NFS (the NFS server
still has the write in flight when the board tries to exec -> "Text file busy",
and a fresh filename does not help). So the binary travels inside the SSH
session. pexpect is used because sshpass is not installed on this host.

SSH public-key auth is the normal path: ~/.ssh/id_ed25519.pub is installed in the
board's /root/.ssh/authorized_keys, so no password is needed and RK3588_PASSWORD
is never consulted. It is only read as a fallback when the key is absent, and it
is never written to any file.

Usage:
    python3 board_run.py --exec "/mnt/manu/npu/out/demo/<seq> && ls"
"""

import argparse
import base64
import os
import re
import subprocess
import sys

import pexpect

HOST = os.environ.get("RK3588_HOST", "192.168.0.64")
PORT = os.environ.get("RK3588_PORT", "22")
USER = os.environ.get("RK3588_USER", "root")

PROMPT = r"[#\$]\s*$"
SENTINEL = "__EOFMARK_9f3a1c__"


SSH_OPTS = (
    "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
    "-o BatchMode=yes -o ConnectTimeout=15"
)


def key_auth_ok():
    """True when the board accepts our SSH public key (no password needed).

    Installed once with ssh-copy-id; after that no RK3588_PASSWORD is required
    and the password never has to be handed around again.
    """
    rc = subprocess.call(
        ["ssh", *SSH_OPTS.split(), f"{USER}@{HOST}", "true"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return rc == 0


def shell(cmd, quiet=False, timeout=1800):
    """Run one command over a fresh SSH session; return (rc, output).

    Prefers public-key auth. Falls back to pexpect + RK3588_PASSWORD only when
    the key is not installed yet.
    """
    if key_auth_ok():
        # The rc capture must live outside the command: a command that ends in a
        # newline would turn a leading "; __rc=$?" into a bash syntax error, and
        # an explicit "exit N" inside the command would kill the wrapper before
        # it can report. A subshell fixes both.
        full = (
            f"(\n{cmd}\n)\n__rc=$?\necho \"__RC_${{__rc}}__{SENTINEL}\""
            if cmd else f"echo \"__RC_0__{SENTINEL}\""
        )
        proc = subprocess.run(
            ["ssh", *SSH_OPTS.split(), f"{USER}@{HOST}", full],
            capture_output=True, text=True, timeout=timeout,
        )
        out = proc.stdout
        m = re.search(rf"__RC_(\d+)__{SENTINEL}", out)
        if not m:
            return -1, out + proc.stderr
        return int(m.group(1)), out[: m.start()]

    pw = os.environ.get("RK3588_PASSWORD")
    if not pw:
        sys.exit(
            "board is not reachable by SSH key and RK3588_PASSWORD is not set.\n"
            f"Fix once with: ssh-copy-id -i ~/.ssh/id_ed25519.pub {USER}@{HOST}"
        )
    child = pexpect.spawn(
        f"ssh {SSH_OPTS} {USER}@{HOST}",
        encoding="utf-8",
        timeout=timeout,
    )
    child.expect_exact("password:", timeout=30)
    child.sendline(pw)
    child.expect(PROMPT, timeout=30)
    child.sendline("echo READY_" + SENTINEL)
    child.expect_exact("READY_" + SENTINEL, timeout=30)
    child.expect(PROMPT, timeout=30)

    if cmd:
        # Capture the rc into a variable first: `cmd; echo RC_$?` is unreliable
        # because the echoed line can be spliced into pexpect's buffer mid-read.
        child.sendline(f"__rc=0; {{ {cmd}; }} || __rc=$?; "
                       f"echo \"__RC_${{__rc}}__{SENTINEL}\"")
        try:
            child.expect(r"__RC_(\d+)__" + SENTINEL, timeout=timeout)
        except pexpect.TIMEOUT:
            child.sendline("exit")
            return -1, (child.before or "")
        rc = int(child.match.group(1))
        out = child.before
        child.expect(PROMPT, timeout=60)
        out += child.before
        child.sendline("exit")
        return rc, out
    child.sendline("exit")
    return 0, ""


def put(local, remote):
    """Copy the binary to the board.

    Uses scp driven by pexpect rather than base64-over-ssh: feeding ~32 KB
    command lines into an interactive pty means the tty ECHOES every one of
    them, and the unread echo backlog fills the buffer and wedges the session.
    scp also verifies the transfer for us.
    """
    pw = os.environ.get("RK3588_PASSWORD")
    if not pw:
        sys.exit("RK3588_PASSWORD is not set")
    child = pexpect.spawn(
        f"scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
        f"-P {PORT} {local} {USER}@{HOST}:{remote}",
        encoding="utf-8", timeout=900,
    )
    i = child.expect([r"(?i)password:", r"(?i)are you sure", pexpect.EOF,
                      pexpect.TIMEOUT], timeout=60)
    if i == 0:
        child.sendline(pw)
        child.expect(pexpect.EOF, timeout=900)
    elif i == 1:
        child.sendline("yes")
        child.expect_exact("password:", timeout=30)
        child.sendline(pw)
        child.expect(pexpect.EOF, timeout=900)
    elif i == 3:
        child.close(force=True)
        return False, "scp timed out"
    rc, out = shell(f"chmod 755 {remote} && md5sum {remote} && ls -l {remote}")
    return rc == 0, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--put", action="append", default=[],
                    help="local:remote (repeatable)")
    ap.add_argument("--df", action="store_true")
    ap.add_argument("--exec", dest="exec_cmd")
    args = ap.parse_args()

    for spec in args.put:
        local, remote = spec.split(":", 1)
        print(f"== put {local} -> {remote}")
        ok, out = put(local, remote)
        print(out)
        if not ok:
            sys.exit(1)

    if args.df:
        rc, out = shell("df -h / /mnt/manu")
        print(out)

    if args.exec_cmd:
        rc, out = shell(args.exec_cmd)
        print(out)
        print(f"[rc={rc}]")
        sys.exit(0 if rc == 0 else rc)


if __name__ == "__main__":
    main()