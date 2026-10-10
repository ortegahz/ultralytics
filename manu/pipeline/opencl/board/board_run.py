#!/usr/bin/env python3
"""
board_run.py -- transfer a binary to the RK3588 board over one SSH session and
run it with given arguments.

Why not scp/NFS: board executables must NOT go through NFS (the NFS server
still has the write in flight when the board tries to exec -> "Text file busy",
and a fresh filename does not help). So the binary travels inside the SSH
session. pexpect is used because sshpass is not installed on this host.

The password is read from RK3588_PASSWORD (or prompted); it is never written to
any file.

Usage:
    RK3588_PASSWORD=... python3 board_run.py \
        --put out/gmc_stream_ocl_arm64:/mnt/manu/ocvparity_exec/gmc_ocl_pyr \
        --df \
        --exec "/mnt/manu/ocvparity_exec/gmc_ocl_pyr --raw-root /mnt/manu/ocvparity/bmp ..."
"""

import argparse
import base64
import os
import sys

import pexpect

HOST = os.environ.get("RK3588_HOST", "192.168.0.64")
PORT = os.environ.get("RK3588_PORT", "22")
USER = os.environ.get("RK3588_USER", "root")

PROMPT = r"[#\$]\s*$"
SENTINEL = "__EOFMARK_9f3a1c__"


def shell(cmd, quiet=False, timeout=1800):
    """Run one command over a fresh SSH session; return (rc, output)."""
    pw = os.environ.get("RK3588_PASSWORD")
    if not pw:
        sys.exit("RK3588_PASSWORD is not set")
    child = pexpect.spawn(
        f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
        f"-p {PORT} {USER}@{HOST}",
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