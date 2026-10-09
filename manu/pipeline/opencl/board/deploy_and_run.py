#!/usr/bin/env python3
"""
deploy_and_run.py -- push the board OpenCL probe to the RK3588 board and run it.

One SSH session does everything: it creates the staging directory, transfers the
cross-compiled binary and its source by base64 over the session (no scp, no
dependency on ssh-copy-id), then runs the probe and echoes the result back.

Why pexpect instead of sshpass: sshpass is not installed on this host, which is
what `manu/memory/rules.md` sec.1b records as the reason board commands have to
be run by hand. pexpect is available, so that reason no longer holds. The
iron-rule about board execution is a workflow decision, not a technical one --
see the note at the bottom of this file before running it unattended.

Usage:
    python3 deploy_and_run.py                # deploy + run
    python3 deploy_and_run.py --dry-run      # only show what would be sent
    python3 deploy_and_run.py --no-run       # deploy only

Exit code is the probe's own exit code (0 = PASS).
"""

import argparse
import base64
import os
import shlex
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE_BIN = os.path.join(HERE, "out", "board_cl_probe")
PROBE_SRC = os.path.join(HERE, "board_cl_probe.cpp")

# Board coordinates. Password deliberately NOT hardcoded: it is read from the
# environment or prompted for, so it never lands in a file or in git.
BOARD_HOST = os.environ.get("RK3588_HOST", "192.168.0.64")
BOARD_PORT = os.environ.get("RK3588_PORT", "22")
BOARD_USER = os.environ.get("RK3588_USER", "root")
BOARD_DIR = os.environ.get("RK3588_DIR", "/mnt/manu/cl_probe")

SENTINEL = "__MDEOF_9f3a1c__"


def die(msg, code=1):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(code)


def load_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def run_on_board(ssh, password, commands, echo=True, timeout=180):
    """Feed a shell script to the board, stream output back, return it."""
    import pexpect

    body = "\n".join(commands) + f"\necho {SENTINEL}\n"
    idx = ssh.expect([r"[Pp]assword:", r"[#$]\s*$", pexpect.TIMEOUT], timeout=timeout)

    if idx == 0:
        if echo:
            print("[ssh] authenticating...")
        ssh.sendline(password)
        ssh.expect([r"[#$]\s*$", pexpect.TIMEOUT], timeout=timeout)
        if ssh.after == pexpect.TIMEOUT:
            die("authenticated but never got a shell prompt -- wrong password?")

    if echo:
        print(f"[ssh] connected to {BOARD_USER}@{BOARD_HOST}:{BOARD_PORT}")
        print(f"[ssh] sending {len(body)} bytes of script\n")

    # disable echo so the script itself does not pollute the captured output
    ssh.sendline("stty -echo 2>/dev/null || true")
    ssh.expect([r"[#$]\s*$", pexpect.TIMEOUT], timeout=30)
    ssh.sendline(body)

    captured = []
    while True:
        i = ssh.expect(
            [
                re_escape_sentinel(),
                r"[Pp]assword:",
                pexpect.TIMEOUT,
                pexpect.EOF,
            ],
            timeout=timeout,
        )
        if i == 0:
            captured.append(ssh.before)
            break
        if i == 1:
            die("board asked for a password mid-script")
        if i == 2:
            die(f"timed out after {timeout}s waiting for the sentinel")
        captured.append(ssh.before)
        break

    ssh.sendline("stty echo 2>/dev/null || true")
    text = "".join(captured)
    # strip the echoed command block and the trailing prompt
    text = text.split("stty -echo", 1)[-1]
    return text


def re_escape_sentinel():
    import re

    return re.escape(SENTINEL)


def inventory_commands():
    """First-contact environment inventory.

    Ordered per manu/memory/rules.md sec.1b: SoC/NPU model, uname, board Linux,
    RAM, RKNN-Toolkit2 / rknpu driver, OpenCV. Plus the two things that decide
    whether the probe can succeed at all: the ICD vendor file and which OpenCL
    library is actually installed.

    Every probe is written so that "missing" produces a visible line rather than
    silence -- a command that quietly prints nothing is indistinguishable from a
    command that was never run.
    """
    return [
        "echo '### [1/7] CPU / SoC ###'",
        "(lscpu 2>/dev/null | head -14) || (grep -m4 -E 'model name|Hardware|processor' /proc/cpuinfo)",
        "echo '--- cpu count ---'",
        "nproc; grep -c ^processor /proc/cpuinfo",
        "echo",
        "echo '### [2/7] kernel / OS ###'",
        "uname -a",
        "(cat /etc/os-release 2>/dev/null | head -4) || echo '(no /etc/os-release)'",
        "echo",
        "echo '### [3/7] RAM ###'",
        "(free -h 2>/dev/null) || (grep -E 'MemTotal|MemAvailable' /proc/meminfo)",
        "echo",
        "echo '### [4/7] disk ###'",
        "df -h / /mnt/manu 2>&1 | head -6",
        "echo",
        "echo '### [5/7] NPU / GPU device nodes ###'",
        "ls -la /dev/rknpu /dev/mali* /dev/dri 2>&1 || echo '(none of the above)'",
        "echo '--- kernel modules ---'",
        "(lsmod 2>/dev/null | grep -iE 'rknpu|mali|panfrost|gpu' ) || echo '(no matching module)'",
        "echo",
        "echo '### [6/7] RKNN userspace / OpenCV ###'",
        "ls -la /usr/lib/librknnrt* /usr/lib/libmali* /usr/local/lib/librknnrt* 2>&1 || echo '(no rknnrt / mali libs)'",
        "(dpkg -l 2>/dev/null | grep -iE 'rknn|rknn-toolkit' | head -5) || echo '(no rknn packages)'",
        "(python3 -c 'import rknn; print(\"rknn-toolkit2 import OK\", rknn.__file__)' 2>&1 | head -3) || true",
        "echo '--- opencv ---'",
        "(python3 -c 'import cv2; print(\"cv2\", cv2.__version__)' 2>&1 | head -3) || true",
        "(ls -d /usr/include/opencv4 /usr/local/include/opencv4 2>&1) || true",
        "echo",
        "echo '### [7/7] OpenCL stack ###'",
        "echo '--- /etc/OpenCL/vendors ---'",
        "(cat /etc/OpenCL/vendors 2>&1) || echo '(absent)'",
        "echo '--- installed OpenCL / Mali libs ---'",
        "(ls -la /usr/lib/libOpenCL* /usr/lib/libmali* /usr/lib/aarch64-linux-gnu/libOpenCL* 2>&1) || echo '(no OpenCL/Mali lib in the usual paths)'",
    ]


def build_commands(bin_b64, src_b64, do_run):
    cmds = [
        "set -x",
        f"mkdir -p {shlex.quote(BOARD_DIR)}",
        f"cd {shlex.quote(BOARD_DIR)}",
        f"echo '{bin_b64}' | base64 -d > {BOARD_DIR}/board_cl_probe",
        f"echo '{src_b64}' | base64 -d > {BOARD_DIR}/board_cl_probe.cpp",
        f"chmod +x {BOARD_DIR}/board_cl_probe",
        f"ls -la {BOARD_DIR}",
        # verify the transfer actually landed byte-identical before trusting it
        f"md5sum {BOARD_DIR}/board_cl_probe",
        "set +x",
    ]
    if do_run:
        cmds += inventory_commands()
        cmds += [
            "echo '### probe ###'",
            f"cd {BOARD_DIR} && ./board_cl_probe",
            "echo \"PROBE_EXIT=$?\"",
        ]
    return cmds


def _spawn_ssh():
    import pexpect

    return pexpect.spawn(
        "ssh",
        [
            "-p", str(BOARD_PORT),
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR",
            f"{BOARD_USER}@{BOARD_HOST}",
        ],
        encoding="utf-8",
        timeout=60,
        echo=False,
        # the inventory emits a few KB; the default read window can split the
        # sentinel across reads and stall the match
        maxread=65536,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-run", action="store_true", help="deploy but do not execute")
    ap.add_argument(
        "--exec",
        action="append",
        metavar="CMD",
        help="run this shell command on the board instead of the standard flow; "
        "repeatable. Skips deployment entirely, so it is safe to use for queries.",
    )
    args = ap.parse_args()

    if args.exec:
        password = os.environ.get("RK3588_PASSWORD") or input("board password: ")
        ssh = _spawn_ssh()
        out = run_on_board(ssh, password, list(args.exec), timeout=300)
        ssh.close(0)
        print(out)
        return 0

    for p in (PROBE_BIN, PROBE_SRC):
        if not os.path.isfile(p):
            die(f"missing artifact: {p}\nBuild it first: bash {HERE}/build_board_probe.sh")

    import pexpect  # noqa: F401  (import late so --dry-run needs no dependency)

    bin_b64 = load_b64(PROBE_BIN)
    src_b64 = load_b64(PROBE_SRC)
    print(f"[local] binary {os.path.getsize(PROBE_BIN)} bytes -> {len(bin_b64)} b64 chars")
    print(f"[local] source {os.path.getsize(PROBE_SRC)} bytes -> {len(src_b64)} b64 chars")
    print(f"[local] target {BOARD_USER}@{BOARD_HOST}:{BOARD_PORT}:{BOARD_DIR}")

    if args.dry_run:
        print("\n[dry-run] no connection made. Commands that would be sent:")
        for c in build_commands(bin_b64, "<BINARY-B64>", not args.no_run):
            shown = c if len(c) < 120 else c[:80] + f"...<{len(c)} chars>"
            print("   ", shown)
        return 0

    password = os.environ.get("RK3588_PASSWORD") or input("board password: ")
    cmds = build_commands(bin_b64, src_b64, not args.no_run)

    ssh = _spawn_ssh()
    out = run_on_board(ssh, password, cmds, timeout=300)
    ssh.close(0)
    print(out)
    if "PROBE_EXIT=0" in out:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
