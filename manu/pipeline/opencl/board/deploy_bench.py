#!/usr/bin/env python3
"""
deploy_bench.py -- ship the benchmark corpus, binary and kernels to the board.

WHY SCRIPT FILES INSTEAD OF INLINE COMMANDS
  Every inline `sh -c` with nested quoting that goes through board_run.py's
  pexpect session risks wedging on the pty echo backlog. Scripts are transferred,
  then invoked by path, so the remote command line is always a single bare token.

WHY scp -r AND NOT THE NFS MOUNT
  The NFS export 192.168.0.64:/mnt/manu is (a) not mounted on this host and (b) the
  board's own rootfs has the "Text file busy" race documented in board_run.py when
  a file is still in flight as the target execs it. scp also verifies the transfer.
"""

import argparse
import os
import sys

import pexpect

HOST = os.environ.get("RK3588_HOST", "192.168.0.64")
PORT = os.environ.get("RK3588_PORT", "22")
USER = os.environ.get("RK3588_USER", "root")
PROMPT = r"[#\$]\s*$"
SENT = "__SENTINEL_7c4e21__"


def shell(cmd, timeout=1800):
    pw = os.environ.get("RK3588_PASSWORD")
    if not pw:
        sys.exit("RK3588_PASSWORD is not set")
    child = pexpect.spawn(
        f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p {PORT} {USER}@{HOST}",
        encoding="utf-8", timeout=timeout)
    child.expect_exact("password:", timeout=30)
    child.sendline(pw)
    child.expect(PROMPT, timeout=30)
    child.sendline(f"echo READY_{SENT}")
    child.expect_exact(f"READY_{SENT}", timeout=30)
    child.expect(PROMPT, timeout=30)
    child.sendline(f"__rc=0; {{ {cmd}; }} || __rc=$?; echo \"__RC_${{__rc}}__{SENT}\"")
    try:
        child.expect(rf"__RC_(\d+)__{SENT}", timeout=timeout)
    except pexpect.TIMEOUT:
        child.sendline("exit")
        return -1, (child.before or "")
    rc = int(child.match.group(1))
    out = child.before
    child.expect(PROMPT, timeout=60)
    out += child.before
    child.sendline("exit")
    return rc, out


def push(local, remote, recursive=False):
    pw = os.environ.get("RK3588_PASSWORD")
    if not pw:
        sys.exit("RK3588_PASSWORD is not set")
    r = " -r" if recursive else ""
    child = pexpect.spawn(
        f"scp{r} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
        f"-P {PORT} {local} {USER}@{HOST}:{remote}",
        encoding="utf-8", timeout=3600)
    i = child.expect([r"(?i)password:", r"(?i)are you sure", pexpect.EOF, pexpect.TIMEOUT],
                     timeout=120)
    if i == 0:
        child.sendline(pw)
        child.expect(pexpect.EOF, timeout=3600)
    elif i == 1:
        child.sendline("yes")
        child.expect_exact("password:", timeout=30)
        child.sendline(pw)
        child.expect(pexpect.EOF, timeout=3600)
    elif i == 3:
        child.close(force=True)
        return False, "scp timed out"
    child.close()
    return child.exitstatus == 0, f"exit={child.exitstatus}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", required=True)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--kernel-dir", required=True)
    ap.add_argument("--remote-root", default="/userdata/prof")
    ap.add_argument("--skip-data", action="store_true")
    args = ap.parse_args()

    rc, out = shell(f"mkdir -p {args.remote_root} {args.remote_root}/kernels && echo OK")
    print(out)
    if rc != 0:
        sys.exit("cannot create remote root")

    print("== push kernels")
    for f in ("warp_median_fused.cl", "sortnet_generated.inc"):
        ok, msg = push(os.path.join(args.kernel_dir, f), f"{args.remote_root}/kernels/{f}")
        print(f"   {f}: {msg}")
        if not ok:
            sys.exit(f"kernel push failed: {f}")

    print("== push binary")
    ok, msg = push(args.binary, f"{args.remote_root}/gmc_stream_ocl")
    print(f"   binary: {msg}")
    if not ok:
        sys.exit("binary push failed")

    if not args.skip_data:
        print("== push benchmark corpus (this is the slow one)")
        ok, msg = push(args.bench_dir, f"{args.remote_root}/bmp", recursive=True)
        print(f"   corpus: {msg}")
        if not ok:
            sys.exit("corpus push failed")

    print("== verify")
    rc, out = shell(
        f"chmod 755 {args.remote_root}/gmc_stream_ocl && "
        f"ls -l {args.remote_root}/gmc_stream_ocl && "
        f"md5sum {args.remote_root}/gmc_stream_ocl && "
        f"ls {args.remote_root}/kernels && "
        f"for d in {args.remote_root}/bmp/*/; do "
        f"echo -n \"$(basename $d) \"; ls $d | wc -l; done")
    print(out)
    sys.exit(0 if rc == 0 else rc)


if __name__ == "__main__":
    main()