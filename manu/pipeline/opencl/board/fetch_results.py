#!/usr/bin/env python3
"""
fetch_results.py -- pull the board's benchmark artefacts back to the x86 host.

scp a DIRECTORY of CSVs and logs. The numbers must be recomputed on the host (see
bench_report.py / accuracy_report.py) so that the delivered tables are produced by
an independent program rather than by the binary under test -- if the board's own
summary and the host recomputation disagree, that disagreement is the finding.
"""
import argparse
import os
import sys

import pexpect

HOST = os.environ.get("RK3588_HOST", "192.168.0.64")
PORT = os.environ.get("RK3588_PORT", "22")
USER = os.environ.get("RK3588_USER", "root")


def pull(remote_dir, local_dir):
    pw = os.environ.get("RK3588_PASSWORD")
    if not pw:
        sys.exit("RK3588_PASSWORD is not set")
    os.makedirs(local_dir, exist_ok=True)
    child = pexpect.spawn(
        f"scp -r -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
        f"-P {PORT} {USER}@{HOST}:{remote_dir}/* {local_dir}/",
        encoding="utf-8", timeout=1800)
    i = child.expect([r"(?i)password:", r"(?i)are you sure", pexpect.EOF, pexpect.TIMEOUT],
                     timeout=120)
    if i == 0:
        child.sendline(pw)
        child.expect(pexpect.EOF, timeout=1800)
    elif i == 1:
        child.sendline("yes")
        child.expect_exact("password:", timeout=30)
        child.sendline(pw)
        child.expect(pexpect.EOF, timeout=1800)
    elif i == 3:
        child.close(force=True)
        return False
    child.close()
    return child.exitstatus == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", required=True)
    ap.add_argument("--local", required=True)
    args = ap.parse_args()
    ok = pull(args.remote, args.local)
    print(f"pull {args.remote} -> {args.local}: {'OK' if ok else 'FAILED'}")
    for f in sorted(os.listdir(args.local)):
        p = os.path.join(args.local, f)
        if os.path.isfile(p):
            print(f"  {f}  {os.path.getsize(p)} B")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()