#!/usr/bin/env python3
"""Orchestrate the L2/L3 NPU accuracy alignment for the full val run.

L1 (board feature vs frozen feature) and the speed roll-up live in
hm_val_verify.py. This script owns the DETECTION-level reconciliation:

    board NPU detections (all 24 sequences, concatenated)
        vs
    CPU authoritative arm (one global detections.json, frozen features, FP32)

The existing tools (hm_score_board_det.py / hm_demo_report.py) already handle a
multi-sequence ref without modification -- the join key is `Path(name).stem`,
which is globally unique across the 31,613 val frames. What they do NOT do is
FAIL when the board side is short, so that check is enforced here, in code:

  * every sequence's TSV must hold exactly its expected frame count;
  * the concatenation must hold exactly the full-list frame count;
  * stems must be unique;
  * hm_score_board_det.py prints a `missing` line that it does NOT assert on --
    we parse it and fail if it is non-zero.

Rationale for hard assertions: a partial board run still produces a well-formed
report with plausible-looking numbers. Frame accounting is the only cheap way to
tell a real full-val number from a silently truncated one.
"""
import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def read_expected(path):
    exp = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            seq, n = line.split("\t")
            exp[seq] = int(n)
    return exp


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board-root", required=True,
                    help="host-side NFS demo root (contains <seq>/npu_det.tsv)")
    ap.add_argument("--expected", required=True, help="<seq>\\t<count> tsv")
    ap.add_argument("--ref", required=True,
                    help="dir holding the CPU arm's detections.json")
    ap.add_argument("--work", required=True, help="output dir for intermediate + report")
    ap.add_argument("--th", default="0.25")
    ap.add_argument("--dist", type=float, default=8.0)
    ap.add_argument("--gmc-warmup", type=int, default=100)
    args = ap.parse_args()

    os.makedirs(args.work, exist_ok=True)
    exp = read_expected(args.expected)
    total_expected = sum(exp.values())
    print(f"[plan] {len(exp)} 序列, 期望共 {total_expected} 帧")

    # ---- 1. per-sequence frame accounting -----------------------------------
    all_tsv = os.path.join(args.work, "all_det.tsv")
    problems = []
    stems = {}
    written = 0
    with open(all_tsv, "w") as out:
        for seq in sorted(exp):
            p = os.path.join(args.board_root, seq, "npu_det.tsv")
            if not os.path.isfile(p):
                problems.append(f"{seq}: TSV 缺失 {p}")
                continue
            with open(p) as f:
                lines = [ln for ln in f.read().splitlines() if ln.strip()]
            if len(lines) != exp[seq]:
                problems.append(f"{seq}: TSV {len(lines)} 行 != 期望 {exp[seq]}")
            for ln in lines:
                stem = ln.split("\t", 1)[0]
                stem = stem.rsplit(".", 1)[0]
                stems[stem] = stems.get(stem, 0) + 1
                out.write(ln + "\n")
                written += 1

    dup = [s for s, c in stems.items() if c > 1]
    if dup:
        problems.append(f"stem 重复 {len(dup)} 个, 例: {dup[:5]}")
    print(f"[acc] 拼接 {written} 行, 唯一 stem {len(stems)} 个")

    if problems:
        print("[FAIL] 帧账目不平:", file=sys.stderr)
        for p in problems:
            print("  " + p, file=sys.stderr)
        return 1
    if written != total_expected:
        print(f"[FAIL] 合计 {written} != 期望 {total_expected}", file=sys.stderr)
        return 1

    # ---- 2. detection-level reconciliation ----------------------------------
    py = os.environ.get("PY", "python3")
    perframe = os.path.join(args.work, "board_perframe_all.json")
    score_log = os.path.join(args.work, "score.log")
    cmd = [py, os.path.join(HERE, "hm_score_board_det.py"),
           "--tsv", all_tsv, "--ref", args.ref,
           "--dist", str(args.dist), "--out", perframe]
    print("[run]", " ".join(cmd))
    with open(score_log, "w") as lf:
        rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT)
    if rc != 0:
        print(f"[FAIL] hm_score_board_det rc={rc}, 见 {score_log}", file=sys.stderr)
        return 1

    # hm_score_board_det prints `frames scored : N   (no board line for M)` and
    # never asserts on M. Match the ACTUAL wording: the token "missing" does not
    # appear in its output, so a regex looking for it silently finds nothing and
    # the check below would degrade into a no-op that always passes.
    miss = None
    scored = None
    with open(score_log) as lf:
        for ln in lf:
            m = re.search(r"frames scored\s*:\s*(\d+)", ln)
            if m:
                scored = int(m.group(1))
            m = re.search(r"no board line for\s*(\d+)", ln)
            if m:
                miss = int(m.group(1))
    if scored is None or miss is None:
        print(f"[FAIL] 未能从 {score_log} 解析 'frames scored' / 'no board line for'，"
              f"拒绝在无法核对的情况下继续", file=sys.stderr)
        return 1

    # "missing" counts ref records the board did NOT produce, so on a SUBSET run
    # it is non-zero BY DESIGN. The exact identity is
    #     missing == |ref stems| - |board stems|
    # which holds for the full val and for any subset, and is far stronger than
    # asserting missing == 0 (that would silently forbid subset analysis).
    ref_path = os.path.join(args.ref, "detections.json")
    with open(ref_path) as lf:
        ref_stems = {os.path.basename(r["name"]).rsplit(".", 1)[0]
                     for r in json.load(lf)}
    expect_missing = len(ref_stems) - len(stems)
    print(f"[acc] 逐帧计分 {scored}, missing={miss} "
          f"(ref {len(ref_stems)} 帧 - board {len(stems)} 帧 = {expect_missing})")
    if scored != written:
        print(f"[FAIL] 计分 {scored} != 板端 TSV {written} 行（板端有帧没进对账）",
              file=sys.stderr)
        return 1
    if miss != expect_missing:
        print(f"[FAIL] missing={miss} != 期望 {expect_missing}"
              f"（{len(ref_stems)} ref 帧 - {len(stems)} 板端帧）⇒ "
              f"板端有 stem 不在 ref 里，或 ref 有帧对不上", file=sys.stderr)
        return 1
    orphan = set(stems) - ref_stems
    if orphan:
        print(f"[FAIL] 板端有 {len(orphan)} 个 stem 不在 CPU 参考里，例: {list(orphan)[:3]}",
              file=sys.stderr)
        return 1

    # ---- 3. L2 / L3 roll-up -------------------------------------------------
    # --sequence is required by the CLI but is only used for the L1 title and
    # the [SEQ] log regex; L2/L3 iterate every record, so a placeholder is safe.
    report = os.path.join(args.work, "report_L2L3.txt")
    cmd = [py, os.path.join(HERE, "hm_demo_report.py"),
           "--sequence", "ALL",
           "--tsv", all_tsv, "--ref", args.ref, "--perframe", perframe,
           "--gmc-warmup", str(args.gmc_warmup)]
    print("[run]", " ".join(cmd))
    with open(report, "w") as lf:
        rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT)
    if rc != 0:
        print(f"[FAIL] hm_demo_report rc={rc}, 见 {report}", file=sys.stderr)
        return 1

    # layer2_det() sums the CPU arm over EVERY record in detections.json but the
    # board arm only over the perframe rows -- so a partial board run yields two
    # rows with DIFFERENT GT denominators that still look comparable. Enforce that
    # both arms cover the same frame set, or the precision/recall pair is meaningless.
    gts = []
    with open(report) as lf:
        for ln in lf:
            if ln.startswith("| CPU") or ln.startswith("| 板端"):
                cols = [c.strip() for c in ln.strip().strip("|").split("|")]
                if len(cols) >= 4:
                    gts.append((cols[0], cols[1]))
    if len(gts) == 2:
        print(f"[acc] 两臂 GT: {gts[0][0]}={gts[0][1]} / {gts[1][0]}={gts[1][1]}")
        if gts[0][1] != gts[1][1]:
            print(f"[FAIL] 两臂 GT 不等（{gts[0][1]} vs {gts[1][1]}）=> 覆盖帧集不同，"
                  f"L2 对比无效；通常是板端未跑完", file=sys.stderr)
            return 1
    else:
        print(f"[FAIL] 未能从报告解析出两臂 L2 行（找到 {len(gts)} 行），拒绝在无法核对时继续",
              file=sys.stderr)
        return 1

    print(f"\n[done] 报告: {report}")
    print(f"[done] 逐帧: {perframe}")
    print(f"[done] 拼接 TSV: {all_tsv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())