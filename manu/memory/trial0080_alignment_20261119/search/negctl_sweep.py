#!/usr/bin/env python3
"""Negative-control sweep for the trial_0080 delivery gate (verify_all.py).

Why this exists
---------------
A gate that only ever prints "[PASS]" proves nothing: it may be checking the wrong
thing, or checking nothing at all (a vacuous criterion).  The only way to tell the
difference is to *break* the thing the gate claims to protect and confirm the gate
actually goes red.

Each case below:
  1. snapshots the target file's bytes,
  2. breaks exactly one thing,
  3. runs the real gate (``manu/memory/delivery_20261119/artifacts/verify_all.py``),
  4. records the rc and the FAIL lines,
  5. restores the file byte-for-byte and re-checks its md5.

A case that produces **no** FAIL is a *vacuous criterion* and is reported as such.

This script deliberately corrupts files inside the delivery package, therefore it
must NOT be shipped in the package -- it lives on the server only (see the
"事故工具（仅 server 留证，未随包交付）" list in the §257 十八 body, which
verify_all.py §6i cross-checks: a script named in the "未随包" column must NOT
exist inside the package).

usage:  python negctl_sweep.py            # run every control
        python negctl_sweep.py --list     # print the case list and exit
exit 0 = every control fired (good);  exit 1 = at least one control was vacuous
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/mnt/data/siping/runs/npu20ms/nano_search/repo")
RUNS = Path("/mnt/data/siping/runs/trial0080_onnx")
GATE = REPO / "manu/memory/delivery_20261119/artifacts/verify_all.py"
PY = os.path.expanduser("~/anaconda3/envs/uav/bin/python")
ENV = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
           OPENBLAS_NUM_THREADS="4", NUMEXPR_NUM_THREADS="4")

README = REPO / "manu/memory/delivery_20261119/artifacts/README.md"
OPTFLAT = REPO / "manu/memory/delivery_20261119/artifacts/opt_params_t0080_v3_flat.json"
UPLOAD = REPO / "manu/videos/upload_to_server.sh"
BODY = RUNS / "search/memory_section_257_18.md"
PARAMS = REPO / "manu/evaluation/system_sota_longquan_trial0080_delivery_v3_optgapfit_params.json"


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def c1():
    """Break the README headline that declares the gate's item count."""
    text = README.read_text(encoding="utf-8")
    assert text.count("（**190 项 PASS / 0 FAIL**") == 1, "README 头条格式已变，先更新本脚本"
    README.write_text(text.replace("（**190 项 PASS / 0 FAIL**", "（**189 项 PASS / 0 FAIL**"),
                      encoding="utf-8")
    return "包内 README 头条 190→189（闸门项数自证）"


def c2():
    """Put a rejected, line-breaking parameter set back into the shipped flat params."""
    doc = json.loads(OPTFLAT.read_text(encoding="utf-8"))
    doc["match_dist"] = 16.0
    OPTFLAT.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    return "包内 opt_params_t0080_v3_flat.json: match_dist 10.0→16.0（把被拒参数塞回包）"


def c3():
    """Drop one line from the package manifest table in the README."""
    lines = README.read_text(encoding="utf-8").splitlines(keepends=True)
    out = [line for line in lines if not line.startswith("| `osd_media_check.json` |")]
    assert len(out) == len(lines) - 1, f"README 清单行形态已变（{len(lines)}→{len(out)}），先更新本脚本"
    README.write_text("".join(out), encoding="utf-8")
    return "包内 README 清单删掉 osd_media_check.json 的登记行"


def c4():
    """Remove one memory exclusion from the upload script."""
    text = UPLOAD.read_text(encoding="utf-8")
    assert text.count("  --exclude=/manu/memory/memory_compact.md\n") == 1, "排除项形态已变，先更新本脚本"
    UPLOAD.write_text(text.replace("  --exclude=/manu/memory/memory_compact.md\n", ""), encoding="utf-8")
    return "upload_to_server.sh: 删掉一条 memory 排除项"


def c5():
    """Change one character of the §257 十八 body heading."""
    text = BODY.read_text(encoding="utf-8")
    assert text.count("### 三十三、") == 1, "章节标题形态已变，先更新本脚本"
    BODY.write_text(text.replace("### 三十三、", "### 三十三x、"), encoding="utf-8")
    return "search/memory_section_257_18.md: 章节标题改 1 个字符"


def c6():
    """Shift the delivered baseline threshold."""
    doc = json.loads(PARAMS.read_text(encoding="utf-8"))
    doc["params"]["th_base"] = 0.47
    PARAMS.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    return "交付参数档: th_base 0.48→0.47"


CASES = [("C1", README, c1), ("C2", OPTFLAT, c2), ("C3", README, c3),
         ("C4", UPLOAD, c4), ("C5", BODY, c5), ("C6", PARAMS, c6)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="只列出用例，不执行")
    args = ap.parse_args()

    if args.list:
        for cid, target, fn in CASES:
            print(f"{cid}  {target}")
        return 0

    for path in (GATE, README, OPTFLAT, UPLOAD, BODY, PARAMS):
        if not path.exists():
            print(f"[FAIL] 缺文件: {path}")
            return 1
    if (REPO / "manu/memory/delivery_20261119/artifacts/negctl_sweep.py").exists():
        print("[FAIL] 本脚本被复制进交付包了 —— 它会故意破坏交付包，必须只留在 server 侧")
        return 1

    results = []
    for cid, target, fn in CASES:
        original = target.read_bytes()
        original_md5 = hashlib.md5(original).hexdigest()
        desc = ""
        try:
            desc = fn()
            started = time.strftime("%H:%M:%S")
            run = subprocess.run([PY, str(GATE)], capture_output=True, text=True, env=ENV, cwd=str(REPO))
            fails = [line.strip() for line in run.stdout.splitlines() if "[FAIL]" in line]
            results.append((cid, desc, run.returncode, len(fails)))
            print(f"\n### {cid} [{started}] {desc}")
            print(f"    rc={run.returncode}  FAIL 数={len(fails)}")
            for line in fails[:6]:
                print("     ", line[:190])
        finally:
            target.write_bytes(original)
            back = hashlib.md5(target.read_bytes()).hexdigest()
            same = back == original_md5
            print(f"    [还原] md5 {original_md5} -> {back}  {'✓ 一致' if same else '✗ 不一致!'}")
            if not same:
                results.append((cid, desc, -1, -1))
                print("    [ABORT] 还原失败，闸门当前状态不可信，停止后续用例")
                return 2

    print("\n=== 汇总 ===")
    vacuous = []
    for cid, desc, rc, n_fail in results:
        ok = n_fail > 0
        print(f"{cid}: rc={rc} FAIL={n_fail}  {'✓ 判据报警' if ok else '✗ 判据空转!'}")
        if not ok:
            vacuous.append(cid)
    print("RESULT:", "ALL CONTROLS FIRED" if not vacuous else f"VACUOUS: {vacuous}")
    return 1 if vacuous else 0


if __name__ == "__main__":
    sys.exit(main())
