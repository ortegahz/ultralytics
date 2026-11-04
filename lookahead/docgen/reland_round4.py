#!/usr/bin/env python3
"""把 §27 / §30.9.22 / §5.11 三个新章节「回退 -> 重新生成 -> 重新落盘」。

用于生成器内容改动后重放（避免重复追加，也避免旧内容残留）。
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path("/tmp/pycharm_project_10ae9e2e")
MEM = ROOT / "manu/memory"
PY = "/home/huangzhe/anaconda3/envs/uav/bin/python"
GEN = Path(__file__).resolve().parent       # 与本脚本同目录 ⇒ 自包含

TARGETS = [
    (MEM / "lookahead_extrapolator_20261102.md", "## 27. 2026-11-04 服务器夜间第 2 轮"),
    (MEM / "memory.md", "### 30.9.22 ⭐⭐⭐ 2026-11-04 服务器夜间第 2 轮"),
    (ROOT / "manu/tracking/lookahead/README.md", "### 5.11 ★★★ 「不自信就不上报」"),
]


def truncate(path: Path, marker: str) -> int:
    lines = path.read_text(encoding="utf-8").splitlines()
    idx = next((i for i, ln in enumerate(lines) if ln.startswith(marker)), None)
    if idx is None:
        print(f"  [skip] {path.name}: 未找到 {marker!r}")
        return len(lines)
    while idx and not lines[idx - 1].strip():
        idx -= 1
    path.write_text("\n".join(lines[:idx]) + "\n", encoding="utf-8")
    print(f"  [trunc] {path.name}: 回退到第 {idx} 行")
    return idx


print("== 回退 ==")
for path, marker in TARGETS:
    truncate(path, marker)

print("== 重新生成 ==")
for script in ("gen_docs_round4.py", "gen_docs_round4b.py"):
    subprocess.run([PY, str(GEN / script)], check=True)

print("== 重新落盘 ==")
subprocess.run([PY, str(GEN / "land_round4.py")], check=True)
print("done")
