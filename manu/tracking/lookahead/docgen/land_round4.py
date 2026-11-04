#!/usr/bin/env python3
"""Land the round-4 doc sections into the repo and fix the nav-table rows.

Idempotent: refuses to double-append (marker check) and recomputes every
nav row's line number from the *actual* heading position, so nothing is
hand-typed.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path("/tmp/pycharm_project_10ae9e2e")
MEM = ROOT / "manu/memory"
GEN = Path("/mnt/data/codex-overnight")

TOP = MEM / "lookahead_extrapolator_20261102.md"
MAIN = MEM / "memory.md"
README = ROOT / "manu/tracking/lookahead/README.md"
CONTRACT = MEM / "lookahead_board_port_contract_20261103.md"

TOP_TITLE_27 = "专题档 §27 ⭐⭐⭐ 2026-11-04 服务器夜间第 2 轮：误报逐帧归属 + 「不自信就不上报」新机理 + 24 序列 FAR 缺口关闭"
MAIN_TITLE_30922_PREFIX = "主档 §30.9.22 ⭐⭐⭐ 2026-11-04 服务器夜间第 2 轮：「不自信就不上报」—— 24 序列 FAR"


def read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def write(p: Path, text: str) -> None:
    p.write_text(text, encoding="utf-8")


def append_once(path: Path, chunk: str, marker: str) -> bool:
    text = read(path)
    if marker in text:
        print(f"  [skip] {path.name}: 已包含 {marker!r}，不重复追加")
        return False
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text + chunk.rstrip("\n") + "\n", encoding="utf-8")
    print(f"  [ok]   {path.name}: 追加 {len(chunk.splitlines())} 行")
    return True


def heading_line(path: Path, pattern: str) -> int:
    for i, ln in enumerate(read(path).splitlines(), start=1):
        if re.match(pattern, ln):
            return i
    raise SystemExit(f"heading not found: {pattern} in {path}")


def set_row_by_title(path: Path, title: str, file_col: str, target: int) -> None:
    """Rewrite the 起始行 cell of the nav row whose title cell matches exactly."""
    lines = read(path).splitlines()
    hits = []
    for i, ln in enumerate(lines):
        if not ln.startswith("| "):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cells) == 4 and cells[1].startswith(title):
            hits.append(i)
    if len(hits) == 0:
        raise SystemExit(f"{path.name}: 标题 {title!r} 未找到")
    if len(hits) > 1:
        raise SystemExit(f"{path.name}: 标题 {title!r} 匹配到 {len(hits)} 行（应 ≤ 1 行）")
    i = hits[0]
    cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
    assert len(cells) == 4, f"{path.name}: 导航行列数异常 -> {lines[i]}"
    cells[3] = str(target)
    lines[i] = "| " + " | ".join(cells) + " |"
    write(path, "\n".join(lines) + "\n")
    print(f"  [nav]  {path.name}: 行 {i + 1} -> {target}  ({title[:34]}…)")


def add_topical_nav_row(path: Path, title: str, target: int) -> None:
    """Add (or update) the §27 nav row inside the topical doc's own nav table."""
    lines = read(path).splitlines()
    row = f"| 二十七 | {title} | lookahead_extrapolator_20261102.md | {target} |"
    for i, ln in enumerate(lines):
        if ln.startswith("| 二十七 |"):
            lines[i] = row
            write(path, "\n".join(lines) + "\n")
            print(f"  [nav]  {path.name}: §27 行 {i + 1} -> {target}")
            return
    lines += [
        "",
        "### 27.11 导航条目（本节新增）",
        "",
        "| § | 标题（简） | 归档文件 | 起始行 |",
        "|:--|:--|:--|--:|",
        row,
    ]
    write(path, "\n".join(lines) + "\n")
    print(f"  [nav]  {path.name}: 新增 §27 导航行 -> {target}")


def patch_contract(path: Path) -> None:
    """契约档：§4 CLI 真值表 + §7 风险清单 + §9 交付参数原文 同步本轮两个新旋钮。"""
    text = read(path)

    # §4 CLI 真值表：在 uniq-gate 那一行后面补两行
    anchor = "| `--replay-xcorr-uniq-gate` | **`0.0`（关）** | **0.6** |"
    if "`--replay-abstain-peak-below`" not in text:
        if anchor not in text:
            raise SystemExit("契约 §4：找不到 uniq-gate 行，无法插入新旋钮")
        text = text.replace(
            anchor,
            anchor
            + "\n| `--replay-abstain-peak-below` | **`0.0`（关）** | **0.30** | | `--replay-abstain-disp-px` | **`0.0`（关）** | **24** |",
        )
        print("  [ok]   契约 §4：CLI 真值表补两行")

    # §7 风险清单：在 R14 行后面补 R16 / R17
    if "| R16 |" not in text:
        anchor = "| R14 | **拷贝纪律** | 改 `manu/evaluation/*` ⇒ 与原脚本不可比 | 只改 `manu/tracking/lookahead/`；`manu/evaluation/system_sota_longquan.py` md5 `3361e7b70e76d3a92346fd8fd9879dad` 保持不变 |"
        if anchor not in text:
            raise SystemExit("契约 §7：找不到 R14 行，无法插入 R16/R17")
        text = text.replace(
            anchor,
            anchor
            + "\n| R16 | **弃权导致召回下降**（转台长时间收不到坐标） | 阈值太松会把好帧也弃掉 | 交付阈值由 8/24 序列双口径扫描确定（peak < 0.30 且位移 > 24 px）；单测锁住「低峰 + 静止链不弃权」 |"
            + "\n| R17 | **目标持续机动**时链位移一直很大 ⇒ 弃权条件可能长期成立 | 高速机动序列（`001324`）实测低峰帧链位移 p50 = 0 px，未触发；极端机动需复核 | 实跑 hit@5px 未掉（见 §6）；板端把该旋钮做成**可调**并保留关闭位 |",
        )
        print("  [ok]   契约 §7：补 R16 / R17")

    # §9 交付档参数原文
    if "--replay-abstain-peak-below" not in text:
        anchor = "--replay-xcorr-th-k 1.5 --replay-polarity auto --replay-xcorr-uniq-gate 0.6\n```"
        if anchor not in text:
            raise SystemExit("契约 §9：找不到交付档参数原文块")
        text = text.replace(
            anchor,
            "--replay-xcorr-th-k 1.5 --replay-polarity auto --replay-xcorr-uniq-gate 0.6\n"
            "--replay-abstain-peak-below 0.30 --replay-abstain-disp-px 24\n```",
        )
        print("  [ok]   契约 §9：交付档参数原文补两行")

    write(path, text)


def main() -> int:
    topical = read(GEN / "round4_topical.md")
    main_md = read(GEN / "round4_main.md")
    readme_md = read(GEN / "round4_readme.md")
    contract_md = read(GEN / "round4_contract.md")

    print("== 落盘 ==")
    append_once(TOP, topical, "## 27. 2026-11-04 服务器夜间第 2 轮")
    append_once(MAIN, main_md, "### 30.9.22 ⭐⭐⭐ 2026-11-04 服务器夜间第 2 轮")
    append_once(README, readme_md, "### 5.11 ★★★ 「不自信就不上报」")
    append_once(CONTRACT, contract_md, "### 3.7 契约七 · 「不自信就不上报」")

    patch_contract(CONTRACT)

    print("== 导航表 ==")
    top27 = heading_line(TOP, r"^## 27\. ")
    mem30922 = heading_line(MAIN, r"^### 30\.9\.22 ")
    mem_lines = read(MAIN).splitlines()
    row21 = next((l for l in mem_lines if l.startswith("| 三十 |") and "主档 §30.9.21 " in l), None)
    if row21 is None:
        raise SystemExit("找不到 §30.9.21 的导航行，无法推导 memory.md 行偏移")
    global21 = int(row21.strip().strip("|").split("|")[-1].strip())
    mem_offset = global21 - heading_line(MAIN, r"^### 30\.9\.21 ")
    global30922 = mem30922 + mem_offset
    print(f"  专题档 §27={top27} ; memory §30.9.22={mem30922} (offset {mem_offset}) -> global {global30922}")

    add_topical_nav_row(TOP, TOP_TITLE_27, top27)
    set_row_by_title(MAIN, TOP_TITLE_27, "lookahead_extrapolator_20261102.md", top27)
    set_row_by_title(MAIN, MAIN_TITLE_30922_PREFIX, "memory.md", global30922)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
