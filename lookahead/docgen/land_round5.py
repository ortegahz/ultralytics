#!/usr/bin/env python3
"""Land the round-5 doc sections into the repo and fix the nav-table rows.

Idempotent (marker check) and **recomputes** every nav row's line number from the
actual heading position, so nothing is hand-typed.
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

TOP_MARKER = "## 28. 2026-11-04 服务器夜间第 3 轮"
MAIN_MARKER = "### 30.9.23 ⭐⭐⭐ 2026-11-04 服务器夜间第 3 轮"
README_MARKER = "### 5.12 ★★★ 帧级精度损失归属"

TOP_TITLE = "专题档 §28 ⭐⭐⭐ 2026-11-04 服务器夜间第 3 轮：帧级精度损失归属 + 锚点 oracle 天花板 + 「源帧无检测 ⇒ 再往回退重推」"
MAIN_TITLE = "主档 §30.9.23 ⭐⭐⭐ 2026-11-04 服务器夜间第 3 轮：帧级精度损失归属 + 锚点 oracle 天花板 + 「源帧无检测 ⇒ 再往回退重推」"


def read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def write(p: Path, text: str) -> None:
    p.write_text(text, encoding="utf-8")


def append_once(path: Path, chunk: str, marker: str) -> bool:
    text = read(path)
    if marker in text:
        print(f"  [skip] {path.name}: 已包含 {marker!r}")
        return False
    if not text.endswith("\n"):
        text += "\n"
    write(path, text + chunk.rstrip("\n") + "\n")
    print(f"  [ok]   {path.name}: 追加 {len(chunk.splitlines())} 行")
    return True


def heading_line(path: Path, pattern: str) -> int:
    for i, ln in enumerate(read(path).splitlines(), start=1):
        if re.match(pattern, ln):
            return i
    raise SystemExit(f"heading not found: {pattern} in {path}")


def set_row_by_title(path: Path, title: str, file_col: str, target: int) -> None:
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
        raise SystemExit(f"{path.name}: 标题 {title!r} 匹配到 {len(hits)} 行")
    i = hits[0]
    cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
    cells[2] = file_col
    cells[3] = str(target)
    lines[i] = "| " + " | ".join(cells) + " |"
    write(path, "\n".join(lines) + "\n")
    print(f"  [nav]  {path.name}: 行 {i + 1} -> {target}")


def add_main_nav_row(path: Path, title: str, target: int, after_title: str) -> None:
    """Insert a nav row for the new §30.9.23 right after the §30.9.22 row.

    The label must stay "三十" — the validator normalises "三十一" to 31 while the
    heading "### 30.9.23 …" normalises to 30, so a distinct label would never match.
    """
    lines = read(path).splitlines()
    # 幂等：先删掉同标题的旧行，再插一行新的（重跑生成器不会堆重复行）
    kept = [ln for ln in lines if not (ln.startswith("| 三十 |") and title[:24] in ln)]
    row = f"| 三十 | {title} | memory.md | {target} |"
    for i, ln in enumerate(kept):
        if ln.startswith("| ") and after_title in ln:
            kept.insert(i + 1, row)
            write(path, "\n".join(kept) + "\n")
            print(f"  [nav]  {path.name}: §30.9.23 导航行 -> {target}")
            return
    raise SystemExit(f"{path.name}: 找不到 {after_title!r} 那一行，无法插入新导航行")


def add_topical_nav_row(path: Path, title: str, target: int) -> None:
    lines = read(path).splitlines()
    row = f"| 二十八 | {title} | lookahead_extrapolator_20261102.md | {target} |"
    lines = [ln for ln in lines if not (ln.startswith("| 二十八 |") and title[:24] in ln)]
    for i, ln in enumerate(lines):
        if ln.startswith("| 二十八 |"):
            lines[i] = row
            write(path, "\n".join(lines) + "\n")
            print(f"  [nav]  {path.name}: §28 行 {i + 1} -> {target}")
            return
    # 已有 §28.10 小节就复用它（只补行），不要再堆一个新块 —— 重跑生成器曾把
    # 这个空块堆到 3 份。
    header = "### 28.10 导航条目（本节新增）"
    if header in lines:
        # 插回表头下方（而不是文件末尾），保持块内结构完整。
        head = lines.index(header)
        for j in range(head, len(lines)):
            if lines[j].startswith("|:--"):
                lines.insert(j + 1, row)
                break
        else:
            lines.append(row)
    else:
        lines += [
            "",
            header,
            "",
            "| § | 标题（简） | 归档文件 | 起始行 |",
            "|:--|:--|:--|--:|",
            row,
        ]
    write(path, "\n".join(lines) + "\n")
    print(f"  [nav]  {path.name}: 新增 §28 导航行 -> {target}")


def main() -> int:
    print("== 落盘 ==")
    append_once(TOP, read(GEN / "round5_topical.md"), TOP_MARKER)
    append_once(MAIN, read(GEN / "round5_main.md"), MAIN_MARKER)
    append_once(README, read(GEN / "round5_readme.md"), README_MARKER)

    print("== 导航表 ==")
    # 先把新行**插进去**（占位行号），再重算标题行号 —— 插行会把后面的标题顶下去，
    # 先算后插会差 1 行（踩过一次）。
    add_main_nav_row(MAIN, MAIN_TITLE, 0, "主档 §30.9.22 ")
    add_topical_nav_row(TOP, TOP_TITLE, 0)

    top28 = heading_line(TOP, r"^## 28\. ")
    mem30923 = heading_line(MAIN, r"^### 30\.9\.23 ")
    mem_lines = read(MAIN).splitlines()
    row22 = next((l for l in mem_lines if l.startswith("| 三十 |") and "主档 §30.9.22 " in l), None)
    if row22 is None:
        raise SystemExit("找不到 §30.9.22 的导航行，无法推导 memory.md 行偏移")
    global22 = int(row22.strip().strip("|").split("|")[-1].strip())
    mem_offset = global22 - heading_line(MAIN, r"^### 30\.9\.22 ")
    global30923 = mem30923 + mem_offset
    print(f"  专题档 §28={top28} ; memory §30.9.23={mem30923} (offset {mem_offset}) -> global {global30923}")

    set_row_by_title(MAIN, TOP_TITLE, "lookahead_extrapolator_20261102.md", top28)
    set_row_by_title(MAIN, MAIN_TITLE, "memory.md", global30923)
    add_topical_nav_row(TOP, TOP_TITLE, top28)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
