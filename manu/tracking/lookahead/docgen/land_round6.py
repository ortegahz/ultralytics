#!/usr/bin/env python3
"""Land the round-6 doc sections into the repo and fix the nav-table rows.

Idempotent (marker check). **顺序很关键**：先追加正文 → 再插导航行 → 最后重算行号
（插行会把后面的标题顶下去，先算后插会差 1 行；§28 落盘时踩过一次）。
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

TOP29 = "## 29. 2026-11-04 服务器夜间第 3 轮（续）"
TOP30 = "## 30. 2026-11-04 服务器夜间第 3 轮（收尾）"
MAIN30 = "### 30.9.24 ⭐⭐⭐ 2026-11-04 服务器夜间第 3 轮（收尾）"
README13 = "### 5.13 ★★★ 最终交付候选"

TOP29_TITLE = "专题档 §29 ⭐⭐⭐ 2026-11-04 服务器夜间第 3 轮（续）：伺服抖动口径审计 + 回退机制帧级归属"
TOP30_TITLE = "专题档 §30 ⭐⭐⭐ 2026-11-04 服务器夜间第 3 轮（收尾）：最终交付候选 = 冠军 + 回退重推 + 峰值门 0.90"
MAIN30_TITLE = "主档 §30.9.24 ⭐⭐⭐ 2026-11-04 服务器夜间第 3 轮（收尾）：最终交付候选 = 冠军 + 回退重推 + 峰值门 0.90"


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


def insert_row_after(path: Path, after_marker: str, row: str, label_prefix: str, title_key: str) -> None:
    """在导航表里插一行（幂等：先删同标题旧行）。"""
    lines = read(path).splitlines()
    lines = [ln for ln in lines if not (ln.startswith(label_prefix) and title_key in ln)]
    for i, ln in enumerate(lines):
        if ln.startswith("| ") and after_marker in ln:
            lines.insert(i + 1, row)
            write(path, "\n".join(lines) + "\n")
            print(f"  [nav]  {path.name}: 插入导航行 -> {row[:60]}")
            return
    raise SystemExit(f"{path.name}: 找不到 {after_marker!r} 那一行")


def set_row(path: Path, title: str, file_col: str, target: int) -> None:
    lines = read(path).splitlines()
    hits = [i for i, ln in enumerate(lines) if ln.startswith("| ") and title[:24] in ln]
    if len(hits) != 1:
        raise SystemExit(f"{path.name}: 标题 {title[:24]!r} 匹配到 {len(hits)} 行")
    i = hits[0]
    cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
    cells[2] = file_col
    cells[3] = str(target)
    lines[i] = "| " + " | ".join(cells) + " |"
    write(path, "\n".join(lines) + "\n")
    print(f"  [nav]  {path.name}: 行 {i + 1} -> {target}")


def main() -> int:
    print("== 追加正文 ==")
    append_once(TOP, read(GEN / "round6_topical.md"), TOP29)
    append_once(TOP, read(GEN / "round6_topical2.md"), TOP30)
    append_once(MAIN, read(GEN / "round6_main.md"), MAIN30)
    append_once(README, read(GEN / "round6_readme.md"), README13)

    print("== 导航行（先插，后算行号）==")
    # 主档：§30.9.24 行插在 §30.9.23 之后；专题档 §29/§30 行插在 §28 行之后。
    insert_row_after(MAIN, "主档 §30.9.23 ", f"| 三十 | {MAIN30_TITLE} | memory.md | 0 |", "| 三十 |", MAIN30_TITLE[:24])
    insert_row_after(MAIN, "专题档 §28 ", f"| 二十九 | {TOP29_TITLE} | lookahead_extrapolator_20261102.md | 0 |", "| 二十九 |", TOP29_TITLE[:24])
    insert_row_after(MAIN, "专题档 §29 ", f"| 三十 | {TOP30_TITLE} | lookahead_extrapolator_20261102.md | 0 |", "| 三十 |", TOP30_TITLE[:24])
    # 专题档自己的导航小节
    for label, title in (("二十九", TOP29_TITLE), ("三十", TOP30_TITLE)):
        _append_topical_nav(label, title)

    print("== 重算行号 ==")
    top29 = heading_line(TOP, r"^## 29\. ")
    top30 = heading_line(TOP, r"^## 30\. ")
    set_row(MAIN, TOP29_TITLE, "lookahead_extrapolator_20261102.md", top29)
    set_row(MAIN, TOP30_TITLE, "lookahead_extrapolator_20261102.md", top30)

    mem30924 = heading_line(MAIN, r"^### 30\.9\.24 ")
    mem_lines = read(MAIN).splitlines()
    row23 = next((l for l in mem_lines if l.startswith("| 三十 |") and "主档 §30.9.23 " in l), None)
    if row23 is None:
        raise SystemExit("找不到 §30.9.23 的导航行")
    global23 = int(row23.strip().strip("|").split("|")[-1].strip())
    mem_offset = global23 - heading_line(MAIN, r"^### 30\.9\.23 ")
    global30924 = mem30924 + mem_offset
    print(f"  专题档 §29={top29} §30={top30} ; memory §30.9.24={mem30924} (offset {mem_offset}) -> global {global30924}")
    set_row(MAIN, MAIN30_TITLE, "memory.md", global30924)

    print("== 专题档内部导航小节回填行号 ==")
    _fill_topical_nav("二十九", top29)
    _fill_topical_nav("三十", top30)

    # ---- 最后一步：导航行号自愈 ----
    # memory.md 的导航行散在各章节末尾的「导航条目」小节里 ⇒ 往 memory.md 插一行
    # （哪怕只是导航区加一行）就会把后面的章节整体顶下去，旧行号全部失配。
    # fix_nav_round6.py 按**真实标题位置**重算行号并让同一文件共享同一 offset（幂等）。
    print("== 导航行号自愈 ==")
    import subprocess

    rc = subprocess.call([sys.executable, str(Path(__file__).with_name("fix_nav_round6.py"))])
    if rc != 0:
        raise SystemExit(f"fix_nav_round6.py 退出码 {rc}")
    print("done")
    return 0


def _append_topical_nav(label: str, title: str) -> None:
    """在专题档末尾加/复用一个导航小节（幂等：块已在就只补行）。"""
    lines = read(TOP).splitlines()
    header = "### 导航条目（本轮新增）"
    row = f"| {label} | {title} | lookahead_extrapolator_20261102.md | 0 |"
    lines = [ln for ln in lines if not (ln.startswith(f"| {label} |") and title[:24] in ln)]
    if header in lines:
        head = lines.index(header)
        for j in range(head, len(lines)):
            if lines[j].startswith("|:--"):
                lines.insert(j + 1, row)
                break
        else:
            lines.append(row)
    else:
        lines += ["", header, "", "| § | 标题（简） | 归档文件 | 起始行 |", "|:--|:--|:--|--:|", row]
    write(TOP, "\n".join(lines) + "\n")
    print(f"  [nav]  专题档: {label} 行")


def _fill_topical_nav(label: str, target: int) -> None:
    lines = read(TOP).splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith(f"| {label} |") and "lookahead_extrapolator_20261102.md" in ln:
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            cells[3] = str(target)
            lines[i] = "| " + " | ".join(cells) + " |"
            write(TOP, "\n".join(lines) + "\n")
            print(f"  [nav]  专题档: {label} -> {target}")
            return


if __name__ == "__main__":
    sys.exit(main())
