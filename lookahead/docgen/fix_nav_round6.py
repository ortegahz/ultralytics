#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""导航表行号自愈（只修 memory.md 自己的行）。

背景（2026-11-04 夜间第 3 轮踩到的坑）：瘦版 memory.md 的「§编号 → 归档文件 + 起始行」
导航行**散布在各章节末尾的「导航条目」小节里**（不是一张连续的表）。
往 memory.md 里插入任何一行（哪怕只是在导航区加一行），都会把**后面**的章节整体顶下去
⇒ 旧行号失配 ⇒ `validate_nav_table.py` 直接 FAIL（校验器要求同一文件的所有行共享
**同一个** offset）。

修法（只动 memory.md 的行，别人的归档文件一律不碰）：
  ① 按「章节标题标签 + 出现顺序」把每条 memory.md 导航行解析到**真实标题行**（local）；
  ② offset = round(median(旧全局行 − local))（保持历史口径不漂移，-410）；
  ③ 每条重写为 `全局行 = local + offset`（校验器口径：local = global − offset）
     ⇒ 同一文件共享同一 offset，且历史数值只平移不乱跳。

用法：python3 /mnt/data/codex-overnight/fix_nav_round6.py [--dry]
"""
from __future__ import annotations

import importlib.util
import statistics
import sys
from pathlib import Path

ROOT = Path("/tmp/pycharm_project_10ae9e2e")
MEM = ROOT / "manu/memory"

spec = importlib.util.spec_from_file_location("v", MEM / "validate_nav_table.py")
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)

DRY = "--dry" in sys.argv
MAIN = MEM / "memory.md"


def main() -> int:
    text = MAIN.read_text(encoding="utf-8")
    flines = text.splitlines()
    heads = [(i, ln) for i, ln in enumerate(flines, start=1) if v.heading_label(ln)]
    rows = sorted([r for r in v.parse_nav(text) if r["file"] == "memory.md"], key=lambda x: x["line"])

    import re

    SEC_RE = re.compile(r"§\s*([0-9]+(?:\.[0-9]+)*)")

    def key_of(title: str) -> str | None:
        m = SEC_RE.search(title)
        return m.group(1) if m else None

    # 先用「标题里的章节号」精确匹配一轮（能定死的就定死），拿到 offset 的中位数；
    # 剩下的（中文数字标题）在「期望位置 ± 窗口」里按标签就近匹配。
    off_guess = -410
    picked: dict[int, int] = {}
    for r in rows:
        key = key_of(r["title"])
        if not key:
            continue
        pat = re.compile(r"(?:^|[^\d.])" + re.escape(key) + r"(?:[^\d]|$)")
        cands = [i for i, ln in heads if pat.search(ln)]
        if len(cands) == 1:
            picked[id(r)] = cands[0]
    if picked:
        off_guess = round(statistics.median([r["line"] - picked[id(r)] for r in rows if id(r) in picked]))

    resolved, prev = [], 0
    for r in rows:
        want = v.norm_label(r["label"])
        expected = r["line"] - off_guess
        cands = [i for i, ln in heads if i > prev and v.heading_label(ln) == want]
        if not cands:
            raise SystemExit(f"memory.md: 找不到 §{r['label']} 的标题行（{r['title'][:50]}）")
        # 就近取：插入只会让标题整体平移几行，不该跨几百行去认别的章节
        local = min(cands, key=lambda i: (abs(i - expected), i))
        if abs(local - expected) > 12:
            raise SystemExit(
                f"memory.md: §{r['label']} 的标题行绑定不可信（期望 ~{expected}，最近 {local}）"
            )
        resolved.append((r, local))
        prev = local

    off = round(statistics.median([r["line"] - local for r, local in resolved]))
    print(f"[OFFSET] memory.md offset = {off}")

    lines = text.splitlines()
    changes = []
    for r, local in resolved:
        new = local + off  # 校验器口径：local = global - off ⇒ global = local + off
        if new != r["line"]:
            changes.append((r["title"], r["line"], new))
        for i, ln in enumerate(lines):
            if ln.startswith("| ") and r["title"] in ln and ln.strip().strip("|").split("|")[2].strip() == "memory.md":
                cells = [c.strip() for c in ln.strip().strip("|").split("|")]
                cells[3] = str(new)
                lines[i] = "| " + " | ".join(cells) + " |"
                break
        else:
            raise SystemExit(f"memory.md: 导航行没找到（{r['title'][:50]}）")

    for title, old, new in changes:
        print(f"  {title[:52]:54s} {old:>6} -> {new:>6}")
    print(f"共 {len(changes)} 行行号更新")
    if DRY:
        print("[DRY] 未写盘")
        return 0
    MAIN.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[WRITE] manu/memory/memory.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
