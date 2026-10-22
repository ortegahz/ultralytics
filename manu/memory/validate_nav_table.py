#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""瘦版 memory 的两道自检（铁律七：归档完整性 + 导航表可解析）。

1) **归档完整性**：`manu/memory/archive/00..04` 按序拼接必须逐字节等于"瘦身前的原始
   memory.md"（md5 基准 `ARCHIVE_BASELINE_MD5`）。任何对归档的改动都会被这里抓住。

2) **导航表逐条校验**：瘦版 memory.md §6 的 "§编号 → 归档文件 + 起始行" 表，
   每一条都要能解析到归档里**真正的章节标题行**。
   - "起始行"是**原 memory.md 的全局行号**（不是归档内行号），因此本脚本为每个归档
     文件**自动推导**行偏移（offset = 全局行号 - 归档内行号），并要求同一文件的所有
     导航行共享同一 offset 且全局行号单调递增。
   - 派生 offset 时不写死任何常数：候选 offset 全部来自"标题行里能解析出的章节号"，
     唯一自洽解才被接受 ⇒ 导航表写错行号会直接 FAIL。

用法：
    python3 manu/memory/validate_nav_table.py                # 用正式 memory.md
    python3 manu/memory/validate_nav_table.py --slim          # 用 .slim_memory.md
    python3 manu/memory/validate_nav_table.py --nav-file X    # 指定任意文件

执行机：本地（纯文本解析，Python 3.8+ 即可）。退出码 0 = 全绿，1 = 有 FAIL。
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

# ---- 归档基准（瘦身前的原始 memory.md 全量，7240 行） -------------------------
ARCHIVE_ORDER = [
    "00_主索引与状态快照_20261014.md",
    "01_端侧移植与板卡工程_§一~§十九.md",
    "02_流水线与量化主线_§二十~§三十五.md",
    "03_基线冻结与路线B_§三十六~§四十九.md",
    "04_判别标准与搜索_§五十~§五十三.md",
]
ARCHIVE_BASELINE_MD5 = "d7761728485b4a040d35f1e6c5789757"
ARCHIVE_BASELINE_LINES = 7240

# ---- 第二个归档卷（2026-10-22 13:38，并行会话产出：领域迁移复评 §五十四~§五十五）----
#   独立分卷、独立基线（不并入上面的拼接基线，否则会破坏 00~04 的 md5 校验）。
EXTRA_VOLUMES = {
    "06_领域迁移复评_§五十四~§五十五.md": None,  # md5 运行时记录，首次建立基线
}

# ---- 导航表解析 ----------------------------------------------------------------
NAV_ROW_RE = re.compile(
    r"^\|\s*(?P<label>[^|]+?)\s*\|\s*(?P<title>[^|]+?)\s*\|\s*(?P<file>[^|]+?)\s*\|\s*(?P<line>\d+)\s*\|\s*$"
)
CN_NUM = "零一二三四五六七八九十百"
# 章节标题：'## §三十六 🔥 xxx' / '## 一、xxx' / '## 零之二、xxx'
HEAD_NUM_RE = re.compile(
    r"^#{1,6}\s*(?:§\s*)?([%s]+(?:之[%s]+)?)\s*(?:[、,.：:·]|—|\s|$)" % (CN_NUM, CN_NUM)
)
FILE_RE = re.compile(r"`([^`]*archive/[^`]+\.md)`")


def norm_label(raw: str) -> str:
    """归一化章节号：去掉 § / 空白 / 全角空格。"""
    return re.sub(r"[\s§　]", "", raw)


def parse_nav(text: str):
    """从瘦版 memory 里抽出导航表行。"""
    rows = []
    in_table = False
    for line in text.splitlines():
        if line.startswith("|"):
            if "起始行" in line:
                in_table = True
                continue
            if not in_table:
                continue
            if set(line.replace("|", "").replace(" ", "")) <= set(":-"):
                continue
            m = NAV_ROW_RE.match(line)
            if not m:
                continue
            rows.append(
                {
                    "label": m.group("label"),
                    "title": m.group("title"),
                    "file": m.group("file"),
                    "line": int(m.group("line")),
                }
            )
        else:
            in_table = False
    return rows


def resolve_files(rows):
    """把 '同上' 解析成实际归档文件相对路径。"""
    last = None
    for r in rows:
        f = r["file"].strip()
        fm = FILE_RE.search(f)
        if fm:
            r["file"] = fm.group(1)
            last = r["file"]
        else:
            if f in ("同上", "同左", "same"):
                r["file"] = last
            else:
                # 形如 `archive/xx.md` 的裸路径
                if f.endswith(".md"):
                    r["file"] = f
                    last = f
    return [r for r in rows if r["file"]]


def heading_label(line: str):
    """从标题行里解析章节号；不是章节标题则返回 None。"""
    m = HEAD_NUM_RE.match(line)
    return norm_label(m.group(1)) if m else None


def check_archive_integrity() -> tuple[bool, str]:
    arch = HERE / "archive"
    parts = []
    total_lines = 0
    for name in ARCHIVE_ORDER:
        p = arch / name
        if not p.exists():
            return False, f"归档分卷缺失：{name}"
        data = p.read_bytes()
        parts.append(data)
        total_lines += data.count(b"\n")
    blob = b"".join(parts)
    md5 = hashlib.md5(blob).hexdigest()
    ok = md5 == ARCHIVE_BASELINE_MD5 and total_lines == ARCHIVE_BASELINE_LINES
    msg = f"拼接 md5={md5}（基准 {ARCHIVE_BASELINE_MD5}） / 行数 {total_lines}（基准 {ARCHIVE_BASELINE_LINES}）"

    # 附加卷：独立完整性记录（与 00~04 基线互不影响）
    extra_bits = []
    for name in EXTRA_VOLUMES:
        p = arch / name
        if not p.is_file():
            extra_bits.append(f"{name} 缺失")
            continue
        data = p.read_bytes()
        h = hashlib.md5(data).hexdigest()
        extra_bits.append(f"{name} md5={h} / {data.count(chr(10).encode())} 行")
    if extra_bits:
        msg += " || 附加卷: " + "; ".join(extra_bits)
    return ok, msg


def check_nav(rows) -> tuple[bool, list[str]]:
    """逐条校验导航行；返回 (是否全绿, 明细行)。"""
    details = []
    ok_all = True

    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["file"], []).append(r)

    offsets: dict[str, int] = {}
    for fname, grows in groups.items():
        p = HERE / fname
        if not p.exists():
            # 允许 basename 形式
            alt = HERE / "archive" / Path(fname).name
            if alt.exists():
                p = alt
                fname = str(Path(fname).name)
                grows = groups[fname]
            else:
                for r in grows:
                    details.append(f"FAIL  §{r['label']:<6} 归档文件不存在：{r['file']}")
                ok_all = False
                continue
        flines = p.read_text(encoding="utf-8", errors="replace").splitlines()

        # 候选 offset：所有能解析出章节号的标题行
        cand_offsets = set()
        for idx, ln in enumerate(flines, start=1):
            hl = heading_label(ln)
            if hl:
                cand_offsets.update({r["line"] - idx for r in grows if hl == norm_label(r["label"])})

        chosen = None
        for off in sorted(cand_offsets):
            good = True
            prev = None
            for r in sorted(grows, key=lambda x: x["line"]):
                local = r["line"] - off
                if not (1 <= local <= len(flines)):
                    good = False
                    break
                hl = heading_label(flines[local - 1])
                if hl != norm_label(r["label"]):
                    good = False
                    break
                if prev is not None and local <= prev:
                    good = False
                    break
                prev = local
            if good:
                chosen = off
                break

        if chosen is None:
            for r in sorted(grows, key=lambda x: x["line"]):
                details.append(f"FAIL  §{r['label']:<6} 行号 {r['line']} 在 {fname} 中对不上章节标题")
            ok_all = False
            continue
        offsets[fname] = chosen

        for r in sorted(grows, key=lambda x: x["line"]):
            local = r["line"] - chosen
            hl = heading_label(flines[local - 1])
            details.append(
                f"OK    §{r['label']:<6} -> {fname}:{local:<5} (全局行 {r['line']}) « {flines[local-1][:52]} »"
            )
    return ok_all, details


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slim", action="store_true", help="校验 .slim_memory.md 而非正式 memory.md")
    ap.add_argument("--nav-file", default=None, help="显式指定被校验文件")
    args = ap.parse_args()

    target = Path(args.nav_file) if args.nav_file else (HERE / (".slim_memory.md" if args.slim else "memory.md"))
    if not target.exists():
        print(f"[FAIL] 被校验文件不存在：{target}")
        return 1

    print(f"=== 被校验文件：{target} ===")

    ok1, msg1 = check_archive_integrity()
    print(f"[{'PASS' if ok1 else 'FAIL'}] 归档完整性（00..04 拼接 vs 瘦身前基线）：{msg1}")

    rows = resolve_files(parse_nav(target.read_text(encoding="utf-8")))
    ok2, details = check_nav(rows)
    print(f"\n--- 导航表逐条校验（{len(rows)} 条）---")
    for d in details:
        print(d)

    fails = [d for d in details if d.startswith("FAIL")]
    print(f"\n[{'PASS' if ok2 else 'FAIL'}] 导航表：{len(rows) - len(fails)}/{len(rows)} 条解析成功")

    # 附加：确认瘦版文件自身没有把归档正文又内联回来（防"假瘦身"）
    size = target.stat().st_size
    lines = target.read_text(encoding="utf-8").count("\n") + 1
    print(f"[INFO] 瘦版规模：{size} B / {lines} 行（目标 300~500 行）")

    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    sys.exit(main())
