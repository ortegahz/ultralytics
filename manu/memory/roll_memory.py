#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""记忆主档滚动分卷 —— 把 `memory.md` 的正文按 § 边界切进 `archive/`，`memory.md` 只留「索引头 + 活章」。

## 为什么

`memory.md` 一度长到 **2.9 万行 / 2.6 MB**，单文件既装不下上下文、也没法按主题定位。
本脚本把它拆成：

```
manu/memory/
├── memory.md        索引头（状态快照 / 擂主 / 环境 / 最新结论 / 全量导航表）+ 活章（新 § 追加点）
└── archive/         正文分卷：08_主档正文_§011-§086_….md …（只增不改）
```

## 硬约束（全部有自动校验兜底）

1. **切分点只落在 `## ` 章节标题前** —— 绝不把一节劈成两半。
2. **零字节丢失**：`archive/_roll_manifest.json` 记录原 `memory.md` 的 md5 与逐卷 md5；
   `check_memory_roll.py` 重算拼接 md5 与原文比对。
3. **导航不断裂**：原本指向 `memory.md` 全局行号的导航行，改写为 `archive/NN.md` + 卷内行号。
   校验器**不做行号偏移换算**：它只做 md5 全等比对（闸门 1/2/3）与「去掉导航行后的正文指纹」
   比对（闸门 5）⇒ 导航行被改写**不会**掩盖正文的任何丢失（导航行本来就整行排除在指纹外）。
4. **主导航表全量重建**：`memory.md` §6 变成「每一节一行」的全量索引，不再需要先猜在哪个卷。
5. **幂等**：已归档的节不在 `memory.md` 里 ⇒ 重复运行不会重复切分。
   ⇒ 零丢失基准必须覆盖 manifest 里的**全部**卷（不是本次新卷），否则二次运行基准会退化成空串。

## manifest 记什么（`archive/_roll_manifest.json`）

| 字段 | 含义 |
|:--|:--|
| `origin` | **pre-split 单体源锚点**（首次切分时算一次，之后只继承不重算）：源 md5/行数、正文起始全局行、**去掉导航行后**的正文 md5/行数 ⇒ 证明「除导航行外一字未丢」 |
| `origin.origin_head_prefix_*` / `origin.origin_head_suffix_*` | **索引头锚点**：NAV 块外两段的 md5 + 行数 |
| `archived_body_md5` / `archived_body_lines` | 全部卷正文拼接（卷间补 `\n`）的指纹 ⇒ 少一卷/少一行都现形 |
| `live_md5` / `live_lines` / `head_md5` | 当前 `memory.md` 活区与索引头的指纹 |
| `volumes[].sections[]` | 每节的**原始**全局行号 `global_start` + 卷内行号 `local_line` |
| `live_sections[]` | 活章的**原始**全局行号 `global_start` + 当前 `memory.md` 行号 `local_line` |
| `head_map.segments[]` | 索引头原始行号 → 当前行号（等长 1:1 段），供旧文档里的 `memory.md:20` 这类引用换算 |
| `nav_rows_remapped` | **累计语义**：历次运行累计改写过的导航行条数（逐次累加）⇒ 二次运行不会把首次切分的记录冲成 0 |

⇒ 坐标一律以 **pre-split 源**（切分前那份 29,344 行单体）为基准；重跑不得让它们平移。

索引头锚点（`origin.origin_head_prefix_*` / `origin.origin_head_suffix_*`）= 索引头里
`NAV AUTO-GENERATED` 块**之外**两段的 md5 + 行数：`origin` 的其它字段只冻结**正文**，
查不出「索引头少了/多了一行」⇒ 这两段单独做锚点，由 `check_memory_roll.py` 的闸门 6 盯着。

## 用法

```bash
python3 manu/memory/roll_memory.py --dry      # 只打印计划，不写盘
python3 manu/memory/roll_memory.py            # 执行切分
python3 manu/memory/roll_memory.py --check    # 只跑零丢失校验
python3 manu/memory/roll_memory.py --refresh-head-anchor   # 有意改过索引头后显式重锚
```

执行机：本地（纯文本处理，Python 3.8+）。退出码 0 = 绿，1 = 有问题。
"""

from __future__ import annotations

import argparse
import datetime
import difflib
import hashlib
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "archive"
MAIN = HERE / "memory.md"
MANIFEST = ARCHIVE / "_roll_manifest.json"

NAV_AUTO_START = "<!-- ===== NAV AUTO-GENERATED START ===== -->"
NAV_AUTO_END = "<!-- ===== NAV AUTO-GENERATED END ===== -->"
BODY_START = "<!-- ===== MEMORY BODY START ===== -->"
BODY_END = "<!-- ===== MEMORY BODY END ===== -->"

VOLUME_TARGET_LINES = 2500   # 单卷目标行数
LIVE_MAX_LINES = 1500        # memory.md 里保留的「活章」行数（最新若干节）
# 最早可用卷号（下限）。新卷号 = max(本下限, 盘上/manifest 现有最大卷号 + 1)，
# **不写死** —— 写死会在「冻结卷 00~07 + 主档卷 08~19」之后产出与既有卷**同号**的文件。
MIN_VOLUME_NO = 7
# pre-split 单体源副本（一次性播种索引头锚点用；缺失时退回「当前头」口径并告警）
MONOLITH_COPY = Path(os.environ.get("MEMORY_MONOLITH_COPY", "/tmp/memory_backup_20261124.md"))

# 索引头锚点的字段名（存在 manifest 的 `origin` 块里）
HEAD_ANCHOR_KEYS = (
    "origin_head_prefix_md5", "origin_head_prefix_lines",
    "origin_head_suffix_md5", "origin_head_suffix_lines",
)

CN_NUM = "零一二三四五六七八九十百"
NAV_ROW_RE = re.compile(
    r"^\|\s*(?P<label>[^|]+?)\s*\|\s*(?P<title>[^|]+?)\s*\|\s*(?P<file>[^|]+?)\s*\|\s*(?P<line>\d+)\s*\|\s*$"
)
DATE_RE = re.compile(r"(20\d{2})-(\d{2})-(\d{2})")
HEAD_RE = re.compile(
    r"^#{1,6}\s*(?:§\s*)?([%s0-9]+(?:之[%s0-9]+)?)\s*(?:[、,.：:·（）()\[\]【】—–\-|/]|\s|$)"
    % (CN_NUM, CN_NUM)
)

_CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000}


# ---------------------------------------------------------------- 章节号工具
def cn_to_int(text: str):
    total = section = number = 0
    seen = False
    for ch in text:
        if ch in _CN_DIGITS:
            number = _CN_DIGITS[ch]
            seen = True
        elif ch in _CN_UNITS:
            unit = _CN_UNITS[ch]
            if number == 0:
                number = 1
            section += number * unit
            number = 0
            seen = True
        else:
            return None
    return (total + section + number) if seen else None


def norm_label(raw: str):
    s = re.sub(r"[\s§　*`]", "", raw or "")
    m = re.match(r"^([0-9]+)", s)
    if m:
        return str(int(m.group(1)))
    n = cn_to_int(s)
    return str(n) if n is not None else s


def heading_label(line: str):
    m = HEAD_RE.match(line)
    return norm_label(m.group(1)) if m else None


def sec_label(heading: str):
    return heading_label(heading) or ""


def sec_title(heading: str, limit: int = 76):
    t = heading.lstrip("#").strip()
    t = re.sub(r"^[§\s]+", "", t)
    t = t.strip("*")
    return t if len(t) <= limit else t[: limit - 1] + "…"


def md5_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def nav_free_lines(lines):
    """去掉导航行：切分时导航行的 file/line 列被改写过，不能参与正文指纹比对。"""
    return [ln for ln in lines if not NAV_ROW_RE.match(ln)]


def section_no(label):
    """§ 标签 → 整数（拿不到就 None；标签可能是「一百二十三」这类中文编号）。"""
    try:
        return int(label)
    except (TypeError, ValueError):
        return None


def existing_volume_numbers(manifest):
    """已存在的全部卷号 = manifest 的 `volumes[].vol` ∪ `archive/NN_*.md` 的 NN。

    ⚠ 必须同时看盘上和 manifest：冻结卷 00~07 只在盘上、manifest 里没有。
    """
    nums = set()
    for m in (manifest or {}).get("volumes", []):
        n = section_no(str(m.get("vol", "")))
        if n is not None:
            nums.add(n)
    if ARCHIVE.is_dir():
        for p in ARCHIVE.glob("[0-9][0-9]_*.md"):
            n = section_no(p.name[:2])
            if n is not None:
                nums.add(n)
    return nums


def next_volume_no(manifest):
    """新卷起始号 = max(最早可用卷号, 现有最大卷号 + 1)。

    ⚠ 原来这里是常量 8 ⇒ 冻结卷 00~07 之后又切出主档卷 08~19，再切一次就会
    产出**同号**的 `08_主档正文_…md`（manifest 里出现两个 `vol: "08"`）。
    """
    nums = existing_volume_numbers(manifest)
    newest = max(nums) if nums else MIN_VOLUME_NO - 1
    return max(MIN_VOLUME_NO, newest + 1)


def last_archived_section_no(vols):
    """已归档区里最大的 § 编号（用来判定当前正文是否与归档区重叠）。"""
    best = None
    for v in vols or []:
        for s in v.get("sections", []):
            n = section_no(s.get("label"))
            if n is not None and (best is None or n > best):
                best = n
    return best


def origin_anchor(source_text, source_lines, body_start_global, body_lines):
    """pre-split 单体源的锚点：源身份 + 去掉导航行后的正文指纹。

    锚点只在**首次切分**（源还是单体 monolith）时现算一次，之后**只继承不重算**：
    重算会拿切分后的产物反推自己，零丢失校验就成了自证。
    """
    norm = nav_free_lines(body_lines)
    return {
        "origin_source_md5": md5_text(source_text),
        # 原件行数（等价 wc -l：末尾无换行时多算一行）
        "origin_source_lines": source_text.count("\n") + (0 if source_text.endswith("\n") else 1),
        "origin_body_start_global": body_start_global,
        "origin_body_norm_md5": md5_text("\n".join(norm)),
        "origin_body_norm_lines": len(norm),
    }


def head_anchor_fields(head_lines):
    """索引头锚点：NAV AUTO-GENERATED 块**之外**两段（prefix / suffix）的 md5 + 行数。

    切分只重写 NAV 自动块 ⇒ 块外的 prefix / suffix 逐字节稳定 ⇒ 可以当独立锚点。
    `origin` 的其它字段只冻结**正文** ⇒ 索引头少拼一行不会被发现 ⇒ 这里补一道。
    切分后新头 = prefix + [NAV_AUTO_START] + 自动块 + [NAV_AUTO_END] + suffix，
    与单体源的 head 位置一一对应 ⇒ 两段可以逐字节比对。
    """
    start, end, _rows = find_nav_block(head_lines)
    prefix, suffix = head_lines[:start], head_lines[end:]
    return {
        "origin_head_prefix_md5": md5_text("\n".join(prefix)),
        "origin_head_prefix_lines": len(prefix),
        "origin_head_suffix_md5": md5_text("\n".join(suffix)),
        "origin_head_suffix_lines": len(suffix),
    }


def has_head_anchor(origin):
    """origin 块里是否已带索引头锚点。"""
    return bool(origin) and all(k in origin for k in HEAD_ANCHOR_KEYS)


def monolith_head_anchor(path=None):
    """从 pre-split 单体源现算索引头锚点（权威口径：与切分后的头一一对应）。"""
    p = Path(path or MONOLITH_COPY)
    if not p.is_file():
        return None
    try:
        lines = p.read_text(encoding="utf-8").split("\n")
        head_lines, _body, _tail, _bs = locate_body(lines)
        return head_anchor_fields(head_lines)
    except SystemExit:
        return None


def seed_head_anchor(origin, ctx):
    """一次性补齐 origin 块里的索引头锚点（历史遗留：旧脚本从没写过）。

    优先用 pre-split 单体源现算（权威口径，不受切分之后的改动影响）；
    单体源不可用时退回「当前 memory.md 的索引头」并**醒目告警**
    （此时锚点是自证的：只能挡住播种之后的改动，挡不住播种之前就已发生的改动）。
    """
    anchor = monolith_head_anchor()
    if anchor is None:
        anchor = head_anchor_fields(ctx["head_lines"])
        origin.update(anchor)
        print("[SEED][WARN] ⚠ pre-split 单体源不可用（%s），索引头锚点改由**当前** memory.md 现算 —— "
              "这是自证口径，挡不住播种之前就已发生的索引头改动" % MONOLITH_COPY)
        return anchor
    origin.update(anchor)
    print("[SEED] 索引头锚点已从 pre-split 单体源播种：prefix %d 行 / suffix %d 行" % (
        anchor["origin_head_prefix_lines"], anchor["origin_head_suffix_lines"]))
    return anchor


def reanchor_head(origin, ctx):
    """显式重锚：按**当前** memory.md 的索引头重算 prefix / suffix。

    只在用户**有意**改过索引头内容（例如订正一个过期的行数）时用；
    默认不重算（与 origin 的其它字段一致）。
    """
    anchor = head_anchor_fields(ctx["head_lines"])
    origin.update(anchor)
    print("[RE-ANCHOR] ⚠ 索引头锚点已按**当前** memory.md 重算：prefix %d 行 (%s) / suffix %d 行 (%s)"
          " —— 若这不是你有意改的索引头，请立即回退" % (
        anchor["origin_head_prefix_lines"], anchor["origin_head_prefix_md5"][:12],
        anchor["origin_head_suffix_lines"], anchor["origin_head_suffix_md5"][:12]))
    return anchor


def head_segments(old_lines, new_lines):
    """对齐两版索引头 → [(old_start, old_end, new_start, new_end)]（1-based、等长 1:1）。

    只保留「等长 1:1」的段：等长段才能保证 `new = new_start + (orig - orig_start)`
    的偏移语义成立（`resolve_ref.py` 依赖这一点）。
    """
    segs = []
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if i2 == i1:
            continue
        if tag == "equal" or (tag == "replace" and (i2 - i1) == (j2 - j1)):
            segs.append((i1 + 1, i2, j1 + 1, j2))
    return segs


def compose_segments(stored, cur_to_new):
    """接上「原始→上一轮新头」与「上一轮新头→本轮新头」两段映射。"""
    out = []
    for o1, o2, c1, c2 in stored:
        for cc1, cc2, n1, n2 in cur_to_new:
            lo, hi = max(c1, cc1), min(c2, cc2)
            if lo > hi:
                continue
            out.append((o1 + (lo - c1), o1 + (hi - c1), n1 + (lo - cc1), n1 + (hi - cc1)))
    return out


# ---------------------------------------------------------------- 结构切分
def locate_body(orig_lines):
    """返回 (head_lines, body_lines, tail_lines, body_start_global)。

    `body_start_global` = 正文首行在**原文件**里的 1-based 全局行号。

    ⚠️ 两种布局的偏移不同，必须分别算，不能用一个公式套两边：
      * 已带标记：head | BODY_START | body | BODY_END | tail ⇒ 正文首行 = len(head)+2
      * 首次初始化（无标记）：head | body ⇒ 正文首行 = len(head)+1
    混用会让所有指针整体偏一行（校验器做的是 md5 全等比对 ⇒ 偏一行就会让
    归档区拼接 / 活区的 md5 直接对不上，当场现形，不会被任何机制吸收掉）。
    """
    if BODY_START in orig_lines and BODY_END in orig_lines:
        bs = orig_lines.index(BODY_START)
        be = orig_lines.index(BODY_END)
        return orig_lines[:bs], orig_lines[bs + 1:be], orig_lines[be + 1:], bs + 2
    for i, ln in enumerate(orig_lines):
        if ln.startswith("## ") and heading_label(ln) == "11":
            # 标记由写盘时统一补 ⇒ 这里 head 不含 BODY_START，tail 为空
            return orig_lines[:i], orig_lines[i:], [], i + 1
    raise SystemExit("[FAIL] 找不到 `## 11.` 标题，无法初始化正文边界")


def sectionize(body_lines):
    """按 `## ` 标题切节，返回 [(start, end_exclusive, heading)]（0-based）。"""
    starts = [i for i, ln in enumerate(body_lines) if ln.startswith("## ")]
    out = []
    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else len(body_lines)
        out.append((s, e, body_lines[s]))
    return out


def plan_volumes(sections, target, live_max):
    """返回 (归档卷的节列表们, 活章节列表)。切分点只落在节边界。

    幂等性要求：正文整体已经装得下活区窗口时**一卷都不建**（否则二次运行
    会把活章重新归档一遍）。

    活区的取法：从**尾部**贪心地找出能装进 `live_max` 的**最长后缀**；
    剩下的前缀再按 `target` 切卷。

    ⚠ 活区**至少保留最后一节**（哪怕它单独就超过 `live_max`）——
    旧实现是「先按 target 切卷、再把整卷倒回活区」，正文刚超过 `live_max`
    时只会产出 1 卷 ⇒ 循环不执行 ⇒ 活区被整个清空 ⇒ `memory.md` 退化成
    纯索引头（正文全进了归档区，尾部章节再也看不到）。
    """
    total = sum(s[1] - s[0] for s in sections)
    if total <= live_max:
        return [], list(sections)
    # 活区 = 从尾部贪心取「能装进 live_max 的最长后缀」（最后一节必留）
    live = []
    live_lines = 0
    for sec in reversed(sections):
        sec_lines = sec[1] - sec[0]
        if live and live_lines + sec_lines > live_max:
            break
        live.insert(0, sec)
        live_lines += sec_lines
    rest = sections[:len(sections) - len(live)]
    # 剩余前缀按 target 切卷
    vols, cur, cur_lines = [], [], 0
    for sec in rest:
        sec_lines = sec[1] - sec[0]
        if cur and cur_lines + sec_lines > target:
            vols.append(cur)
            cur, cur_lines = [], 0
        cur.append(sec)
        cur_lines += sec_lines
    if cur:
        vols.append(cur)
    return vols, live


def date_span(sections):
    dates = []
    for s in sections:
        m = DATE_RE.search(s[2])
        if m:
            dates.append("%s%s%s" % m.groups())
    return (min(dates), max(dates)) if dates else ("nodate", "nodate")


def sec_range(sections):
    """卷的 § 区间。

    正文里 § 编号**并非严格单调**（实测 6 处回退：`## 9. 维护规则`、`## 一、…`、
    以及 §78/§140/§266/§288.1 几处乱序补记）。所以这里不取「首尾标签」，
    而是取「运行中最大值的最后一次出现」⇒ 卷名永远单调可读。
    """
    labels = [sec_label(x[2]) for x in sections if sec_label(x[2])]
    if not labels:
        return "unknown"
    best, best_val = labels[0], -1
    for lb in labels:
        try:
            v = int(lb)
        except ValueError:
            continue
        if v >= best_val:
            best_val, best = v, lb
    return "%s-%s" % (labels[0], best)


# ---------------------------------------------------------------- 导航行
def remap_nav_rows(text, mapping):
    """把指向 `memory.md` 的导航行改写为 mapping 给出的 (目标文件, 目标行号)。"""
    out, changed = [], 0
    for line in text.split("\n"):
        m = NAV_ROW_RE.match(line)
        if m:
            f = m.group("file").strip()
            if "memory.md" in f:
                g = int(m.group("line"))
                if g in mapping:
                    # 指向 memory.md 的行一律重映射：目标可能在别的卷（跨卷引用）
                    tgt_file, tgt_line = mapping[g]
                    cell = tgt_file if tgt_file == "memory.md" else "`%s`" % tgt_file
                    line = "| %s | %s | %s | %d |" % (
                        m.group("label"), m.group("title"), cell, tgt_line)
                    changed += 1
        out.append(line)
    return "\n".join(out), changed


def _table_rows_after(head_lines, start, stop):
    """取 [start, stop) 内最后一张带「起始行」表头的表体（不含表头与分隔行）。"""
    hdr = None
    for i in range(start, stop):
        if head_lines[i].startswith("|") and "起始行" in head_lines[i]:
            hdr = i
    if hdr is None:
        return []
    end = hdr + 1
    while end < stop and head_lines[end].startswith("|"):
        end += 1
    return head_lines[hdr + 2:end]


def find_nav_block(head_lines):
    """定位 §6 导航表区域。

    返回 (replace_start, replace_end, legacy_rows)：
      * replace_start..replace_end 是要被**整体替换**的行区间；
      * legacy_rows 是要保留的「既有归档分卷与专题档」表体。

    🔴 关键：第二次运行起，§6 里已经有本脚本生成的 6.1/6.2/6.3 三张表。
    若还按「找第一张带起始行的表」去定位，就会把**自己刚生成的表**当成遗留表
    再包一层 ⇒ 每跑一次表就多一层（幂等性破坏）。
    ⇒ 用显式生成块标记把自己产出的区域圈起来，只在块外找遗留表。
    """
    if NAV_AUTO_START in head_lines and NAV_AUTO_END in head_lines:
        s = head_lines.index(NAV_AUTO_START)
        e = head_lines.index(NAV_AUTO_END)
        return s, e + 1, _table_rows_after(head_lines, s, e + 1)
    # 首次运行：§6 里的原始表就是要保留的遗留表
    start = None
    for i, ln in enumerate(head_lines):
        if ln.startswith("## 6. "):
            start = i
            break
    if start is None:
        raise SystemExit("[FAIL] 索引头里找不到 §6 导航表")
    hdr = None
    for i in range(start, len(head_lines)):
        if head_lines[i].startswith("|") and "起始行" in head_lines[i]:
            hdr = i
            break
    if hdr is None:
        raise SystemExit("[FAIL] §6 里找不到导航表")
    end = hdr + 1
    while end < len(head_lines) and head_lines[end].startswith("|"):
        end += 1
    return hdr, end, head_lines[hdr + 2:end]


# ---------------------------------------------------------------- 规划
def plan(manifest=None):
    original = MAIN.read_text(encoding="utf-8")
    orig_lines = original.split("\n")
    head_lines, body_lines, tail_lines, body_start_global = locate_body(orig_lines)

    manifest = manifest or {}
    origin = manifest.get("origin")
    old_vols = manifest.get("volumes", [])
    stored_map = manifest.get("head_map") or {}
    if bool(origin) != bool(stored_map.get("segments")):
        raise SystemExit(
            "[FAIL] manifest 的锚点不完整：origin 与 head_map 必须成对存在"
            "（缺一说明 pre-split 源的一部分坐标已丢失，无法安全续跑）")
    if origin:
        # 已切分过：坐标一律用 **pre-split 源** 的全局行号，不能用当前文件的
        origin_body_start = origin["origin_body_start_global"]
        # 原正文里已被切走的前缀长度（当前正文 = 原正文的后缀）
        if any("body_last" not in m for m in old_vols):
            raise SystemExit("[FAIL] manifest 的分卷缺 body_last，无法确定原正文坐标偏移")
        origin_offset = max([m.get("body_last", 0) for m in old_vols] or [0])
    else:
        if BODY_START in orig_lines:
            raise SystemExit(
                "[FAIL] manifest 里没有 origin 锚点：pre-split 单体源已不在盘上，"
                "无法安全续跑（请用原始单体源重跑一次以重建锚点）")
        # 首次切分：当前文件就是 pre-split 单体源
        origin_body_start = body_start_global
        origin_offset = 0

    sections = sectionize(body_lines)
    if not sections:
        raise SystemExit("[FAIL] 正文里没有 `## ` 章节，拒绝执行")

    # 前置闸门：当前正文与已归档区**重叠** ⇒ 拒绝续跑。
    # 典型误操作：把 `memory.md` 还原成切分前的单体源再跑一次 —— 那样会
    # 静默产出 12 个同名卷、把现有 12 个卷全部覆盖写，manifest 写成 24 条。
    if origin and old_vols:
        archived_max = last_archived_section_no(old_vols)
        body_first = next((section_no(sec_label(s[2])) for s in sections
                           if section_no(sec_label(s[2])) is not None), None)
        if body_first is not None and archived_max is not None and body_first <= archived_max:
            raise SystemExit(
                "[FAIL] 当前 memory.md 与已归档区重叠（正文首节 §%d ≤ 已归档末节 §%d），拒绝续跑"
                % (body_first, archived_max))

    vols, live = plan_volumes(sections, VOLUME_TARGET_LINES, LIVE_MAX_LINES)

    # 新卷号**动态续算**（盘上 ∪ manifest 的现有最大卷号 + 1），
    # 不能写死：写死会在冻结卷 00~07 + 主档卷 08~19 之后产出**同号**卷。
    vol_start = next_volume_no(manifest)

    # 卷元数据 + 卷文本
    vol_meta = []
    for vi, vsec in enumerate(vols):
        dfrom, dto = date_span(vsec)
        srange = sec_range(vsec)
        vol_no = vol_start + vi
        name = "%02d_主档正文_§%s_%s-%s.md" % (vol_no, srange, dfrom, dto)
        first, last = vsec[0][0], vsec[-1][1]
        # 分卷在**原正文**里的下标（pre-split 坐标），manifest 记的就是它
        body_first, body_last = origin_offset + first, origin_offset + last
        hdr = [
            "<!-- ============================================================ -->",
            "<!-- 主档正文分卷 %02d ｜ §%s ｜ %s ~ %s ｜ 原始全局行 %d-%d -->" % (
                vol_no, srange, dfrom, dto,
                origin_body_start + body_first, origin_body_start + body_last - 1),
            "<!-- 由 `manu/memory/roll_memory.py` 从 `memory.md` 切出；本卷只增不改。 -->",
            "<!-- 零丢失校验：python3 manu/memory/check_memory_roll.py -->",
            "<!-- ============================================================ -->",
            "",
        ]
        body_part = body_lines[first:last]
        header_text = "\n".join(hdr)
        vol_meta.append({
            "vol": "%02d" % vol_no,
            "name": name, "path": "archive/%s" % name,
            "sec_range": srange, "date_from": dfrom, "date_to": dto,
            "sec_count": len(vsec),
            "lines": len(body_part),
            # 实际占用的行数（hdr 末尾的空元素不占行）＝ 换行符个数
            "header_lines": header_text.count("\n"),
            "body_first": body_first, "body_last": body_last,
            "text": "\n".join(hdr) + "\n".join(body_part),
            "header_chars": len(header_text),   # 正文首字符的位置
            "sections": [
                {
                    "label": sec_label(x[2]), "title": sec_title(x[2]),
                    "heading": x[2],
                    "body_start": origin_offset + x[0], "body_end": origin_offset + x[1],
                    "global_start": origin_body_start + origin_offset + x[0],
                    "local_line": header_text.count("\n") + (x[0] - first) + 1,
                }
                for x in vsec
            ],
        })

    # 归档区：原始全局行 → (卷路径, 卷内行号)
    mapping = {}
    for m in vol_meta:
        base = m["header_lines"]
        for k in range(m["body_first"], m["body_last"]):
            mapping[origin_body_start + k] = (m["path"], base + (k - m["body_first"]) + 1)

    # 活章记录
    live_secs = []
    for s in live:
        live_secs.append({
            "label": sec_label(s[2]), "title": sec_title(s[2]),
            "heading": s[2], "body_start": s[0], "body_end": s[1],
            "global_start": origin_body_start + origin_offset + s[0],
            "lines": s[1] - s[0],
        })

    return {
        "original": original, "orig_lines": orig_lines,
        "head_lines": head_lines, "body_lines": body_lines, "tail_lines": tail_lines,
        "body_start_global": body_start_global, "sections": sections,
        "vol_meta": vol_meta, "live_secs": live_secs, "mapping": mapping,
        "origin_body_start": origin_body_start, "origin_offset": origin_offset,
    }


# ---------------------------------------------------------------- 组装索引头
def assemble_head(ctx, live_map):
    head_lines = ctx["head_lines"]
    vol_meta = ctx["vol_meta"]
    mapping = ctx["mapping"]
    live_secs = ctx["live_secs"]

    hdr, end, legacy_rows = find_nav_block(head_lines)

    # 6.3 中「本文档内（活章节）」的行 → 改指向真实分卷
    by_label = {}
    for m in vol_meta:
        for sec in m["sections"]:
            by_label.setdefault(sec["label"], sec["global_start"])
    for sec in live_secs:
        by_label.setdefault(sec["label"], sec["global_start"])

    fixed_rows = []
    for row in legacy_rows:
        m = NAV_ROW_RE.match(row)
        if m and "本文档内" in m.group("file"):
            lab = norm_label(m.group("label"))
            g = by_label.get(lab)
            if g is not None and g in mapping:
                tgt_file, tgt_line = mapping[g]
                fixed_rows.append("| %s | %s | `%s` | %d |" % (
                    lab, m.group("title"), tgt_file, tgt_line))
                continue
            if g is not None:
                fixed_rows.append("| %s | %s | `memory.md` | %d |" % (
                    lab, m.group("title"), live_map.get(g, g)))
                continue
        fixed_rows.append(row)

    # 6.1 分卷索引
    t1 = ["### 6.1 主档正文分卷索引（`manu/memory/archive/`）",
          "",
          "| 卷 | §区间 | 日期区间 | 归档文件 | 正文行数 | 章节数 |",
          "|:--|:--|:--|:--|--:|--:|"]
    for m in vol_meta:
        t1.append("| %s | %s | %s ~ %s | `%s` | %d | %d |" % (
            m["vol"], m["sec_range"], m["date_from"], m["date_to"],
            m["path"], m["lines"], m["sec_count"]))
    t1 += ["",
           ("> 卷名即「`NN_主档正文_<起§>-<止§>_<起日期>-<止日期>.md`」，**按 § 编号区间切**；"
            "每卷首行自带出处标记。"),
           ""]

    # 6.2 正文全量章节索引
    t2 = ["### 6.2 主档正文全量章节索引（§编号 → 分卷 + 卷内行号）",
          "",
          "<!-- 本表由 `manu/memory/roll_memory.py` 自动生成，请勿手工编辑 -->",
          "",
          "| § | 标题（简） | 归档文件 | 起始行 |",
          "|:--|:--|:--|--:|"]
    for m in vol_meta:
        for sec in m["sections"]:
            t2.append("| %s | %s | `%s` | %d |" % (
                sec["label"], sec["title"], m["path"], sec["local_line"]))
    for sec in live_secs:
        t2.append("| %s | %s | `memory.md` | %d |" % (
            sec["label"], sec["title"], live_map.get(sec["global_start"], 0)))
    t2.append("")

    # 6.3 既有归档分卷与专题档（原表保留）
    t3 = ["### 6.3 既有归档分卷与专题档（§一~§五十六 · 原表保留）",
          "",
          "| § | 标题（简） | 归档文件 | 起始行 |",
          "|:--|:--|:--|--:|"] + fixed_rows + [""]

    return (head_lines[:hdr] + [NAV_AUTO_START] + t1 + t2 + t3 + [NAV_AUTO_END]
            + head_lines[end:])


# ---------------------------------------------------------------- manifest
def load_manifest():
    if not MANIFEST.exists():
        return None
    try:
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit("[FAIL] manifest 读取失败：%s" % exc)


def save_manifest(data):
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def body_line_global(head_len, k):
    """正文第 k 行（0-based）在文件中的 1-based 全局行号。

    文件布局：head 行 ×N， 然后 BODY_START 一行， 然后正文。
    ⇒ 正文第 0 行落在第 N+2 行。
    """
    return head_len + 2 + k


def verify_existing_volumes(manifest):
    """跑之前核对既有卷：文件在不在、md5 与 manifest 记的是否一致。

    不一致 ⇒ 拒绝续跑（卷被人手改过 / 丢了）。
    """
    bad = []
    for m in (manifest or {}).get("volumes", []):
        p = HERE / m["path"]
        if not p.is_file():
            bad.append("分卷缺失：%s" % m["path"])
            continue
        h = md5_text(p.read_text(encoding="utf-8"))
        if h != m.get("md5"):
            bad.append("分卷被改动：%s（md5 %s ≠ 记录 %s）" % (m["path"], h[:12], str(m.get("md5"))[:12]))
    if bad:
        raise SystemExit("[FAIL] 既有归档卷与 manifest 不一致，拒绝续跑：\n  " + "\n  ".join(bad))
    return True


# ---------------------------------------------------------------- 主流程
def run_split(dry=False, refresh_head_anchor=False):
    manifest = load_manifest()
    # 跑之前先核对既有卷：用户手改过归档卷（或卷丢了）就拒绝续跑。
    # 不核对的话，续跑会把「被改过的旧卷」原样留在盘上，manifest 却按新内容算，
    # 等于悄悄承认了篡改。
    verify_existing_volumes(manifest)
    ctx = plan(manifest)
    old_vols = (manifest or {}).get("volumes", [])

    # manifest 上**已记录**的索引头锚点（先存快照：下面 reanchor_head 会改 origin）
    stored_head_anchor = None
    if manifest and has_head_anchor(manifest.get("origin") or {}):
        stored_head_anchor = {k: manifest["origin"][k] for k in HEAD_ANCHOR_KEYS}

    # origin 锚点：manifest 里有就**继承**（冻结），没有才从单体源现算
    origin = (manifest or {}).get("origin")
    if origin is None:
        origin = origin_anchor(
            ctx["original"], ctx["orig_lines"], ctx["body_start_global"], ctx["body_lines"])
        # 首次切分：索引头锚点与 origin 同时现算（同一份 pre-split 源）
        origin.update(head_anchor_fields(ctx["head_lines"]))
    elif refresh_head_anchor:
        reanchor_head(origin, ctx)
    elif not has_head_anchor(origin):
        # 历史遗留：旧脚本从没写过索引头锚点 ⇒ 一次性补齐（之后只继承不重算）
        seed_head_anchor(origin, ctx)

    # 定点迭代：活章的新全局行号依赖新索引头长度
    # ⚠️ 活区在新档里是**从 0 开始**的（前面只留活章，不留已归档的部分）
    #    ⇒ 新行号 = 新索引头长度 + 2 + (body_index − live_start)
    live_start = ctx["live_secs"][0]["body_start"] if ctx["live_secs"] else len(ctx["body_lines"])
    live_map = {}
    new_head = []
    for _ in range(8):
        new_head = assemble_head(ctx, live_map)
        nmap = {}
        for sec in ctx["live_secs"]:
            nmap[sec["global_start"]] = body_line_global(
                len(new_head), sec["body_start"] - live_start)
        if nmap == live_map:
            break
        live_map = nmap
    else:
        raise SystemExit("[FAIL] 活章行号定点迭代不收敛")

    # 活区也进 mapping（导航行可能指向活章内部任意行）
    for k in range(live_start, len(ctx["body_lines"])):
        ctx["mapping"][ctx["origin_body_start"] + ctx["origin_offset"] + k] = (
            "memory.md", body_line_global(len(new_head), k - live_start))

    # 合并「既有分卷（manifest）」+「本次新卷」⇒ 主导航表的卷索引
    all_vols = list(old_vols) + [
        {
            "vol": m["vol"], "name": m["name"], "path": m["path"],
            "sec_range": m["sec_range"], "date_from": m["date_from"],
            "date_to": m["date_to"], "lines": m["lines"], "sec_count": m["sec_count"],
            "header_lines": m["header_lines"], "header_chars": m["header_chars"],
            "md5": md5_text(m["text"]), "text": m["text"],
            "body_first": m["body_first"], "body_last": m["body_last"],
            "sections": m["sections"],
        }
        for m in ctx["vol_meta"]
    ]
    ctx["vol_meta"] = all_vols
    ctx["vol_texts"] = {i: m["text"] for i, m in enumerate(ctx["vol_meta"])
                        if "text" in m}

    # 既有分卷也要进 by_label / mapping（6.3 的「本文档内」行可能要指向它们）
    for m in all_vols:
        for s in m.get("sections", []):
            g = s["global_start"]
            if g not in ctx["mapping"]:
                ctx["mapping"][g] = (m["path"], s["local_line"])

    # 重建索引头（此时 vol_meta 已含既有分卷）
    new_head = []
    live_map = {}
    for _ in range(8):
        new_head = assemble_head(ctx, live_map)
        nmap = {}
        for sec in ctx["live_secs"]:
            nmap[sec["global_start"]] = body_line_global(
                len(new_head), sec["body_start"] - live_start)
        if nmap == live_map:
            break
        live_map = nmap
    else:
        raise SystemExit("[FAIL] 活章行号定点迭代不收敛（合并既有分卷后）")
    for k in range(live_start, len(ctx["body_lines"])):
        ctx["mapping"][ctx["origin_body_start"] + ctx["origin_offset"] + k] = (
            "memory.md", body_line_global(len(new_head), k - live_start))

    # 写卷：导航行重映射
    written = []
    remapped = 0
    for m in ctx["vol_meta"]:
        if "text" not in m:
            continue
        txt, n = remap_nav_rows(m["text"], ctx["mapping"])
        m["text"] = txt
        m["md5"] = md5_text(txt)
        written.append(m)
        remapped += n

    # 写 memory.md（活区里的导航行同样要重映射：主档行号会整体平移）
    live_body = ctx["body_lines"][live_start:] if ctx["live_secs"] else []
    live_text = "\n".join(live_body)
    if live_text:
        live_text, live_remapped = remap_nav_rows(live_text, ctx["mapping"])
        remapped += live_remapped
        live_body = live_text.split("\n")
    new_memory_lines = (
        new_head + [BODY_START] + live_body + [BODY_END] + ctx["tail_lines"]
    )
    new_memory = "\n".join(new_memory_lines)

    # 归档区内容（用于零丢失基准）
    # ⚠ 必须覆盖 manifest 里的**全部**卷，不能只算本次新卷：二次运行时正文已经
    #   滚完，`written` 为空，基准会退化成空串，零丢失校验随之失效。
    archived_parts = []
    for m in all_vols:
        text = m["text"] if "text" in m else (HERE / m["path"]).read_text(encoding="utf-8")
        archived_parts.append(text[m["header_chars"]:])
    archived_blob = "\n".join(archived_parts)   # 卷间补换行才是原来那份正文

    if dry:
        print("[DRY] 不写盘。计划：")
        print("  既有分卷：%d 卷" % len(old_vols))
        print("  新增分卷：%d 卷" % len(written))
        for m in written:
            print("    %s  §%s  %s ~ %s  %d 行 / %d 节" % (
                m["path"], m["sec_range"], m["date_from"], m["date_to"],
                m["lines"], m["sec_count"]))
        print("  活章：%d 节 / %d 行（留在 memory.md）" % (
            len(ctx["live_secs"]), len(live_body)))
        print("  导航行重映射：%d 条" % remapped)
        print("  memory.md：%d 行 → %d 行" % (
            len(ctx["orig_lines"]), len(new_memory_lines)))
        print("  分卷索引 §6.2 行数：%d" % (
            sum(m["sec_count"] for m in all_vols) + len(ctx["live_secs"])))
        print("  origin 锚点：源 md5 %s / %d 行，正文起始行 %d，归一化正文 %s / %d 行" % (
            origin["origin_source_md5"], origin["origin_source_lines"],
            origin["origin_body_start_global"],
            origin["origin_body_norm_md5"], origin["origin_body_norm_lines"]))
        if has_head_anchor(origin):
            print("  索引头锚点（manifest 记录）：prefix %d 行 md5 %s / suffix %d 行 md5 %s" % (
                origin["origin_head_prefix_lines"], origin["origin_head_prefix_md5"][:12],
                origin["origin_head_suffix_lines"], origin["origin_head_suffix_md5"][:12]))
            # 索引头锚点只对「NAV 自动块之外两段」敏感 ⇒ 用户改过索引头正文它就会漂移。
            # --dry 必须把「当前值 vs 记录值」并排打出来：否则用户改了索引头后跑 --dry
            # 看到的还是旧锚点（=「没漂移」的错觉），跑真实 roll 后才发现闸门 6 变红。
            cur_anchor = head_anchor_fields(ctx["head_lines"])
            drift = [k for k in HEAD_ANCHOR_KEYS
                     if stored_head_anchor and cur_anchor.get(k) != stored_head_anchor.get(k)]
            if refresh_head_anchor:
                print("  索引头锚点：已按 --refresh-head-anchor 重锚为**当前**值"
                      "（prefix %s / suffix %s）—— 去掉 --dry 才会落进 manifest" % (
                          cur_anchor["origin_head_prefix_md5"][:12],
                          cur_anchor["origin_head_suffix_md5"][:12]))
            elif drift:
                print("  ⚠ 索引头锚点**已漂移**（当前 ≠ manifest 记录）：%s" % " / ".join(
                    "%s 当前 %s ≠ 记录 %s" % (
                        k, str(cur_anchor.get(k))[:12], str(stored_head_anchor.get(k))[:12])
                    for k in drift))
                print("     ⇒ 跑真实 roll 会让闸门 6 变红；请加 --refresh-head-anchor 重锚")
            else:
                print("  索引头锚点：当前 == 记录，一致")
        if refresh_head_anchor:
            print("  ⚠ --refresh-head-anchor：锚点已重算，但 --dry **未写盘**"
                  "（去掉 --dry 才会落进 manifest）")
        return 0

    # 落盘前：同名保护 —— 目标卷已存在且内容不同 ⇒ **拒绝覆盖**。
    # 旧实现会静默把同名卷覆盖写（把 memory.md 还原成单体源再跑一次就能
    # 把 12 个卷全部覆盖掉，manifest 写成 24 条，exit 0 零警告）。
    for m in written:
        target = HERE / m["path"]
        if not target.exists():
            continue
        cur = md5_text(target.read_text(encoding="utf-8"))
        if cur != m["md5"]:
            raise SystemExit(
                "[FAIL] 目标分卷已存在且内容不同，拒绝覆盖：%s（盘上 %s ≠ 待写 %s）"
                % (m["path"], cur[:12], m["md5"][:12]))

    # 落盘
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    for m in written:
        (HERE / m["path"]).write_text(m["text"], encoding="utf-8")
    MAIN.write_text(new_memory, encoding="utf-8")

    # 活章坐标：原始全局行号 → 新 memory.md 行号
    live_sections = [
        {
            "label": sec["label"], "title": sec["title"], "heading": sec["heading"],
            "global_start": sec["global_start"],
            "local_line": live_map.get(sec["global_start"], 0),
        }
        for sec in ctx["live_secs"]
    ]

    # 索引头坐标：原始全局行号 → 新 memory.md 行号
    # 首次切分：当前头就是原始头，直接对齐；之后：上一轮「原始→新头」接上
    # 「上一轮新头→本轮新头」。
    stored_map = (manifest or {}).get("head_map") or {}
    cur_to_new = head_segments(ctx["head_lines"], new_head)
    if stored_map.get("segments"):
        segs = compose_segments(
            [(s["orig_start"], s["orig_end"], s["new_start"], s["new_end"])
             for s in stored_map["segments"]],
            cur_to_new)
        orig_head_lines = stored_map["orig_head_lines"]
    else:
        segs = cur_to_new
        orig_head_lines = len(ctx["head_lines"])
    head_map = {
        "orig_head_lines": orig_head_lines,
        "segments": [
            {"orig_start": a, "orig_end": b, "new_start": c, "new_end": d}
            for a, b, c, d in segs
        ],
    }

    # manifest
    new_manifest = {
        "version": 2,
        "updated": datetime.date.today().isoformat(),
        "origin": origin,
        "source_md5": origin["origin_source_md5"],
        "source_lines": origin["origin_source_lines"],
        "body_start_global": origin["origin_body_start_global"],
        "archived_body_md5": md5_text(archived_blob),
        "archived_body_lines": archived_blob.count("\n"),
        "live_md5": md5_text("\n".join(live_body)),
        "live_lines": len(live_body),
        "head_md5": md5_text("\n".join(new_head)),
        "head_lines": len(new_head),
        "live_sections": live_sections,
        "head_map": head_map,
        "volumes": [
            {k: v for k, v in m.items() if k not in ("text",)}
            for m in all_vols
        ],
        # 累计语义：本字段 = 历次运行**累计**改写过的导航行条数（逐次累加）。
        # 旧实现每次直接覆写 ⇒ 第二次运行把它冲成 0，首次切分的改写记录（75 条）
        # 就此消失，看起来像「切分没改过导航行」。
        "nav_rows_remapped": int((manifest or {}).get("nav_rows_remapped", 0) or 0) + remapped,
    }
    save_manifest(new_manifest)

    print("[OK] 切分完成：")
    print("  新增分卷 %d 卷（既有 %d 卷，共 %d 卷）" % (
        len(written), len(old_vols), len(all_vols)))
    for m in written:
        print("    %s  §%s  %s ~ %s  %d 行 / %d 节" % (
            m["path"], m["sec_range"], m["date_from"], m["date_to"],
            m["lines"], m["sec_count"]))
    print("  活章：%d 节 / %d 行（留在 memory.md）" % (
        len(ctx["live_secs"]), len(live_body)))
    print("  导航行重映射：%d 条" % remapped)
    print("  memory.md：%d 行 → %d 行" % (len(ctx["orig_lines"]), len(new_memory_lines)))
    # 写盘后提醒索引头锚点的漂移：闸门 6 只认「manifest 记录 vs 落盘后索引头」。
    # 用户改过索引头正文而没带 --refresh-head-anchor ⇒ 闸门 6 会变红，
    # 这里提前把话说清楚，别让人跑完校验才发现。
    if stored_head_anchor and not refresh_head_anchor:
        landed = head_anchor_fields(new_head)
        drift = [k for k in HEAD_ANCHOR_KEYS
                 if landed.get(k) != stored_head_anchor.get(k)]
        if drift:
            print("  ⚠ 索引头锚点**已漂移**（落盘后 ≠ manifest 记录）：%s" % " / ".join(
                "%s 落盘 %s ≠ 记录 %s" % (
                    k, str(landed.get(k))[:12], str(stored_head_anchor.get(k))[:12])
                for k in drift))
            print("     ⇒ 闸门 6 会变红；重跑 `roll_memory.py --refresh-head-anchor` 重锚即可")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="只打印计划，不写盘")
    ap.add_argument("--check", action="store_true", help="只跑零丢失校验")
    ap.add_argument("--refresh-head-anchor", action="store_true",
                    help="显式重锚索引头（你**有意**改过索引头内容后用；默认不重算）")
    args = ap.parse_args()
    if args.check:
        sys.path.insert(0, str(HERE))
        import check_memory_roll
        return check_memory_roll.run()
    return run_split(dry=args.dry, refresh_head_anchor=args.refresh_head_anchor)


if __name__ == "__main__":
    sys.exit(main())
