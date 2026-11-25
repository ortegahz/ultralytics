#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""记忆分卷的零丢失校验 —— 证明「切分没有丢一个字」。

六道闸门：

1. **逐卷 md5**：每个 `archive/NN_主档正文_*.md` 的 md5 必须等于 manifest 记录值
   ⇒ 分卷被人手改过立刻现形。
2. **归档区拼接**：所有分卷正文（去掉卷首标记）按序拼接，md5 与行数必须等于
   manifest 的 `archived_body_md5` / `archived_body_lines`
   ⇒ 少一卷、少一行、少一个字都会被抓。
3. **活区一致**：`memory.md` 的活章正文 md5 必须等于 manifest 的 `live_md5`
   ⇒ 主档被截断/篡改会被抓。
4. **索引全覆盖**：`memory.md` §6.2 的全量章节索引必须覆盖全部分卷里的每一节
   ⇒ 切了卷却没建索引 = 找不到内容 = 等同于丢失。
5. **导航行归一化零丢失**：把「归档区 + 活区」拼回 pre-split 的正文，**去掉导航行**
   之后与 manifest 里冻结的 `origin` 锚点比对 ⇒ 证明除导航行外一个字都没丢，
   且不依赖任何外部原件。
   ⚠ 比对口径是「**前缀相等 + 尾部增量报数**」：锚点冻结的是首次切分时的正文指纹，
   之后任何一次合法 roll 都会给正文**追加**内容 ⇒ 全等比对必然变红（那是新增，不是丢失）。
   ⇒ 前 `origin_body_norm_lines` 行必须逐字节不变；多出来的尾部行数按「新增」报数。
6. **索引头锚点**：索引头里 `NAV AUTO-GENERATED` 块**之外**的两段（prefix / suffix）
   的 md5 + 行数必须等于 manifest 的 `origin` 锚点 ⇒ 索引头少了/多了内容立刻现形
   （`origin` 的其它字段只冻结正文，查不出头的缺失）。

用法：
    python3 manu/memory/check_memory_roll.py

执行机：本地（纯文本 + hash）。退出码 0 = 全绿，1 = 有 FAIL。
"""

from __future__ import annotations

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

BODY_START = "<!-- ===== MEMORY BODY START ===== -->"
BODY_END = "<!-- ===== MEMORY BODY END ===== -->"
NAV_AUTO_START = "<!-- ===== NAV AUTO-GENERATED START ===== -->"
NAV_AUTO_END = "<!-- ===== NAV AUTO-GENERATED END ===== -->"
NAV_ROW_RE = re.compile(
    r"^\|\s*(?P<label>[^|]+?)\s*\|\s*(?P<title>[^|]+?)\s*\|\s*(?P<file>[^|]+?)\s*\|\s*(?P<line>\d+)\s*\|\s*$"
)
HEAD11_RE = re.compile(r"^##\s*(?:§\s*)?(\d+)\s*[.、,：:]")
ORIGINAL_COPY = Path(os.environ.get("MEMORY_ORIGINAL_COPY", "/tmp/memory_backup_20261124.md"))


def md5_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def original_body_lines(path: Path):
    """pre-split 原件的正文：从 `## 11.` 标题行到文件尾。"""
    lines = path.read_text(encoding="utf-8").split("\n")
    for i, ln in enumerate(lines):
        m = HEAD11_RE.match(ln)
        if m and m.group(1) == "11":
            return lines[i:]
    return []


def run() -> int:
    fails = []
    if not MANIFEST.exists():
        print("[FAIL] manifest 不存在：%s（还没切分过？）" % MANIFEST)
        return 1
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    print("=== 记忆分卷零丢失校验 ===")
    print("manifest：%s（updated %s）" % (MANIFEST.name, man.get("updated")))

    # 1) 逐卷 md5
    print("\n--- 闸门 1：逐卷 md5 ---")
    parts = []
    for m in man["volumes"]:
        p = HERE / m["path"]
        if not p.is_file():
            fails.append("分卷缺失：%s" % m["path"])
            print("[FAIL] 分卷缺失：%s" % m["path"])
            continue
        text = p.read_text(encoding="utf-8")
        h = md5_text(text)
        ok = h == m["md5"]
        if not ok:
            fails.append("分卷被改动：%s（md5 %s ≠ 记录 %s）" % (m["path"], h, m["md5"]))
        print("[%s] %-46s %6d 行  md5 %s" % (
            "PASS" if ok else "FAIL", m["path"], text.count("\n"), h[:12]))
        parts.append(text[m["header_chars"]:] if "header_chars" in m else text)

    # 2) 归档区拼接
    print("\n--- 闸门 2：归档区拼接（零丢失） ---")
    blob = "\n".join(parts)   # 卷间补换行
    h, n = md5_text(blob), blob.count("\n")
    ok = h == man["archived_body_md5"] and n == man["archived_body_lines"]
    if not ok:
        fails.append("归档区拼接不一致：md5 %s/记录 %s，行数 %d/记录 %d" % (
            h, man["archived_body_md5"], n, man["archived_body_lines"]))
    print("[%s] 归档区 %d 卷拼接：md5 %s / %d 行（记录 %s / %d 行）" % (
        "PASS" if ok else "FAIL", len(parts), h, n,
        man["archived_body_md5"], man["archived_body_lines"]))

    # 2b) 每卷正文首行必须是 `## ` 章节标题
    print("\n--- 闸门 2b：每卷正文首行是章节标题 ---")
    for m in man["volumes"]:
        p = HERE / m["path"]
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8")
        body = text[m["header_chars"]:] if "header_chars" in m else text
        first = body.split("\n", 1)[0]
        ok = first.startswith("## ")
        if not ok:
            fails.append("卷 %s 正文首行不是章节标题：%r" % (m["path"], first[:60]))
        print("[%s] %-46s 首行 %r" % ("PASS" if ok else "FAIL", m["path"], first[:44]))

    # 3) 活区一致
    print("\n--- 闸门 3：memory.md 活区一致 ---")
    live_lines = []
    head_lines = []
    if not MAIN.is_file():
        fails.append("memory.md 不存在")
        print("[FAIL] memory.md 不存在")
    else:
        lines = MAIN.read_text(encoding="utf-8").split("\n")
        if BODY_START not in lines or BODY_END not in lines:
            fails.append("memory.md 缺少 BODY 标记")
            print("[FAIL] memory.md 缺少 BODY 标记")
        else:
            bs, be = lines.index(BODY_START), lines.index(BODY_END)
            head = lines[:bs]
            live = lines[bs + 1:be]
            live_lines = live
            head_lines = head
            hh, lh = md5_text("\n".join(head)), md5_text("\n".join(live))
            okh = hh == man["head_md5"]
            okl = lh == man["live_md5"]
            if not okh:
                fails.append("索引头被改动（md5 %s ≠ 记录 %s）" % (hh, man["head_md5"]))
            if not okl:
                fails.append("活区被改动（md5 %s ≠ 记录 %s）" % (lh, man["live_md5"]))
            print("[%s] 索引头 %d 行 md5 %s" % ("PASS" if okh else "FAIL", len(head), hh[:12]))
            print("[%s] 活  区 %d 行 md5 %s" % ("PASS" if okl else "FAIL", len(live), lh[:12]))

    # 4) 索引全覆盖
    print("\n--- 闸门 4：§6.2 全量索引覆盖每一节 ---")
    if MAIN.is_file():
        text = MAIN.read_text(encoding="utf-8")
        idx = set()
        for line in text.split("\n"):
            m = NAV_ROW_RE.match(line)
            if m and m.group("file").strip().startswith("`archive/"):
                idx.add((m.group("file").strip().strip("`"), m.group("label").strip()))
        missing = []
        total = 0
        for mv in man["volumes"]:
            for s in mv.get("sections", []):
                total += 1
                if (mv["path"], s["label"]) not in idx:
                    missing.append("%s §%s" % (mv["path"], s["label"]))
        ok = not missing
        if not ok:
            fails.append("索引缺 %d 节：%s" % (len(missing), missing[:5]))
        print("[%s] 分卷共 %d 节，索引命中 %d 节，缺 %d 节" % (
            "PASS" if ok else "FAIL", total, total - len(missing), len(missing)))

    # 5) 导航行归一化零丢失（对照 manifest 里冻结的 origin 锚点）
    print("\n--- 闸门 5：导航行归一化后的正文零丢失 ---")
    origin = man.get("origin") or {}
    if not origin:
        fails.append("manifest 缺少 origin 锚点（pre-split 源身份 + 归一化正文指纹）")
        print("[FAIL] manifest 缺少 origin 锚点")
    else:
        rebuild = blob + "\n" + "\n".join(live_lines)
        norm = [ln for ln in rebuild.split("\n") if not NAV_ROW_RE.match(ln)]
        h, n = md5_text("\n".join(norm)), len(norm)
        anchor_md5 = origin.get("origin_body_norm_md5", "")
        anchor_lines = origin.get("origin_body_norm_lines", 0)
        # 锚点冻结的是**首次切分**时的正文指纹 ⇒ 之后任何一次合法 roll 都会给
        # 正文**追加**内容 ⇒ 全等比对必然变红（看起来像丢内容，其实是新增）。
        # ⇒ 改成「前缀相等 + 尾部增量报数」：前 anchor_lines 行必须逐字节不变
        #    （这是「切分至今一个字没丢」的证明），多出来的尾部算新增。
        if n < anchor_lines:
            ok = False
            fails.append("归一化正文比锚点**变少**了：现 %d 行 < 锚点 %d 行（内容丢了 %d 行）" % (
                n, anchor_lines, anchor_lines - n))
            print("[FAIL] 归一化正文 %d 行 < 锚点 %d 行（内容变少 ⇒ 丢了 %d 行）" % (
                n, anchor_lines, anchor_lines - n))
        else:
            pre_md5 = md5_text("\n".join(norm[:anchor_lines]))
            ok = pre_md5 == anchor_md5
            if not ok:
                fails.append("归一化正文前缀不一致：前 %d 行 md5 %s ≠ 锚点 %s" % (
                    anchor_lines, pre_md5, anchor_md5))
            print("[%s] 归一化正文 %d 行（总 md5 %s）/ 锚点前缀 %d 行 md5 %s（锚点 %s）" % (
                "PASS" if ok else "FAIL", n, h[:12], anchor_lines, pre_md5[:12], anchor_md5[:12]))
            inc = n - anchor_lines
            if inc > 0:
                print("[INFO] 本轮新增 %d 行（锚点后增量）—— 新增内容不算 FAIL" % inc)

        # 附加提示：切分改写了多少条导航行（需要 pre-split 原件，只读、不参与闸门）
        if ORIGINAL_COPY.is_file():
            onav = [ln for ln in original_body_lines(ORIGINAL_COPY) if NAV_ROW_RE.match(ln)]
            rnav = [ln for ln in rebuild.split("\n") if NAV_ROW_RE.match(ln)]
            diff = sum(1 for a, b in zip(onav, rnav) if a != b) + abs(len(onav) - len(rnav))
            print("[INFO] 导航行差异 %d 条（原 %d 条 / 现 %d 条，切分时改写了 file/line 列）" % (
                diff, len(onav), len(rnav)))
        else:
            print("[INFO] original not available (skipping nav-diff check)")

    # 6) 索引头锚点：NAV 自动块之外的两段必须与锚点逐字节一致
    print("\n--- 闸门 6：索引头锚点（NAV 块外 prefix / suffix） ---")
    origin = man.get("origin") or {}
    anchor_keys = ("origin_head_prefix_md5", "origin_head_prefix_lines",
                   "origin_head_suffix_md5", "origin_head_suffix_lines")
    missing = [k for k in anchor_keys if k not in origin]
    if missing:
        fails.append("manifest 的 origin 块缺少索引头锚点字段：%s" % ", ".join(missing))
        print("[FAIL] manifest 的 origin 块缺少索引头锚点字段：%s" % ", ".join(missing))
    elif not head_lines:
        fails.append("取不到 memory.md 的索引头（缺 BODY 标记？），无法校验索引头锚点")
        print("[FAIL] 取不到 memory.md 的索引头，无法校验索引头锚点")
    elif NAV_AUTO_START not in head_lines or NAV_AUTO_END not in head_lines:
        fails.append("索引头里找不到 NAV AUTO-GENERATED 标记块，无法定位 prefix / suffix")
        print("[FAIL] 索引头里找不到 NAV AUTO-GENERATED 标记块，无法定位 prefix / suffix")
    else:
        s = head_lines.index(NAV_AUTO_START)
        e = head_lines.index(NAV_AUTO_END)
        segs = (("prefix", head_lines[:s], "origin_head_prefix_md5", "origin_head_prefix_lines"),
                ("suffix", head_lines[e + 1:], "origin_head_suffix_md5", "origin_head_suffix_lines"))
        for tag, seg, md5key, linekey in segs:
            cur_md5 = md5_text("\n".join(seg))
            cur_n = len(seg)
            exp_md5 = origin[md5key]
            exp_n = origin[linekey]
            ok = (cur_md5 == exp_md5 and cur_n == exp_n)
            if not ok:
                if cur_n < exp_n:
                    detail = "索引头 %s 段**少了 %d 行**（现 %d 行 / 锚点 %d 行）" % (
                        tag, exp_n - cur_n, cur_n, exp_n)
                elif cur_n > exp_n:
                    detail = "索引头 %s 段**多了 %d 行**（现 %d 行 / 锚点 %d 行）" % (
                        tag, cur_n - exp_n, cur_n, exp_n)
                else:
                    detail = "索引头 %s 段行数相同但内容不同（md5 %s ≠ 锚点 %s）" % (
                        tag, cur_md5[:12], exp_md5[:12])
                fails.append(detail)
            print("[%s] 索引头 %s 段 %d 行 md5 %s（锚点 %s / %d 行）" % (
                "PASS" if ok else "FAIL", tag, cur_n, cur_md5[:12], exp_md5[:12], exp_n))

    print("\n=== 结论：%s ===" % ("全绿" if not fails else "有 %d 项 FAIL" % len(fails)))
    for f in fails:
        print("  [FAIL] %s" % f)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(run())
