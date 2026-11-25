#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把「切分前的 memory.md 全局行号」解析成「切分后的 (文件, 行号)」。

背景（2026-11-24）：`manu/memory/memory.md` 从 29,344 行单文件切成
「索引头 + 活章 + archive/ 分卷」后，旧文档里写的 `memory.md:27888` 这类
**全局行号引用全部失效**（行号会随新增章节整体平移）。

本脚本按 `archive/_roll_manifest.json` 里的章节坐标做解析：
- **归档区**（原 §11 ~ §298）：`sections[]` 每节记了 `global_start`（切分前的
  全局行号）与 `local_line`（卷内行号）⇒ 命中该节后按行偏移平移。
- **活区**（原 §299 ~ §309）：记在 manifest 的 `live_sections[]`。
- **索引头**（原 1 ~ 736 行）：记在 manifest 的 `head_map.segments[]`。

用法：
    python3 manu/memory/resolve_ref.py 27888 27890-27891
    python3 manu/memory/resolve_ref.py --json 1686

输出形如：
    memory.md:27888  ->  archive/19_主档正文_§283-298_20261120-20261123.md:2357  (§298 区)

退出码 0 = 全部解析成功；1 = 有解析不到的行号。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "archive" / "_roll_manifest.json"


def load():
    if not MANIFEST.exists():
        raise SystemExit("[FAIL] manifest 不存在：%s" % MANIFEST)
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


BODY_START = "<!-- ===== MEMORY BODY START ===== -->"
BODY_END = "<!-- ===== MEMORY BODY END ===== -->"


def read_live_region():
    """从 memory.md 实测活区位置。

    返回 (活区首个正文行的 1-based 行号, 活区**真实**正文行数)。

    活区末尾若有一个空元素，那是原文件尾随换行切出来的幽灵行，
    不对应任何全局行号（manifest 的 live_lines 把它算进去了，这里扣掉）。
    """
    text = (HERE / "memory.md").read_text(encoding="utf-8")
    lines = text.split("\n")
    try:
        start = lines.index(BODY_START)
        end = lines.index(BODY_END)
    except ValueError:
        raise SystemExit("[FAIL] memory.md 里找不到 BODY 标记，无法定位活区")
    body = lines[start + 1:end]
    real = len(body)
    if real and body[-1] == "":
        real -= 1
    return start + 2, real


def build_index(man):
    """返回 (归档节列表, 活节列表, 头部分段列表)。"""
    arch = []
    for vol in man.get("volumes", []):
        for sec in vol.get("sections", []):
            span = sec["body_end"] - sec["body_start"]
            arch.append({
                "g_start": sec["global_start"], "g_end": sec["global_start"] + span - 1,
                "path": vol["path"], "local": sec["local_line"], "label": sec["label"],
                "title": sec.get("title", ""),
            })
    arch.sort(key=lambda s: s["g_start"])

    # 活区：原 §299 ~ §309 在新 memory.md 里是**连续的一段**（中间不再切分），
    # 所以「原全局行号 → 新行号」是恒定平移，区间边界必须按实际文件取，
    # 不能只按章节首行（否则每节只有第一行能被解析）。
    live_first_line, live_real = read_live_region()
    live_secs = sorted(man.get("live_sections", []), key=lambda s: s["global_start"])
    live = []
    if live_secs:
        g_min = live_secs[0]["global_start"]
        # 交叉校验：manifest 记的首节卷内行号应与文件里 BODY 标记后的第一行一致
        if live_secs[0].get("local_line") != live_first_line:
            print("[WARN] manifest 的 live_sections[0].local_line=%s 与文件实测 %s 不一致，"
                  "以文件为准" % (live_secs[0].get("local_line"), live_first_line))
        for i, sec in enumerate(live_secs):
            g_start = sec["global_start"]
            g_end = (live_secs[i + 1]["global_start"] - 1) if i + 1 < len(live_secs) \
                else (g_min + live_real - 1)
            live.append({
                "g_start": g_start, "g_end": g_end,
                "path": "memory.md", "local": live_first_line + (g_start - g_min),
                "label": sec.get("label", ""), "title": sec.get("title", ""),
            })

    head = man.get("head_map", {}).get("segments", [])
    return arch, live, head


def resolve_line(g, arch, live, head):
    """把单个原始全局行号解析成 (path, local_line, label, title)；解析不到返回 None。"""
    for seg in head:
        if seg["orig_start"] <= g <= seg["orig_end"]:
            new = seg["new_start"] + (g - seg["orig_start"])
            return "memory.md", new, "索引头", ""
    for sec in arch:
        if sec["g_start"] <= g <= sec["g_end"]:
            return sec["path"], sec["local"] + (g - sec["g_start"]), sec["label"], sec["title"]
    for sec in live:
        if sec["g_start"] <= g <= sec["g_end"]:
            return sec["path"], sec["local"] + (g - sec["g_start"]), sec["label"], sec["title"]
    return None


def parse_spec(spec):
    """'27888' 或 '27890-27891' → (start, end)。"""
    spec = spec.strip().replace("，", ",")
    if "-" in spec:
        a, b = spec.split("-", 1)
        return int(a), int(b)
    return int(spec), int(spec)


def main() -> int:
    ap = argparse.ArgumentParser(description="解析切分前的 memory.md 全局行号 → 切分后的位置")
    ap.add_argument("lines", nargs="+", help="行号或行号区间，如 27888 或 27890-27891")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args()

    man = load()
    arch, live, head = build_index(man)
    results, failed = [], []
    for spec in args.lines:
        try:
            start, end = parse_spec(spec)
        except ValueError:
            print("[FAIL] 行号格式不对：%s" % spec)
            return 1
        for g in range(start, end + 1):
            hit = resolve_line(g, arch, live, head)
            if hit is None:
                failed.append(g)
                continue
            path, local, label, title = hit
            results.append({
                "orig_line": g, "path": path, "new_line": local,
                "label": label, "title": title,
            })

    if args.json:
        # stdout 只出一个 JSON 文档（结论放进 summary），保证能被 json.load 直接吃
        print(json.dumps({
            "summary": {
                "requested": len(results) + len(failed),
                "resolved": len(results),
                "unresolved": len(failed),
                "ok": not failed,
            },
            "resolved": results,
            "unresolved": failed,
        }, ensure_ascii=False, indent=2))
        return 1 if failed else 0

    for r in results:
        tail = "  (§%s %s)" % (r["label"], r["title"][:40]) if r["label"] else ""
        print("memory.md:%-6d ->  %s:%-6d%s" % (r["orig_line"], r["path"], r["new_line"], tail))
    for g in failed:
        print("[FAIL] 解析不到原始行号：%d" % g)

    if failed:
        print("\n=== 结论：%d 行解析失败 ===" % len(failed))
        return 1
    print("\n=== 结论：%d 行全部解析成功 ===" % len(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
