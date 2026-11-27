#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修复深轮报告生成器 ``iqa_v2/report_deep_v2.py`` 的「最重档跟住率取反」缺陷。

缺陷（2026-11-27 独立验收发现，P0）
----------------------------------
``report_deep_v2.py`` 的 ``level_rates()`` 把每一族的档位按
``degradation_strength(family, level)`` **递增**排序（退化严重度递增序，
轻 → 重），于是：

* ``series[0]``  = **最轻档**
* ``series[-1]`` = **最重档**

但生成器在两处把 ``series[0]`` 当成「最重档」用：

1. ``heavy[(family, arm)] = series[0][1]``  ⇒ §4.2 / §八 的 A1 实测值
2. ``fmt_rate(dser[0][1])`` 等 5 处  ⇒ §4.1「最重档跟住率」表

⇒ 后果：报告里「最重档跟住率」全部写成了**最轻档**的读数
（`blur`/XCorr 写成 98.7%，真值 **87.9%**；`motion`/XCorr 写成 99.0%，真值 **89.0%**），
A1 的实测值也被写成 98.7%。判定结论「不通过」本身没错，但**对外可发布的数字是错的**。

用法
----
    python3 iqa_v2/fix_deep_report_heaviest.py --check    # 只体检，不改文件
    python3 iqa_v2/fix_deep_report_heaviest.py --apply    # 打补丁（先备份 .bak）

设计约束
--------
* **精确串匹配**：每个补丁点必须**恰好命中一次**，否则报错退出（不猜、不模糊匹配）；
* **幂等**：已经打过补丁时，命中数变 0 ⇒ 视为「已修复」，不重复改；
* **可回滚**：``--apply`` 先写 ``<file>.bak``，回滚命令见 ``--help``；
* 不碰仓库里其它任何文件。

验证
----
    python3 iqa_v2/acceptance_deep.py --report iqa_v2/REPORT_V2_DEEP.md
    # D6「报告『最重档跟住率』表与明细一致」应从 FAIL 变 PASS
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

DEFAULT_TARGET = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "report_deep_v2.py")

#: (补丁编号, 说明, 旧串, 新串)
PATCHES = [
    (
        "P1",
        "heavy[(family, arm)] 取最重档（严重度递增序 ⇒ 末位才是最重档）",
        """            if series:
                heavy[(family, arm)] = series[0][1]""",
        """            if series:
                heavy[(family, arm)] = series[-1][1]""",
    ),
    (
        "P2",
        "§4.1「最重档跟住率」表的 5 个取数点",
        """               fmt_rate(bser[0][1]) if bser else "—",
               fmt_rate(dser[0][1]) if dser else "—",
               fmt_rate(bser_k[0][1]) if bser_k else "—",
               fmt_rate(dser_k[0][1]) if dser_k else "—",
               fmt_rate(dser_c[0][1]) if dser_c else "—"))""",
        """               fmt_rate(bser[-1][1]) if bser else "—",
               fmt_rate(dser[-1][1]) if dser else "—",
               fmt_rate(bser_k[-1][1]) if bser_k else "—",
               fmt_rate(dser_k[-1][1]) if dser_k else "—",
               fmt_rate(dser_c[-1][1]) if dser_c else "—"))""",
    ),
]


def main(argv=None):
    parser = argparse.ArgumentParser(description="修复 report_deep_v2.py 的「最重档取反」缺陷")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="目标源文件")
    parser.add_argument("--check", action="store_true", help="只体检，不改文件")
    parser.add_argument("--apply", action="store_true", help="打补丁（先备份 .bak）")
    args = parser.parse_args(argv)

    if not os.path.exists(args.target):
        print("[FIX] 目标文件不存在：%s" % args.target)
        return 2
    text = open(args.target, encoding="utf-8").read()

    pending = []
    already = []
    broken = []
    for ident, why, old, new in PATCHES:
        hits = text.count(old)
        if hits == 1:
            pending.append((ident, why, old, new))
        elif hits == 0 and new in text:
            already.append((ident, why))
        else:
            broken.append((ident, why, hits))

    for ident, why in already:
        print("[FIX] %s 已修复（跳过）：%s" % (ident, why))
    for ident, why, hits in broken:
        print("[FIX] %s **无法自动匹配**（命中 %d 次）：%s" % (ident, hits, why))
    for ident, why, _old, _new in pending:
        print("[FIX] %s 待修：%s" % (ident, why))

    if broken:
        print("[FIX] 存在无法匹配的补丁点 ⇒ 已放弃（不猜、不模糊匹配）")
        return 1
    if not pending:
        print("[FIX] 无需修改")
        return 0
    if args.check:
        print("[FIX] --check：%d 个补丁点待修（未改文件）" % len(pending))
        return 0
    if not args.apply:
        print("[FIX] 只做了体检（加 --apply 才改文件）")
        return 0

    backup = args.target + ".bak"
    shutil.copy2(args.target, backup)
    new_text = text
    for _ident, _why, old, new in pending:
        new_text = new_text.replace(old, new, 1)
    with open(args.target, "w", encoding="utf-8") as handle:
        handle.write(new_text)
    print("[FIX] 已应用 %d 个补丁点：%s" % (len(pending), args.target))
    print("[FIX] 备份：%s  回滚：cp '%s' '%s'" % (backup, backup, args.target))
    print("[FIX] 改完请重新生成报告并复核：")
    print("      python3 iqa_v2/report_deep_v2.py --output-dir iqa_v2/output_deep "
          "--baseline-dir iqa_v2/output --report iqa_v2/REPORT_V2_DEEP.md")
    print("      python3 iqa_v2/acceptance_deep.py   # 看 D6 是否由 FAIL 转 PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
