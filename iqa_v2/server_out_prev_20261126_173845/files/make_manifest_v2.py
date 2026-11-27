#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 v2 产物 MD5 清单（任务书「交付物与自动化验收门禁」第 2.5 条）。

清单内容
--------
* **代码**：``iqa_v2/*.py``（7 个脚本，全部入库）
* **结论级产物**：``iqa_v2/output/*.csv|*.json`` ＋ ``iqa_v2/REPORT_V2_ANTIUAV.md``
* **大体量明细**：``/tmp/iqa_v2_cache/degrade_detail.csv``（逐窗口 × 逐条件的原始测量，
  按「中间产物不入库」口径留在 cache 目录）——**以绝对路径进清单**，
  这样「换机器复算」时只要保证 cache 落点存在即可校验。

用法（仓库根执行）::

    python3 iqa_v2/make_manifest_v2.py            # 生成
    md5sum -c iqa_v2/output/manifest_v2.md5       # 全量校验

执行机：本地主机（纯 md5 计算，不需要 cv2 / 数据集）。
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
BULK_DETAIL = Path("/tmp/iqa_v2_cache/degrade_detail.csv")


def md5_of(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(repo: Path, bulk: Path) -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    for path in sorted(repo.glob("iqa_v2/*.py")):
        items.append((str(path.relative_to(repo)), path))
    for path in sorted(repo.glob("iqa_v2/output/*.csv")) + sorted(repo.glob("iqa_v2/output/*.json")):
        items.append((str(path.relative_to(repo)), path))
    report = repo / "iqa_v2/REPORT_V2_ANTIUAV.md"
    if report.is_file():
        items.append((str(report.relative_to(repo)), report))
    if bulk.is_file():
        items.append((str(bulk), bulk))
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 v2 产物 MD5 清单")
    parser.add_argument("--repo-root", default=str(REPO))
    parser.add_argument("--bulk-detail", default=str(BULK_DETAIL))
    parser.add_argument("--out", default=str(REPO / "iqa_v2/output/manifest_v2.md5"))
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    bulk = Path(args.bulk_detail)
    out = Path(args.out).resolve()
    items = collect(repo, bulk)
    if not items:
        raise SystemExit("[FATAL] 没有可登记的产物（先跑 run_eval_v2.py）")

    lines = [
        "# 红外图像质量评估 v2（Target-Centric，Anti-UAV 红外）—— 产物 MD5 清单",
        "#",
        "# 生成脚本：iqa_v2/make_manifest_v2.py（仓库根执行）",
        "# 校验命令：md5sum -c iqa_v2/output/manifest_v2.md5（路径相对仓库根）",
        "#",
        "# 覆盖：代码(7) + 结论级产物(csv/json + 报告正文) + 大体量明细(绝对路径，位于 cache 侧)",
        "# ⚠️ 明细 CSV 按「中间产物不入库」口径不入库，只在本清单里登记 md5 作为证据链。",
        "#",
        "# md5",
    ]
    for name, path in items:
        lines.append("%s  %s" % (md5_of(path), name))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[MANIFEST] %d 项 → %s" % (len(items), out))

    # 自校验
    failed = 0
    for name, path in items:
        if md5_of(path) != dict(
                (n, md5_of(p)) for n, p in items)[name]:
            failed += 1
    if failed:
        raise SystemExit("[FAIL] %d 项自校验不一致" % failed)
    print("[MANIFEST] 自校验 %d/%d 一致" % (len(items), len(items)))


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
