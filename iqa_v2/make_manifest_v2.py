#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 v2 产物 MD5 清单（任务书「交付物与自动化验收门禁」第 2.5 条）。

清单内容
--------
* **代码**：``iqa_v2/*.py``（7 个脚本，全部入库）
* **结论级产物**：``iqa_v2/output/*.csv|*.json`` ＋ ``iqa_v2/REPORT_V2_ANTIUAV.md``
* **大体量明细**：``/tmp/iqa_v2_cache/degrade_detail.csv``（逐窗口 × 逐条件的原始测量，
  按「中间产物不入库」口径留在 cache 目录）——**不进本清单**，改登记在**旁挂规范摘要清单**。

为什么大体量明细要单独一份旁挂清单
------------------------------------
明细行由 ``iqa_v2/run_eval_v2.py`` 的 ``pool.imap_unordered(...)`` 收集 ⇒ **同一份输入重跑
一次，CSV 的行序就会变**（内容不变）⇒ 对它做**裸 md5** 必然「假失败」（把行序抖动误判成
数据变化）⇒ 把明细登记成裸 md5 等于给 ``md5sum -c`` 门禁埋雷。

因此 ``--bulk-mode canonical``（默认）把明细的**顺序无关规范摘要**（口径见
``iqa_v2/canonical_digest.py``）登记到旁挂清单
``<out 目录>/<out 文件名去扩展名>.bulk.canonical.txt``：

* 本清单里**不再有**明细的 md5 行 ⇒ ``md5sum -c`` 不会因行序抖动假失败；
* 旁挂清单在**行序无关**口径下比对 ⇒ 数据真变了必报、行序抖动不误报。

两条门禁命令（仓库根执行）::

    python3 iqa_v2/make_manifest_v2.py               # 生成清单 + 旁挂规范摘要清单
    md5sum -c iqa_v2/output/manifest_v2.md5          # 门禁 1：代码/结论级产物逐条校验
    python3 iqa_v2/canonical_digest.py verify \
        iqa_v2/output/manifest_v2.bulk.canonical.txt  # 门禁 2：明细（顺序无关）校验

深轮（deep）用**独立产物目录**时，登记该目录并另写一份清单（**不动基线清单**）::

    python3 iqa_v2/make_manifest_v2.py \
        --products-subdir iqa_v2/output_deep \
        --report iqa_v2/REPORT_V2_DEEP.md \
        --bulk-detail /tmp/iqa_v2_cache_deep/degrade_detail.csv \
        --out iqa_v2/output_deep/manifest_v2.md5

（``--extra-report`` 可重复，用来追加登记更多报告正文；``iqa_v2/MEMORY_SECTION.md``
是**必登记**正文，与选哪套报告无关。）

⚠️ **冻结基线清单** ``iqa_v2/output/manifest_v2.md5`` 是 **raw 模式**（历史形状：明细登记
裸 md5）生成的 ⇒ 要逐字复现它必须显式加 ``--bulk-mode raw``；默认的 canonical 口径下本
清单**不含**明细的 md5 行（明细改走旁挂清单的顺序无关摘要）。

执行机：本地主机（纯 md5 计算，不需要 cv2 / 数据集）。
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    # 🔴 复用 ``canonical_digest`` 的规范摘要口径 ⇒ 必须能 import 到它，**不依赖调用者的 cwd**。
    sys.path.insert(0, str(HERE))

from canonical_digest import DEFAULT_KEY_COLS, CanonicalDigestError, load_table  # noqa: E402

REPO = HERE.parent
BULK_DETAIL = Path("/tmp/iqa_v2_cache/degrade_detail.csv")

#: 默认登记的产物目录（**基线轮口径，不得改动**）。
DEFAULT_PRODUCTS_SUBDIR = "iqa_v2/output"

#: **不进清单**的产物文件（运行留痕类，含时间戳 ⇒ 会破坏清单的字节可复现性）。
EXCLUDED_PRODUCT_FILES = ("run_meta.json",)

#: 🔴 **大体量明细的「产物副本」文件名**（cache 侧原件的逐字节拷贝）。
#: 它的**行序**同样由 ``imap_unordered`` 决定 ⇒ 裸 md5 每次重跑都变 ⇒
#: canonical 口径下必须与 cache 侧原件**一起**改走旁挂的顺序无关摘要清单，
#: 否则「本清单不含明细」这句话在**整份清单**尺度上就是假的（产物目录里那一份仍被登记）。
#: raw 口径（复现冻结基线清单的历史形状）下**照旧登记**，不得改动。
BULK_COPY_NAMES = ("degrade_detail_v2.csv",)

#: 默认登记的主报告正文（**基线轮口径，不得改动**）。
DEFAULT_REPORT = "iqa_v2/REPORT_V2_FINAL.md"

#: 默认追加登记的报告正文（**基线轮口径，不得改动**）。
DEFAULT_EXTRA_REPORTS = ("iqa_v2/REPORT_V2_ANTIUAV.md",)

#: 无论登记哪套报告都**必登记**的正文（**基线轮口径，不得改动**）。
ALWAYS_REPORT_FILES = ("iqa_v2/MEMORY_SECTION.md",)

#: 默认登记的报告正文全量清单（主报告 ＋ 追加报告 ＋ 必登记正文）。
DEFAULT_REPORT_FILES = (DEFAULT_REPORT,) + DEFAULT_EXTRA_REPORTS + ALWAYS_REPORT_FILES

#: 清单文件名（默认写在产物目录里）。
MANIFEST_NAME = "manifest_v2.md5"

#: 旁挂规范摘要清单的表头（解释「为什么不是裸 md5」+ 给出校验命令）。
SIDECAR_HEADER = [
    "# 红外图像质量评估 v2 —— 大体量明细 CSV 的**顺序无关规范摘要**（canonical digest）旁挂清单",
    "#",
    "# 为什么单独一份：明细行由 run_eval_v2.py 的 pool.imap_unordered(...) 收集",
    "#   ⇒ 同一份输入重跑一次，CSV 的**行序**会变（内容不变）",
    "#   ⇒ 对它做裸 md5 必然「假失败」（把行序抖动误判成数据变化）",
    "#   ⇒ 这里登记的是「先规范化（键唯一 + 规范行串 + 排序）再摘要」的口径 ⇒ 行序无关。",
    "#",
    "# 校验命令（仓库根执行；退出码 0 = 全部一致）：",
    "#     python3 iqa_v2/canonical_digest.py verify <本文件>",
    "#",
    "# 格式：<规范摘要>  <明细 CSV 绝对路径>（与 md5sum 清单同形；# 开头的行是注释、不校验）",
]


def md5_of(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path, repo: Path) -> str:
    """清单里展示的路径：仓库内用相对路径（相对仓库根），仓库外用绝对路径。

    Args:
        path (Path): 目标路径。
        repo (Path): 仓库根。

    Returns:
        str: 展示用的路径字符串。
    """
    try:
        return str(path.relative_to(repo))
    except ValueError:
        return str(path)


def canonical_entries(bulk: Path, copies=(), repo: Path | None = None) -> list[tuple[str, Path, str]]:
    """大体量明细 ⇒ 顺序无关规范摘要的登记项（``[(摘要, 绝对路径, 展示路径), ...]``）。

    Args:
        bulk (Path): 大体量明细 CSV（cache 侧原件）。
        copies (tuple[Path, ...]): 明细的**产物副本**（产物目录里那一份逐字节拷贝）；
            同样按顺序无关摘要登记 ⇒ 产物目录里的那份**也**被门禁覆盖。
        repo (Path | None): 仓库根。给了它，**仓库内**的副本就登记成**相对仓库根**的路径
            ⇒ 旁挂清单在 host 与服务器两侧**逐字节相同**（绝对路径会随 checkout 位置变）。

    Returns:
        list[tuple[str, Path, str]]: 登记项（明细文件不存在 ⇒ 空列表）。

    Raises:
        SystemExit: 明细不满足规范摘要口径（算不出可信摘要）。
    """
    if not bulk.is_file():
        return []
    entries: list[tuple[str, Path, str]] = []
    for target in [bulk.resolve()] + [c.resolve() for c in copies]:
        if not target.is_file():
            continue
        try:
            digest = load_table(target, list(DEFAULT_KEY_COLS))["digest"]
        except CanonicalDigestError as exc:
            raise SystemExit("[FATAL] 算不出 %s 的规范摘要：%s" % (target, exc))
        shown = display_path(target, repo) if repo is not None else str(target)
        entries.append((digest, target, shown))
    return entries


def write_canonical_sidecar(path: Path, entries: list[tuple[str, Path, str]]) -> None:
    """写旁挂规范摘要清单（**可校验**的只有「<规范摘要>  <路径>」那几行）。

    🔴 登记的路径：仓库**外**的用绝对路径（cache 侧原件，两侧一致）；
    仓库**内**的用**相对仓库根**的路径 ⇒ 旁挂清单在 host 与服务器两侧**逐字节相同**。

    Args:
        path (Path): 旁挂清单路径。
        entries (list[tuple[str, Path, str]]): ``(规范摘要, 明细 CSV 绝对路径, 展示路径)`` 列表。
    """
    lines = list(SIDECAR_HEADER)
    for _, target, shown in entries:
        # 裸 md5 只作参考（**不可校验**）：它会随行序抖动而变。
        lines.append("# 裸 md5（仅供人读、**不可校验**）：%s  %s" % (md5_of(target), shown))
    lines.append("#")
    for digest, _, shown in entries:
        lines.append("%s  %s" % (digest, shown))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect(repo: Path, bulk: Path, products_subdir: str = DEFAULT_PRODUCTS_SUBDIR,
            report_files=DEFAULT_REPORT_FILES, output_dir: str | None = None,
            include_bulk: bool = True) -> list[tuple[str, Path]]:
    """收集要登记进清单的产物（代码 + 结论级产物 + 报告正文 + 大体量明细）。

    Args:
        repo (Path): 仓库根。
        bulk (Path): 大体量明细 CSV（绝对路径；不存在则不登记）。
        products_subdir (str): 结论级产物目录（相对仓库根）。基线轮 = ``iqa_v2/output``；
            深轮 = ``iqa_v2/output_deep``（**基线清单不得被覆盖**）。
        report_files (tuple[str, ...]): 要登记的报告正文（相对仓库根）；不存在则跳过。
        output_dir (str | None): ``products_subdir`` 的**别名**（对应 CLI 的 ``--output-dir``）；
            只在 ``products_subdir`` 仍为默认值（``DEFAULT_PRODUCTS_SUBDIR``）时生效 ⇒
            显式给了别的 ``products_subdir`` 时**以 products_subdir 为准**。
        include_bulk (bool): 是否把大体量明细登记进本清单。canonical 口径下它进**旁挂
            规范摘要清单**、不进本清单（行序不确定 ⇒ 裸 md5 必假失败）。

    Returns:
        list[tuple[str, Path]]: ``(登记名, 绝对路径)`` 列表。
    """
    subdir = (output_dir if output_dir is not None and products_subdir == DEFAULT_PRODUCTS_SUBDIR
              else products_subdir)
    items: list[tuple[str, Path]] = []
    for path in sorted(repo.glob("iqa_v2/*.py")):
        items.append((str(path.relative_to(repo)), path))
    products_dir = repo / subdir
    for path in sorted(products_dir.glob("*.csv")) + sorted(products_dir.glob("*.json")):
        # 🔴 ``run_meta.json`` 是**运行留痕**（含 UTC 时间戳）⇒ **不进清单**：
        #    否则每次重生成清单都因时间戳不同而变 ⇒ 清单失去字节可复现性。
        if path.name in EXCLUDED_PRODUCT_FILES:
            continue
        # 🔴 canonical 口径下：明细的**产物副本**也不进本清单（行序不确定 ⇒ 裸 md5 必假失败）
        if not include_bulk and path.name in BULK_COPY_NAMES:
            continue
        items.append((str(path.relative_to(repo)), path))
    for name in report_files:
        report = repo / name
        if report.is_file():
            items.append((str(report.relative_to(repo)), report))
    if include_bulk and bulk.is_file():
        items.append((str(bulk), bulk))
    return items


def build_header(bulk_mode: str, sidecar: Path | None, repo: Path, out: Path) -> list[str]:
    """构造清单表头（``raw`` 模式 = 冻结基线清单的表头，逐字不变）。

    Args:
        bulk_mode (str): ``canonical``（明细走旁挂清单）或 ``raw``（明细进本清单）。
        sidecar (Path | None): 旁挂规范摘要清单路径（``None`` = 本次没有生成）。
        repo (Path): 仓库根。
        out (Path): 本清单的输出路径（表头里的校验命令必须指向它）。

    Returns:
        list[str]: 表头行。
    """
    lines = [
        "# 红外图像质量评估 v2（Target-Centric，Anti-UAV 红外）—— 产物 MD5 清单",
        "#",
        "# 生成脚本：iqa_v2/make_manifest_v2.py（仓库根执行）",
        "# 校验命令：md5sum -c %s（路径相对仓库根）" % display_path(out, repo),
        "#",
    ]
    if bulk_mode == "raw":
        lines += [
            "# 覆盖：代码 + 结论级产物(csv/json + 报告正文) + 大体量明细(绝对路径，位于 cache 侧)",
            "# ⚠️ 明细 CSV 按「中间产物不入库」口径不入库，只在本清单里登记 md5 作为证据链。",
        ]
    else:
        lines += [
            "# 覆盖：代码 + 结论级产物(csv/json + 报告正文)",
            "# ⚠️ 大体量明细(cache 侧绝对路径)**不在本清单**：它的行序由 imap_unordered 决定 ⇒",
            "#    裸 md5 每次重跑都变 ⇒ 改登记「顺序无关」规范摘要（旁挂清单）：",
        ]
        if sidecar is None:
            lines.append("#    （本次未找到大体量明细文件 ⇒ 无旁挂清单）")
        else:
            shown = display_path(sidecar, repo)
            lines.append("#    %s" % shown)
            lines.append("#    校验：python3 iqa_v2/canonical_digest.py verify %s" % shown)
    lines += ["#", "# md5"]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 v2 产物 MD5 清单")
    parser.add_argument("--repo-root", default=str(REPO))
    parser.add_argument("--bulk-detail", default=str(BULK_DETAIL))
    parser.add_argument("--out", default=None,
                        help="清单输出路径（默认 <产物目录>/manifest_v2.md5 ——"
                             "基线轮即 iqa_v2/output/manifest_v2.md5 冻结基线清单）")
    parser.add_argument("--canonical-out", default=None,
                        help="旁挂规范摘要清单路径"
                             "（默认 <out 目录>/<out 文件名去扩展名>.bulk.canonical.txt）")
    parser.add_argument("--force", action="store_true",
                        help="允许**覆盖已存在且内容不同**的清单文件"
                             "（默认拒绝，避免误覆盖冻结基线清单）")
    parser.add_argument("--bulk-mode", choices=("canonical", "raw"), default="canonical",
                        help="大体量明细登记口径：canonical=旁挂规范摘要清单（默认）；"
                             "raw=清单里登记裸 md5（冻结基线清单的历史形状，重跑必假失败）")
    parser.add_argument("--products-subdir", default=None,
                        help="结论级产物目录（相对仓库根）；基线轮默认 iqa_v2/output，"
                             "深轮用 iqa_v2/output_deep")
    parser.add_argument("--output-dir", default=None,
                        help="--products-subdir 的别名（同一个东西）；两个都给出时以 "
                             "--products-subdir 为准")
    parser.add_argument("--report", default=DEFAULT_REPORT,
                        help="要登记的主报告正文（相对仓库根）；"
                             "基线轮默认 iqa_v2/REPORT_V2_FINAL.md")
    parser.add_argument("--extra-report", action="append", default=None, metavar="PATH",
                        help="追加登记的报告正文（相对仓库根，可重复）；"
                             "默认 iqa_v2/REPORT_V2_ANTIUAV.md")
    parser.add_argument("--report-files", default=None,
                        help="【已废弃】等价于 --report ＋ --extra-report（逗号分隔，相对仓库根）")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    bulk = Path(args.bulk_detail)
    if not repo.is_dir():
        raise SystemExit("[FATAL] 仓库根不存在：%s" % repo)
    products_subdir = args.products_subdir or args.output_dir or DEFAULT_PRODUCTS_SUBDIR
    out = (Path(args.out).resolve() if args.out
           else (repo / products_subdir / MANIFEST_NAME).resolve())
    sidecar = (Path(args.canonical_out).resolve() if args.canonical_out
               else out.parent / (out.stem + ".bulk.canonical.txt"))
    if args.report_files is not None:
        report_files = tuple(name.strip() for name in args.report_files.split(",")
                             if name.strip()) + ALWAYS_REPORT_FILES
    else:
        extra_reports = (tuple(args.extra_report) if args.extra_report is not None
                         else DEFAULT_EXTRA_REPORTS)
        report_files = (args.report,) + extra_reports + ALWAYS_REPORT_FILES

    # canonical 口径：明细**不进本清单**（行序不确定 ⇒ 裸 md5 必假失败），改登记旁挂清单。
    if args.bulk_mode == "canonical" and not bulk.is_file():
        sys.stderr.write(
            "[WARN] canonical 口径下找不到大体量明细 %s ⇒ 本次**没有**旁挂规范摘要清单，"
            "明细**未被任何门禁覆盖**（这不是「通过」，是「没检」）。\n" % bulk)
    bulk_copies = ()
    if args.bulk_mode == "canonical":
        products_path = repo / products_subdir
        bulk_copies = tuple(products_path / name for name in BULK_COPY_NAMES
                            if (products_path / name).is_file())
    entries = (canonical_entries(bulk, bulk_copies, repo)
                if args.bulk_mode == "canonical" else [])
    items = collect(repo, bulk, products_subdir, report_files,
                    include_bulk=args.bulk_mode == "raw")
    if not items:
        raise SystemExit("[FATAL] 没有可登记的产物（先跑 run_eval_v2.py）")

    lines = build_header(args.bulk_mode, sidecar if entries else None, repo, out)
    digests = {}
    for name, path in items:
        digest = md5_of(path)
        digests[name] = digest
        lines.append("%s  %s" % (digest, name))
    payload = "\n".join(lines) + "\n"
    # 🔴 **默认不许覆盖已存在且内容不同的清单** —— 冻结基线清单
    #    ``iqa_v2/output/manifest_v2.md5`` 是**已交付交付物**，
    #    一次「手滑的默认调用」把它改掉 ⇒ 证据链断裂且不可察觉（退出码仍是 0）。
    #    内容逐字相同 ⇒ 幂等重生成，允许；内容不同 ⇒ 必须显式 ``--force``。
    if out.exists() and not args.force:
        old = out.read_text(encoding="utf-8")
        if old != payload:
            raise SystemExit(
                "[FATAL] %s 已存在且内容与本次生成结果不同 ⇒ **拒绝覆盖**。\n"
                "        确认要改写它就显式加 ``--force``；"
                "若只是要生成深轮清单，请用 ``--products-subdir iqa_v2/output_deep "
                "--out iqa_v2/output_deep/manifest_deep.md5``。" % out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(payload, encoding="utf-8")
    print("[MANIFEST] %d 项 → %s" % (len(items), out))

    if entries:
        write_canonical_sidecar(sidecar, entries)
        print("[MANIFEST] 旁挂规范摘要清单 %d 项 → %s" % (len(entries), sidecar))
        print("[MANIFEST] 校验：python3 iqa_v2/canonical_digest.py verify %s" % sidecar)

    # ---- 自校验（**要有检出能力**，不能是恒真的形式检查）----
    failed = 0
    for name, path in items:
        if not path.is_file():
            print("[FAIL] 登记项已消失：%s" % name)
            failed += 1
        elif md5_of(path) != digests[name]:
            print("[FAIL] 登记项摘要已变：%s" % name)
            failed += 1
    # 🔴 登记名重复 ⇒ 同一个文件被登记两次（或两个文件同名）⇒ 门禁会漏检其中一个
    seen: set[str] = set()
    dupes = sorted({name for name, _ in items if name in seen or seen.add(name)})
    if dupes:
        print("[FAIL] 登记名重复：%s" % dupes)
        failed += len(dupes)
    if failed:
        raise SystemExit("[FAIL] 自校验失败 %d 项" % failed)
    print("[MANIFEST] 自校验 %d/%d 一致（登记名无重复）" % (len(items), len(items)))


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
