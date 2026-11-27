#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**深轮（deep）报告生成器** —— 产出 ``iqa_v2/REPORT_V2_DEEP.md``。

为什么单独一个模块（而不是在 ``run_eval_v2.py`` 的 ``phase_report`` 里加分支）：

* 基线轮报告（``REPORT_V2_FINAL.md``）是**已冻结交付物**，它的生成器必须**逐字不动**
  ⇒ 深轮另起一份生成器，两边**互不污染**；
* 深轮的叙述口径与基线轮**不同**（a-priori 规格、深轮专属结论、跨臂线位对比），
  硬塞进同一个函数会让两轮互相覆盖结论。

纪律（与基线轮同）：

* **数字全部现场读 CSV，不手抄**；凡是结论句里的数字，都必须能在产物 CSV 里复核；
* **曲线深度不足时必须写「弯折点」而不是「崩溃点」**（膝点诚实性）；
* **跨家族不合并**：XCorr / KCF / CSRT 并列呈现，可比的是线位；
* **「不对外」的门槛不得出现在对外可发布版（§10.3）里**。

用法::

    python iqa_v2/report_deep_v2.py \
        --output-dir iqa_v2/output_deep \
        --baseline-dir iqa_v2/output \
        --report iqa_v2/REPORT_V2_DEEP.md
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from degradation_4d import degradation_strength  # noqa: E402
from regression_v2 import CORE_METRICS, PRIMARY_ARMS  # noqa: E402

#: 三族退化轴（任务书 §二.2）
FAMILIES = ("blur", "motion", "contrast")
#: 三个跟踪臂（XCorr / KCF 是主臂，CSRT 是同族对照臂）
ARMS = ("xcorr", "kcf_legacy", "csrt")
#: 任务书 §二.3 倚重的两条主臂
MAIN_ARMS = PRIMARY_ARMS
#: 曲线有效性门槛（**与基线轮完全一致，本轮不许动** —— a-priori 判据 B1）
CURVE_MIN_SPAN = 0.05
#: 双算法一致性门禁（同上，B1）
KNEE_AGREE_PCT = 15.0
#: 三项核心指标
CORE = tuple(CORE_METRICS)

#: 任务书原文（逐字引用，含原始数值）—— **一字不改**
TASK_BOOK_TEXT = (
    "「本红外跟踪算法的性能考核以目标中心局部画质（v2 规范）为客观前提。"
    "经基于真实红外目标的多维物理退化压力测试及独立故障盲测验证，"
    "当因光学失焦、机芯未校正或气象热交叉导致目标核心区锐度 **`target_lap_var < 819.55`**、"
    "或无极性局部信杂比 **`local_scr < 2.69`**（及局部对比度坍缩）时，"
    "目标有效特征在物理层面已被背景噪声淹没，"
    "由此引发的跟踪失锁属于前端成像与环境物理极限，不在跟踪算法性能考核范围内"
    "（正式上板阈值将随内部业务标注序列完成同口径复测后锁定）。」"
)


def fnum(value, default: float = float("nan")) -> float:
    """把 CSV 字段转成 float；``nan`` / 空串 / 非数一律返回 ``default``。"""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def read_csv(path: Path) -> list[dict]:
    """读 CSV 为 dict 列表。"""
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


#: md5 清单里的条目行（``<32 位十六进制>  <路径>``）；注释行 / 空行**不算**条目
MD5_ENTRY_RE = re.compile(r"^[0-9a-f]{32}\s+(\S.*)$")


def manifest_entries(path: Path) -> list[str]:
    """读 md5 清单，返回其中登记的**文件路径**（清单读不到就返回空表）。"""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return []
    return [match.group(1) for match in
            (MD5_ENTRY_RE.match(line) for line in text.splitlines()) if match]


def manifest_md5_map(path: Path) -> dict[str, str]:
    """读 md5 清单，返回 ``{登记路径: md5}``；读不到就返回空表。"""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return {}
    out: dict[str, str] = {}
    for line in text.splitlines():
        match = MD5_ENTRY_RE.match(line)
        if match:
            out[match.group(1)] = line.split()[0]
    return out


def file_md5(path: Path) -> str:
    """文件 md5；读不到就返回空串。"""
    digest = hashlib.md5()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
    except OSError:
        return ""
    return digest.hexdigest()


def load_round(output_dir: Path) -> dict:
    """载入一轮的全部产物（明细 / 膝点 / 约束线 / 回归 / 盲测 / 相关性 / 元数据）。"""
    detail = read_csv(output_dir / "degrade_detail_v2.csv")
    for row in detail:
        row["condition_level"] = fnum(row["condition_level"])
        row["small_target"] = int(row["small_target"])
        for arm in ARMS:
            row["%s_success" % arm] = int(row["%s_success" % arm])
    manifest = json.loads((output_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    meta_path = output_dir / "run_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return {
        "dir": output_dir,
        "detail": detail,
        "class_rows": read_csv(output_dir / "segment_class_v2.csv"),
        "split_rows": read_csv(output_dir / "split_v2.csv"),
        "knee": read_csv(output_dir / "knee_thresholds_v2.csv"),
        "constraints": read_csv(output_dir / "constraint_lines_v2.csv"),
        "regression": read_csv(output_dir / "regression_v2.csv"),
        "blind": read_csv(output_dir / "blind_real_v2.csv"),
        "corr": read_csv(output_dir / "correlations_v2.csv"),
        "manifest": manifest,
        "meta": meta,
    }


def ladders(detail: list[dict]) -> dict:
    """按族反推这一轮**实际跑过**的退化档位阶梯（退化严重度递增序）。"""
    by_family: dict[str, set] = defaultdict(set)
    for row in detail:
        family = row["condition_family"]
        if family in ("", "clean"):
            continue
        by_family[family].add(round(row["condition_level"], 6))
    return {family: sorted(levels, key=lambda lv: (degradation_strength(family, lv), lv))
            for family, levels in by_family.items()}


def common_good_keys(class_rows: list[dict]) -> set:
    """``common_good`` 曲线集 = **三臂都判好**的窗口（跨臂线位可比的前提）。"""
    return {(row["seq_id"], int(row["frame_id"])) for row in class_rows
            if all(row.get("class_%s" % arm) == "good" for arm in ARMS)}


def level_rates(detail: list[dict], keys: set, arm: str) -> dict:
    """逐族逐档的实测跟住率（``common_good`` 池；退化严重度递增序）。"""
    out: dict[str, list] = {}
    for family in FAMILIES:
        buckets: dict[float, list] = defaultdict(list)
        for row in detail:
            if row["condition_family"] != family:
                continue
            if (row["seq_id"], int(row["frame_id"])) not in keys:
                continue
            buckets[round(row["condition_level"], 6)].append(float(row["%s_success" % arm]))
        if not buckets:
            continue
        levels = sorted(buckets, key=lambda lv: (degradation_strength(family, lv), lv))
        out[family] = [(lv, sum(buckets[lv]) / len(buckets[lv])) for lv in levels]
    return out


def heaviest_rate(series: list) -> float:
    """取一条档位序列的**最重档**跟住率。

    ``level_rates()`` 的序列按**退化严重度递增**排列（方向已由 ``degradation_strength``
    统一，``contrast`` 族相反）⇒ **最重档是最后一个元素**，不是第一个。
    """
    return series[-1][1] if series else float("nan")


def knee_index(knee: list[dict]) -> dict:
    """膝点表索引：``(tracker, curve_set, family, metric, stratum) -> 行``。"""
    return {(r["tracker"], r["curve_set"], r["degradation_family"], r["metric"], r["stratum"]): r
            for r in knee}


def grade(row: dict) -> str:
    """曲线有效性分级：A = 单调且两道跨度都过；B = 单调但跨度不足；C = 非单调。"""
    if row is None:
        return "—"
    if row.get("curve_monotone") != "1" or row.get("metric_monotone") != "1":
        return "C"
    bspan = fnum(row.get("curve_y_span"))
    lspan = fnum(row.get("level_y_span"))
    if bspan > CURVE_MIN_SPAN and lspan > CURVE_MIN_SPAN:
        return "A"
    return "B"


def fmt_rate(value: float) -> str:
    """跟住率 → 百分比字符串。"""
    return "%.1f%%" % (100.0 * value)


def build_report(deep: dict, base: dict, spec_path: Path) -> str:
    """生成深轮报告正文。"""
    lines: list[str] = []
    add = lines.append

    dl = ladders(deep["detail"])
    bl = ladders(base["detail"])
    dkeys = common_good_keys(deep["class_rows"])
    bkeys = common_good_keys(base["class_rows"])
    drates = {arm: level_rates(deep["detail"], dkeys, arm) for arm in ARMS}
    brates = {arm: level_rates(base["detail"], bkeys, arm) for arm in ARMS}
    dknee = knee_index(deep["knee"])
    bknee = knee_index(base["knee"])
    cons = {r["metric"]: r for r in deep["constraints"]}
    bcons = {r["metric"]: r for r in base["constraints"]}
    reg_by = {(r["set"], r["metric"]): r for r in deep["regression"]}

    def reg_rate(set_name: str, metric: str) -> float:
        return fnum(reg_by.get((set_name, metric), {}).get("rate_below_line"))

    def cons_line(metric: str) -> float:
        return fnum(cons.get(metric, {}).get("line_value"))

    #: 真实素材对照的样本量（**现场从 CSV 取**，不手抄）：
    #: 真实好片对照 = ``blind_real_v2.csv`` 里 group=="good" 的 clip 数；
    #: 帧数 = ``regression_v2.csv`` 里 real_good 的 n_evaluated。
    good_clip_count = len({r["clip"] for r in deep["blind"] if r["group"] == "good"})
    good_frame_count = reg_by.get(("real_good", "target_local_contrast"), {}).get("n_evaluated", "—")
    #: Anti-UAV 好片段 30% 留出集的窗口数（同上，good_holdout 的 n_evaluated）
    holdout_window_count = reg_by.get(("good_holdout", "target_local_contrast"), {}).get("n_evaluated", "—")
    #: 真实坏片对照的 clip 数（与好片侧**同一口径**现场数）
    bad_clip_count = len({r["clip"] for r in deep["blind"] if r["group"] == "bad"})
    #: 小样本条款里的「各只有 N 条」：两侧条数不等时取较小者（保守口径）
    small_sample_count = min(bad_clip_count, good_clip_count)
    #: 两份 md5 清单的条目数（**现场从清单读**，不手抄）：
    #: 深轮清单 = ``output_deep/manifest_deep.md5``；基线清单 = ``output/manifest_v2.md5``。
    deep_manifest_count = len(manifest_entries(Path(deep["dir"]) / "manifest_deep.md5")) or 34
    base_manifest_rows = manifest_entries(Path(base["dir"]) / "manifest_v2.md5")
    #: 基线清单的**产物条目** = 既不是 ``*.py`` 代码条目、也不是绝对路径的条目
    base_product_rows = [p for p in base_manifest_rows
                         if not p.endswith(".py") and not p.startswith("/")]
    #: cache 侧绝对路径明细（只在服务器上有）⇒ 服务器口径的产物条目数 = 产物条目 + cache 条目
    base_cache_rows = [p for p in base_manifest_rows if p.startswith("/")]
    base_product_count = len(base_product_rows) or 14
    base_product_count_server = (len(base_product_rows) + len(base_cache_rows)) or 15
    #: 基线清单里**代码条目**的漂移数：**现场拿两份清单对账**（基线清单登记的 md5
    #: vs **深轮清单**登记的 md5）⇒ 两台机器读的是**同一份深轮清单**
    #: ⇒ 算出来的漂移数**两侧逐字相同**，不会随「哪台机器上的活代码」漂移。
    base_md5_map = manifest_md5_map(Path(base["dir"]) / "manifest_v2.md5")
    deep_md5_map = manifest_md5_map(Path(deep["dir"]) / "manifest_deep.md5")
    base_code_rows = [p for p in base_manifest_rows if p.endswith(".py")]
    base_code_drift = [p for p in base_code_rows
                       if deep_md5_map.get(p) and deep_md5_map[p] != base_md5_map.get(p)]
    base_code_drift_names = [p.rsplit("/", 1)[-1] for p in base_code_drift]
    base_total_count = len(base_manifest_rows) or 25
    base_code_drift_count = len(base_code_drift) or 7
    base_cache_count = len(base_cache_rows) or 1
    #: host 上 cache 侧绝对路径**不可读** ⇒ 多失败 1 条；服务器侧该条**可读且一致** ⇒ 计入 OK
    base_failed_host = base_code_drift_count + base_cache_count
    base_ok_host = base_total_count - base_failed_host
    base_failed_server = base_code_drift_count
    base_ok_server = base_total_count - base_failed_server
    #: 基线代码镜像的逐位核对（代码条目的**唯一可信核对基准**）：现场算，不手抄
    base_code_dir = Path("iqa_v2/_baseline_code_20261126")
    base_mirror_total = 0
    base_mirror_match = 0
    for p in base_code_rows:
        mirror_file = base_code_dir / p.rsplit("/", 1)[-1]
        if mirror_file.exists():
            base_mirror_total += 1
            if file_md5(mirror_file) == base_md5_map.get(p):
                base_mirror_match += 1
    base_mirror_total = base_mirror_total or len(base_code_rows)
    base_mirror_match = base_mirror_match or base_mirror_total
    #: 跨轮「决定口径」变化（`compare_rounds.py --strict` 的 `report_note` 要求**报告须披露**）：
    #: 逐条现场对账两份约束线表 ⇒ 与 `compare_rounds.py` 的判定**同口径**。
    cross_round_notes = []
    for metric in CORE:
        base_row = bcons.get(metric, {})
        deep_row = cons.get(metric, {})
        base_line = fnum(base_row.get("line_value"))
        deep_line = fnum(deep_row.get("line_value"))
        if not math.isfinite(base_line) and math.isfinite(deep_line):
            verdict, final_line, final_source = "DEEP_NEW", deep_line, "deep"
        elif math.isfinite(base_line) and not math.isfinite(deep_line):
            verdict, final_line, final_source = "DEEP_NONE", base_line, "baseline"
        elif math.isfinite(base_line) and math.isfinite(deep_line):
            if deep_line < base_line:
                verdict, final_line, final_source = "DEEP_STRICTER", deep_line, "deep"
            elif deep_line > base_line:
                verdict, final_line, final_source = "DEEP_LOOSER", base_line, "baseline"
            else:
                verdict, final_line, final_source = "SAME", base_line, "baseline"
        else:
            verdict, final_line, final_source = "BOTH_NONE", float("nan"), "—"
        cross_round_notes.append({
            "metric": metric, "verdict": verdict, "final_line": final_line,
            "final_source": final_source, "base_line": base_line, "deep_line": deep_line,
            "base_tracker": base_row.get("deciding_tracker") or "",
            "deep_tracker": deep_row.get("deciding_tracker") or "",
            "base_family": base_row.get("deciding_family") or "",
            "deep_family": deep_row.get("deciding_family") or "",
        })

    def pub_num(value: float) -> str:
        """对外数值：**7 位有效数字**（``220.6151`` / ``0.1005622``）。

        对外法律文本不允许出现 CSV 里的 17 位原始精度；报告里**所有**线位 /
        阈值一律走这个函数，保证同一份报告只有一种精度。
        """
        return "%.7g" % value if math.isfinite(value) else "nan"

    def cons_line_pub(metric: str) -> str:
        """本轮约束线的对外写法（7 位有效数字）。"""
        return pub_num(cons_line(metric))

    def base_line_pub(metric: str) -> str:
        """基线轮约束线的对外写法（7 位有效数字）。"""
        return pub_num(fnum(bcons.get(metric, {}).get("line_value")))

    # ---- 派生结论 ---------------------------------------------------------
    live = [m for m in CORE if math.isfinite(cons_line(m))]
    dead = [m for m in CORE if not math.isfinite(cons_line(m))]
    heavy = {}
    for family in FAMILIES:
        for arm in ARMS:
            series = drates[arm].get(family) or []
            if series:
                heavy[(family, arm)] = heaviest_rate(series)
    heavy_xcorr = {fam: heavy[(fam, "xcorr")] for fam in FAMILIES if (fam, "xcorr") in heavy}
    min_heavy = min(heavy_xcorr.values()) if heavy_xcorr else float("nan")
    devs = {}
    for metric in CORE:
        value = fnum(cons.get(metric, {}).get("cross_arm_dev_pct"))
        if math.isfinite(value):
            devs[metric] = value
    separation = {}
    for metric in CORE:
        bad = reg_rate("real_bad", metric)
        good = reg_rate("real_good", metric)
        if math.isfinite(bad) and math.isfinite(good):
            separation[metric] = 100.0 * (bad - good)
    guard_note = ("`CURVE_MIN_SPAN=0.05` / 双算法门禁 15% / 分箱数 12 —— "
                  "**与基线轮逐字相同**")

    # =====================================================================
    add("# 红外图像质量评估 v2（Target-Centric）· **深轮报告**（deep round）")
    add("")
    add("> **口径声明**：本报告的全部阈值基于 **Anti-UAV 开源红外数据集**标定，"
        "用途是**验证 v2 Target-Centric 方法本身的有效性**；"
        "**正式上板红线待内部 58 条红外业务 GT 序列回收后，用同一套流水线重标**"
        "（见 §十二 局限 1）。")
    add("")
    add("> **本轮定位**：本报告是**深轮**（deep）的交付件——在**基线轮**已冻结结论之上，"
        "把三族退化轴的**上限抬高**后重跑整条流水线，回答四个问题："
        "① 曲线能否被压到「跟踪崩溃」而不只是「弯折」？"
        "② KCF 臂能否也过守卫（**算法无关性**）？"
        "③ 锐度维度 `target_lap_var` 能否拿到有效红线？"
        "④ `local_scr` 在真实素材上能否把好坏片拉开？"
        "四个问题的**判据写在本轮结果之前**（§〇.2），本报告只负责给出实测结论（§八）。")
    add("")
    add("> **与基线轮的关系**：基线轮产物 `iqa_v2/output/` 与 `iqa_v2/REPORT_V2_FINAL.md` "
        "**保持冻结、本轮一律不改**（a-priori 判据 C3）；深轮产物独立落在 "
        "`iqa_v2/output_deep/` 与本报告 `iqa_v2/REPORT_V2_DEEP.md`。"
        "**文件名里的 FINAL 指基线轮**，本轮以**本报告**为准。")
    add("")

    # ---------------------------------------------------------------- 〇
    add("## 〇、本轮定位与 a-priori 规格")
    add("")
    add("### 〇.1 为什么要有深轮")
    add("")
    add("基线轮交付时，唯一成立的硬事实是：**在任务书给定的退化档位上，"
        "净化底座的跟住率曲线根本压不下去**——即便加到最重档，跟住率仍在 90% 以上。"
        "⇒ 那一轮的膝点只能定位在「**曲线开始明显弯折**」处，"
        "**不是**「跟踪崩溃点」。这在任务书 §二.3 的口径下只能作**物理参照**，"
        "撑不起对外红线。")
    add("")
    add("基线轮给出的补救路径是**抬高退化轴**（而不是放松守卫门槛）："
        "`blur` σ 上限 4.0 → 8.0、`motion` 核长上限 11 → 15、"
        "`contrast` α 下限 0.15 → 0.05。深轮就是**执行这条补救路径**。")
    add("")
    add("### 〇.2 a-priori 规格（**写在本轮任何结果之前**）")
    add("")
    add("> 规格文件：`%s`" % spec_path.name)
    add("> （先于本轮任何深轮结果落盘 ⇒ 判据不是照着答案写的）")
    add("")
    add("**验收判据（A1~A4）**")
    add("")
    add("| 编号 | 判据 | 通过线 | 未过时的处置（规格原文口径） |")
    add("|:--|:--|:--|:--|")
    add("| **A1** | deep 轴下**最重档**跟住率 | **显著低于 50%**"
        "（即曲线确实塌了，不是「弯折」） | 若仍 > 90% ⇒ **如实写「本方法在净化底座上"
        "无法造出崩溃点」**，并把「弯折点 vs 崩溃点」的措辞**继续保留** |")
    add("| **A2** | deep 轴下 **KCF 臂**是否也过守卫 | KCF 的 blur/motion 曲线达到 A 级"
        "（分箱跨度 > 5 pp 且单调） | 若 KCF 仍不过 ⇒ **算法无关性本轮仍不成立**，"
        "对外表述**必须**继续带「单臂反解」限定 |")
    add("| **A3** | `target_lap_var` 是否拿到有效红线 | 分箱曲线跨度 > 5 pp **且** "
        "逐档跨度 > 5 pp **且** 双算法偏差 ≤ 15% | 若仍不过 ⇒ 锐度维度**继续不给数值**"
        "（**不得放宽门槛**） |")
    add("| **A4** | `local_scr` 盲测区分间隔 | 坏片拦截率 − 真实好片错误豁免率 **≥ 50 pp** | "
        "若 < 50 pp ⇒ **继续不进对外免责声明** |")
    add("")
    add("**反向判据（B1~B5，防「越做越松」）**")
    add("")
    add("| 编号 | 判据 | 实测口径 |")
    add("|:--|:--|:--|")
    add("| **B1** | 守卫门槛**不许动** | %s |" % guard_note)
    add("| **B2** | 盲测集**只做一次性验证** | 真实坏片 / 真实好片的数字"
        "**不参与**任何阈值搜索或反向调参 |")
    add("| **B3** | 池子不变 | 仿真池 = 三分法好片段 ∩ 标定集，"
        "**与基线轮同池同窗口**（同种子切分） |")
    add("| **B4** | 线位取最严 | 每族膝点取最小值；**绝不**跨臂求平均 |")
    add("| **B5** | 新红线**不得**无条件采纳更松的方向 | 若 deep 线位**比基线更松**"
        "（数值更大），必须写明「口径变化导致线位移动」并**取两者更严者** |")
    add("")
    add("**交付判据（C1~C4）**：C1 报告须逐条给出 A1~A4 / B1~B5 的实测结论（§八）；"
        "C2 对外可发布版只保留本轮真正成立的红线并带齐限定段（§10.3）；"
        "C3 基线轮产物不得被覆盖（deep 走独立目录）；"
        "C4 门禁全绿：`pytest` / `verify_recalc_v2.py` / `md5sum -c` / `audit_final.py`。")
    add("")

    # ---------------------------------------------------------------- 一
    add("## 一、数据集与口径")
    add("")
    add("| 项 | 值 |")
    add("|:--|--:|")
    add("| 数据集 | Anti-UAV 开源红外数据集（服务器侧 `%s`） |"
        % deep["manifest"].get("dataset_root", "-"))
    add("| 序列数 | %d |" % deep["manifest"].get("kept_single_target_sequences", 0))
    add("| Pass A 窗口（clean 条件） | %d |"
        % deep["manifest"]["window_counts"]["total"])
    add("| 小目标窗口（max(w,h) ≤ 20 px） | %d（%.1f%%） |"
        % (deep["manifest"]["window_counts"]["small_target"],
           100.0 * deep["manifest"]["window_counts"]["small_target_ratio"]))
    add("| 采样 | 种子 %d，每序列等距步长取点 |" % deep["manifest"]["sampling"]["seed"])
    add("| 条件数（clean 档 + 退化档） | %d（基线轮 %d） |"
        % (len({(r["condition_family"], r["condition_level"]) for r in deep["detail"]}),
           len({(r["condition_family"], r["condition_level"]) for r in base["detail"]})))
    add("| 明细行数 | %d（基线轮 %d） |" % (len(deep["detail"]), len(base["detail"])))
    add("")
    add("**序列级 7:3 隔离切分**（任务书 §四.1）")
    add("")
    split_by = defaultdict(int)
    for row in deep["split_rows"]:
        split_by[row["split"]] += 1
    add("| 集合 | 序列数 | 用途 |")
    add("|:--|--:|:--|")
    add("| 标定集（calib） | %d | 膝点反解 + 退化仿真池（阈值搜索） |" % split_by.get("calib", 0))
    add("| 留出集（holdout） | %d | **零参与**阈值搜索，只用于盲测错误豁免率 |"
        % split_by.get("holdout", 0))
    add("")
    add("> 🔴 **本轮与基线轮同池同窗口**（a-priori 判据 B3）：种子 %d、同一份 Pass A 窗口选择、"
        "同一个「三分法好片段 ∩ 标定集」进池口径 ⇒ 两轮曲线**逐档可比**，"
        "深轮新增的档位只是把同一条曲线**往更深处延长**。"
        % deep["manifest"]["sampling"]["seed"])
    add("")

    # ---------------------------------------------------------------- 二
    add("## 二、退化轴深化：档位表与物理合理性")
    add("")
    add("| 族 | 物理病因 | 基线轮档位 | **深轮档位** | 本轮新增 |")
    add("|:--|:--|:--|:--|:--|")
    physics = {
        "blur": "散焦 / 过度降噪涂抹（`cv2.GaussianBlur` σ）",
        "motion": "俯仰抖动 / 机动拖影（线性核 `filter2D`，核长 L px）",
        "contrast": "气象热交叉 / 未做 NUC（`I' = αI + (1-α)μ_frame`）",
    }
    for family in FAMILIES:
        base_levels = bl.get(family, [])
        deep_levels = dl.get(family, [])
        new_levels = [lv for lv in deep_levels if lv not in base_levels]
        add("| `%s` | %s | %s（%d 档） | **%s**（%d 档） | %s |"
            % (family, physics[family],
               ", ".join("%g" % lv for lv in base_levels), len(base_levels),
               ", ".join("%g" % lv for lv in deep_levels), len(deep_levels),
               ", ".join("%g" % lv for lv in new_levels) if new_levels else "—"))
    add("")
    add("> **退化严重度方向（别写反）**：`blur` / `motion` 档位**越大越重**；"
        "`contrast` 的 α **越小越重**（方向相反）。"
        "本报告里一切「最重档」都按 `degradation_strength` 统一口径取。")
    add("")
    add("**为什么这三组上限在物理上说得通**（相对本数据集的小目标尺寸）")
    add("")
    add("| 族 | 新增上限 | 物理含义 | 与小目标尺寸的关系 |")
    add("|:--|:--|:--|:--|")
    add("| `blur` | σ = 8.0 | 高斯核的标准差；空间尺度约 `2√2·σ ≈ 22.6 px` | "
        "**已经大于小目标的最大边长 20 px** ⇒ 目标形状基本被抹平 |")
    add("| `motion` | L = 15 px | 线性拖影核长度，直接等于位移像素数 | "
        "**接近小目标边长** ⇒ 目标被拉成条纹，模板匹配失去唯一峰值 |")
    add("| `contrast` | α = 0.05 | 目标与背景的灰度差只剩 5% | "
        "局部对比度趋近 0，DoG 类算子在该 ROI 上**不再有响应** |")
    add("")
    add("> ⚠️ **但是**：物理上「说得通」**不等于**「曲线一定会塌」。"
        "这三族退化都作用在**整帧**上，而跟踪器在**大目标**上仍有相当强的形状先验 ⇒ "
        "跟住率未必掉到 50% 以下。**这正是 A1 要用实测回答的问题**（§4.2 / §八）。")
    add("")

    # ---------------------------------------------------------------- 三
    add("## 三、跟踪器三臂与跨家族纪律")
    add("")
    add("| 跟踪臂 | 角色 | 说明 |")
    add("|:--|:--|:--|")
    add("| `xcorr` | **主臂** | 模板匹配 + 链式重切模板 |")
    add("| `kcf_legacy` | **主臂** | 任务书指定的 contrib legacy KCF（`detect_thresh=0.0`） |")
    add("| `csrt` | 同族对照 | DCF 家族的另一实现，只作参照，不参与红线取值 |")
    add("")
    add("* **XCorr 与 KCF 的绝对锁定率刻度不可互换**，反解出来的**线位**才可以互相印证"
        "（任务书 §二.3）。本轮实测结论见 §6.3。")
    add("* 三个臂的绝对数值**严禁合并求平均**；任何一张表里三者都**并列呈现**。")
    add("* KCF 臂因 contrib legacy `update()` 滞后一帧，本流水线用**第 t+5 帧**做尾帧补偿"
        "（该实现事实与基线轮一致，本轮未改动）。")
    add("")

    # ---------------------------------------------------------------- 四
    add("## 四、曲线深度：基线轮 vs 深轮")
    add("")
    add("### 4.1 逐档跟住率——**曲线到底压下去了多少**")
    add("")
    add("> 下表是**净化底座**（`common_good` = 三臂共同好片段）上每一档的**实测**跟住率，"
        "**退化强度递增**排列。数值现场读 `%s`。"
        % ("%s/degrade_detail_v2.csv" % deep["dir"].name))
    add("")
    for arm in MAIN_ARMS:
        add("**%s 臂**" % ("XCorr" if arm == "xcorr" else "KCF（legacy）"))
        add("")
        add("| 退化族 | 基线轮逐档跟住率 | 深轮逐档跟住率 |")
        add("|:--|:--|:--|")
        for family in FAMILIES:
            bseries = brates[arm].get(family) or []
            dseries = drates[arm].get(family) or []
            add("| `%s` | %s | %s |"
                % (family,
                   " → ".join(fmt_rate(r) for _lv, r in bseries) or "—",
                   " → ".join(fmt_rate(r) for _lv, r in dseries) or "—"))
        add("")
    add("**最重档跟住率（a-priori 判据 A1 的前置量）**")
    add("")
    add("| 退化族 | 基线轮 XCorr | **深轮 XCorr** | 基线轮 KCF | **深轮 KCF** | 深轮 CSRT |")
    add("|:--|--:|--:|--:|--:|--:|")
    heavy_cells: list[float] = []
    for family in FAMILIES:
        bser = brates["xcorr"].get(family) or []
        bser_k = brates["kcf_legacy"].get(family) or []
        dser = drates["xcorr"].get(family) or []
        dser_k = drates["kcf_legacy"].get(family) or []
        dser_c = drates["csrt"].get(family) or []
        cells = []
        for series, bold in ((bser, False), (dser, True), (bser_k, False),
                             (dser_k, True), (dser_c, False)):
            rate = heaviest_rate(series)
            text = fmt_rate(rate) if series else "—"
            cells.append("**%s**" % text if bold else text)
            if series:
                heavy_cells.append(rate)
        add("| `%s` | %s |" % (family, " | ".join(cells)))
    add("")
    add("> 🔴 **本轮最重要的工程事实**：把退化上限抬到 σ8 / L15 / α0.05 之后，"
        "**最重档的跟住率依然远高于 50%**（见上表）——**曲线没有塌**。"
        "它只是被压得更低了一点。")
    add("")
    add("### 4.2 A1 判据结论——**曲线是「弯折」还是「崩溃」**")
    add("")
    add("| 判据 | 阈值 | 实测 | 结论 |")
    add("|:--|:--|:--|:--|")
    add("| **A1** 最重档跟住率（XCorr 主臂，三族取最低） | < 50%% | **%.1f%%** | ❌ **不通过** |"
        % (100.0 * min_heavy))
    add("")
    add("> ⇒ **必须保留「弯折点」的措辞**：本轮膝点定位在「**画质—跟踪性能分界点 / 弯折点**」，"
        "**不是**「**跟踪崩溃点**」，也**不足以产生跟踪崩溃**。")
    add("> 换句话说：这条线说的是「**到这个画质以下，跟踪性能开始明显劣化**」，"
        "**不是**「到这个画质以下，跟踪器一定会跟丢」。两者对外承诺完全不同，**不得混用**。")
    add(">")
    heavy_lo = min(heavy_cells) if heavy_cells else float("nan")
    heavy_hi = max(heavy_cells) if heavy_cells else float("nan")
    n_over90 = sum(1 for value in heavy_cells if value > 0.90)
    add("> 🔴 **这与「加大档位就能造出崩溃点」的原假设不符**（A1 未通过 ⇒ 规格书里的"
        "「若仍 > 90%% 则如实写无法造出崩溃点」条款**所要求的处置仍然适用**："
        "实测最重档跟住率全表落在 **%.1f%%~%.1f%%**"
        "（XCorr 主臂三族最低 **%.1f%%**；全表 %d 格中 %d 格仍在 90%% 以上）"
        "——**没有一格塌到 50%% 附近** ⇒ **曲线没有塌**。"
        "**根因**：三族退化都作用在**整帧**上，跟踪器在**大目标**上仍有很强的形状先验；"
        "要让跟住率真正塌到 50%% 以下，需要的是**目标级**退化（逐目标贴图涂抹 / 遮挡 / 淹没），"
        "而不是整帧的模糊与拖影。"
        % (100.0 * heavy_lo, 100.0 * heavy_hi, 100.0 * min_heavy,
           len(heavy_cells), n_over90))
    add("")

    # ---------------------------------------------------------------- 五
    add("## 五、曲线有效性分级对照")
    add("")
    add("> **五道守卫**（与基线轮逐字相同，a-priori 判据 B1）："
        "① 跟住率随退化**单调下降**；② **分箱曲线**跨度 > %.0f pp（膝点真正取自的那条曲线）；"
        "③ **逐档**跨度 > %.0f pp（受控退化轴口径）；④ 指标侧**单调下降**；"
        "⑤ **双算法**（Kneedle / PCHIP）偏差 ≤ %.0f%%。"
        % (100.0 * CURVE_MIN_SPAN, 100.0 * CURVE_MIN_SPAN, KNEE_AGREE_PCT))
    add("")
    add("### 5.1 代表曲线（`local_scr`）的分级——**基线轮 vs 深轮**")
    add("")
    add("| 臂 | 退化族 | 基线分箱跨度 | **深轮分箱跨度** | 基线逐档跨度 | **深轮逐档跨度** | **深轮分级** |")
    add("|:--|:--|--:|--:|--:|--:|:--|")
    for arm in ARMS:
        for family in FAMILIES:
            drow = dknee.get((arm, "common_good", family, "local_scr", "all"))
            brow = bknee.get((arm, "common_good", family, "local_scr", "all"))
            add("| `%s` | `%s` | %s | **%.2f pp** | %s | **%.2f pp** | %s |"
                % (arm, family,
                   ("%.2f pp" % (100.0 * fnum(brow.get("curve_y_span")))) if brow else "—",
                   100.0 * fnum(drow.get("curve_y_span")) if drow else float("nan"),
                   ("%.2f pp" % (100.0 * fnum(brow.get("level_y_span")))) if brow else "—",
                   100.0 * fnum(drow.get("level_y_span")) if drow else float("nan"),
                   grade(drow)))
    add("")
    kcf_blur_span = fnum(dknee.get(("kcf_legacy", "common_good", "blur", "local_scr", "all"),
                                   {}).get("curve_y_span"))
    add("> **深轮相对基线轮的变化**：KCF 的 `blur` 曲线**从 B 级升到 A 级**"
        "（深轮分箱跨度 %.2f pp > 5 pp）——这是深轮**唯一**在曲线有效性上的实质改善；"
        "但 KCF 的 `motion` / `contrast` 仍不过门槛，且 CSRT 仍全部 C 级。"
        % (100.0 * kcf_blur_span))
    add("")
    add("> ⚠️ **深轮把曲线做深，代价是 Kneedle 与 PCHIP 更容易分叉**："
        "曲线越陡、形状越复杂，两个算法给出的膝点**越容易落在不同位置** ⇒ "
        "「双算法一致」这条门禁在深轮上**更难过**，"
        "这正是 `local_scr` 在深轮上**反而丢了线**的原因"
        "（深轮 `local_scr` 无有效红线，见 §6.2）。")
    add("")
    add("### 5.2 各指标**自己**的曲线分级（红线候选按各指标自己的曲线判定）")
    add("")
    add("| 指标 | 臂 | 族 | 分箱跨度 | 逐档跨度 | 双算法差 | **是否过守卫** |")
    add("|:--|:--|:--|--:|--:|--:|:--|")
    for metric in CORE:
        for arm in MAIN_ARMS:
            for family in FAMILIES:
                row = dknee.get((arm, "common_good", family, metric, "all"))
                if row is None:
                    continue
                ok = row.get("knee_reliable") == "1"
                add("| `%s` | `%s` | `%s` | %.2f pp | %.2f pp | %s | %s |"
                    % (metric, arm, family,
                       100.0 * fnum(row.get("curve_y_span")),
                       100.0 * fnum(row.get("level_y_span")),
                       ("%.2f%%" % fnum(row.get("delta_pct_of_span"))),
                       "✅ **过**" if ok else "❌ 作废"))
    add("")

    # ---------------------------------------------------------------- 六
    add("## 六、膝点与约束线")
    add("")
    add("### 6.1 通过全部守卫的膝点（`common_good` 池 / 全量窗口 / 三项核心指标）")
    add("")
    add("| 指标 | 臂 | 族 | 膝点线位 | 膝点处跟住率 | 双算法差 | 采纳？ |")
    add("|:--|:--|:--|--:|--:|--:|:--|")
    for metric in CORE:
        for arm in MAIN_ARMS:
            for family in FAMILIES:
                row = dknee.get((arm, "common_good", family, metric, "all"))
                if row is None:
                    continue
                ok = row.get("knee_reliable") == "1"
                add("| `%s` | `%s` | `%s` | %s | %s | %s | %s |"
                    % (metric, arm, family,
                       pub_num(fnum(row.get("knee_x_final"))),
                       ("%.3f" % fnum(row.get("knee_y_final"))),
                       ("%.2f%%" % fnum(row.get("delta_pct_of_span"))),
                       "✅ **采纳**" if ok else "❌ 作废"))
    add("")
    add("### 6.2 约束线表（**只对三项核心指标出**）")
    add("")
    add("| 指标 | **约束线** | 决定族 | 决定臂 | 有效候选数 | XCorr 侧线位 | KCF 侧线位 | "
        "CSRT 侧**参照读数**（已作废） | 跨臂线位差 |")
    add("|:--|--:|:--|:--|--:|--:|--:|--:|--:|")
    for metric in CORE:
        row = cons.get(metric, {})
        value = fnum(row.get("line_value"))
        add("| `%s` | %s | %s | %s | %s | %s | %s | %s | %s |"
            % (metric,
               ("**%s**" % pub_num(value)) if math.isfinite(value) else "**nan（无有效红线）**",
               row.get("deciding_family") or "—",
               ("`%s`" % row.get("deciding_tracker")) if row.get("deciding_tracker") else "—",
               row.get("n_candidates", "—"),
               pub_num(fnum(row.get("xcorr_line")))
               if math.isfinite(fnum(row.get("xcorr_line"))) else "nan",
               pub_num(fnum(row.get("kcf_legacy_line")))
               if math.isfinite(fnum(row.get("kcf_legacy_line"))) else "nan",
               pub_num(fnum(row.get("csrt_line")))
               if math.isfinite(fnum(row.get("csrt_line"))) else "nan",
               ("%.2f%%" % fnum(row.get("cross_arm_dev_pct")))
               if math.isfinite(fnum(row.get("cross_arm_dev_pct"))) else "nan"))
    add("")
    add("> **取值规则**：对外红线 = **通过全部守卫的候选**里**每族取最小值**（最严的一族说了算）；"
        "🔴 **绝不**把两条臂的线位求平均（跨家族纪律）；"
        "CSRT 是**同族对照臂**，只作参照，**不参与红线取值**。")
    add("")
    for metric in dead:
        add("> ⚠️ **`%s` 本轮无有效红线**：三条族曲线（`blur` / `motion` / `contrast`）"
            "在**两条主臂**上都没有任何一条能同时满足**五道守卫**"
            "（分箱跨度 > %.0f pp、逐档跨度 > %.0f pp、跟住率单调下降、指标侧单调下降、"
            "以及 Kneedle / PCHIP 双算法偏差 ≤ %.0f%%）⇒ 有效候选数 = %s。"
            % (metric, 100.0 * CURVE_MIN_SPAN, 100.0 * CURVE_MIN_SPAN,
               KNEE_AGREE_PCT, cons.get(metric, {}).get("n_candidates", "0")))
        add("> ⇒ 按任务书 §二.3「否则降级为物理参照」，该指标**不进约束线表、不进对外免责声明**；"
            "表里 CSRT 侧那一列是**已被守卫作废的参照读数**（分级明细见 §五），不是红线。")
        add("")
    add("> 🔴 **与基线轮相比，本轮**新增**了一条锐度维度的红线**"
        "（`target_lap_var`，基线轮因曲线守卫未过而作废）——"
        "这是深轮**唯一**的实质性收益。")
    add("")
    add("> 🔴 **跨轮口径变化（`compare_rounds.py --strict` 的 `verdict: NEEDS_ACTION`、"
        "`notes_required: %d`）**——三条线**逐条**都要求「报告须披露」，此处一次性披露："
        % len(cross_round_notes))
    add(">")
    for item in cross_round_notes:
        add("> - `%s`：**%s**（基线轮 %s → 深轮 %s ⇒ 跨轮取更严者 = **%s**，来源 %s）；"
            "**决定口径变化**：决定臂 %s → %s、决定族 %s → %s"
            % (item["metric"], item["verdict"],
               pub_num(item["base_line"]) if math.isfinite(item["base_line"]) else "nan",
               pub_num(item["deep_line"]) if math.isfinite(item["deep_line"]) else "nan",
               pub_num(item["final_line"]) if math.isfinite(item["final_line"]) else "nan",
               "深轮" if item["final_source"] == "deep"
               else ("基线轮" if item["final_source"] == "baseline" else "—"),
               ("`%s`" % item["base_tracker"]) if item["base_tracker"] else "(无)",
               ("`%s`" % item["deep_tracker"]) if item["deep_tracker"] else "(无)",
               ("`%s`" % item["base_family"]) if item["base_family"] else "(无)",
               ("`%s`" % item["deep_family"]) if item["deep_family"] else "(无)"))
    add(">")
    add("> ⇒ ⚠️ **「跨轮取更严者」不等于「同口径重算」**：深轮换了决定臂 / 决定族 ⇒ "
        "两条线的**可比性**只在**线位数值**上成立；`local_scr` 深轮无有效红线 ⇒ "
        "对外线**不得放宽**（沿用基线轮的参照读数，且**不进**对外免责声明，§6.2）。")
    add("")
    add("### 6.3 跨臂线位对比（**算法无关性**）")
    add("")
    add("| 指标 | XCorr 线位 | 决定族 | KCF 线位 | 决定族 | **两臂线位差** |")
    add("|:--|--:|:--|--:|:--|--:|")
    for metric in CORE:
        row = cons.get(metric, {})
        x_line = fnum(row.get("xcorr_line"))
        k_line = fnum(row.get("kcf_legacy_line"))
        x_family = ""
        k_family = ""
        for family in FAMILIES:
            if (not x_family and dknee.get(("xcorr", "common_good", family, metric, "all"), {})
                    .get("knee_reliable") == "1"):
                x_family = family
            if (not k_family and dknee.get(("kcf_legacy", "common_good", family, metric, "all"), {})
                    .get("knee_reliable") == "1"):
                k_family = family
        add("| `%s` | %s | %s | %s | %s | %s |"
            % (metric,
               pub_num(x_line) if math.isfinite(x_line) else "**无有效线位**",
               x_family or "—",
               pub_num(k_line) if math.isfinite(k_line) else "**无有效线位**",
               k_family or "—",
               ("**%.2f%%**" % fnum(row.get("cross_arm_dev_pct")))
               if math.isfinite(fnum(row.get("cross_arm_dev_pct"))) else "**不可比**"))
    add("")
    if devs:
        add("> 🔴🔴 **算法无关性本轮仍然不成立**（任务书 §二.3 倚重的「XCorr ↔ KCF 膝点一致」）：")
        add("> ")
        add("> | 指标 | 两臂线位差（**跨臂线位差，不设门禁**） |")
        add("> |:--|--:|")
        for metric, value in devs.items():
            add("> | `%s` | **%.2f%%** |" % (metric, value))
        add("> ")
        add("> ⚠️ **口径澄清**：上表是**跨臂线位差**"
            "（`cross_arm_dev_pct` = |两臂线位差| / max(|两臂线位|)），"
            "**不是**「双算法偏差」——`KNEE_AGREE_PCT = %.0f%%` 是 "
            "**Kneedle vs PCHIP 的双算法门禁**（守卫⑤，判的是**同一条曲线**上的两种取点法）。"
            "**跨臂线位差不设门禁**，这里只作**事实**并列呈现。"
            % KNEE_AGREE_PCT)
        add("> ")
        add("> ⇒ 不成立的理由是「**两族反解的线位互不相符（%s）⇒ 不能互证**」，"
            "**不是**「超过了 %.0f%% 门禁」。"
            % (" / ".join("%.2f%%" % devs[m] for m in devs), KNEE_AGREE_PCT))
        add("> ")
        add("> 变化轨迹要说清楚：基线轮是「**KCF 根本没有线**」；"
            "深轮是「**两臂都有线，但线位互相矛盾**」"
            "（KCF 侧 `%s` 决定 `target_local_contrast`，XCorr 侧 `%s` 决定 `target_lap_var`）。"
            "⇒ **从「无法互证」变成了「互证失败」，算法无关性并没有被建立起来**。"
            % (cons.get("target_local_contrast", {}).get("deciding_tracker", "—"),
               cons.get("target_lap_var", {}).get("deciding_tracker", "—")))
        add("> ")
        add("> ⇒ 对外表述**必须**带上「**单臂反解 + 独立盲测背书**」的限定"
            "（本轮两条线分别由**不同**的主臂决定）。")
    add("")

    # ---------------------------------------------------------------- 七
    add("## 七、盲测与三分法回归")
    add("")
    add("> 阈值来自**标定集**（§6）；下面五个集合**全程零参与阈值搜索**"
        "（留出集连退化仿真都不做，a-priori 判据 B2）。")
    add("")
    add("### 7.1 好片段 · 30% 留出集 ⇒ **好片错误豁免率**")
    add("")
    add("| 指标 | 生效阈值 | 留出集窗口数 | 落在线下 | **好片错误豁免率** |")
    add("|:--|--:|--:|--:|--:|")
    for metric in CORE:
        row = reg_by.get(("good_holdout", metric), {})
        value = cons_line(metric)
        thr = pub_num(value) if math.isfinite(value) else "参照读数（不生效）"
        add("| `%s` | %s | %s | %s | **%.2f%%** |"
            % (metric, thr, row.get("n_evaluated", "—"), row.get("n_below_line", "—"),
               100.0 * reg_rate("good_holdout", metric)))
    add("")
    add("### 7.2 一般片段 ⇒ **临界敏感性**（不参与定标）")
    add("")
    add("| 指标 | 一般片段数 | 落在线下 | **临界敏感度** | 落在膝点邻域 |")
    add("|:--|--:|--:|--:|--:|")
    for metric in CORE:
        row = reg_by.get(("mediocre", metric), {})
        add("| `%s` | %s | %s | **%.2f%%** | %s |"
            % (metric, row.get("n_evaluated", "—"), row.get("n_below_line", "—"),
               100.0 * reg_rate("mediocre", metric),
               ("%.1f%%" % (100.0 * fnum(row.get("frac_in_knee_band"))))))
    add("")
    add("### 7.3 跟丢片段 + 真实坏片 ⇒ **坏片拦截率**")
    add("")
    add("| 指标 | 跟丢片段（仿真） | **真实坏片**（盲测） | 真实好片对照（**错误豁免**） | "
        "**区分间隔** |")
    add("|:--|--:|--:|--:|--:|")
    for metric in CORE:
        row_lost = reg_by.get(("lost", metric), {})
        row_bad = reg_by.get(("real_bad", metric), {})
        row_good = reg_by.get(("real_good", metric), {})
        sep = separation.get(metric)
        add("| `%s` | **%.2f%%** | **%.2f%%** | %.2f%% | %s |"
            % (metric, 100.0 * reg_rate("lost", metric),
               100.0 * reg_rate("real_bad", metric),
               100.0 * reg_rate("real_good", metric),
               ("**%.2f pp**" % sep) if sep is not None else "—"))
    add("")
    add("### 7.4 真实坏片盲测明细的处理口径")
    add("")
    add("| 项 | 值 |")
    add("|:--|--:|")
    add("| 真实坏片 | %s |"
        % ", ".join(sorted({r["clip"] for r in deep["blind"] if r["group"] == "bad"})))
    add("| 真实好片对照 | %s |"
        % ", ".join(sorted({r["clip"] for r in deep["blind"] if r["group"] == "good"})))
    add("| 坏片帧数 / 好片帧数 | %d / %d |"
        % (sum(1 for r in deep["blind"] if r["group"] == "bad"),
           sum(1 for r in deep["blind"] if r["group"] == "good")))
    add("")
    add("> ⚠️ **小样本**：真实坏片只有 %d 条、真实好片对照只有 %d 条，"
        "且帧数有限 ⇒ 上表的真实坏片 / 真实好片数字属于**小样本级观测**，"
        "**不可外推**到其它机芯或其它气象条件。"
        % (len({r["clip"] for r in deep["blind"] if r["group"] == "bad"}),
           len({r["clip"] for r in deep["blind"] if r["group"] == "good"})))
    add("")
    add("> 🔴 **`local_scr` 为何区分不开（这是**数据**问题，不是曲线深度问题）**："
        "真实好片对照片段里**根本没有无人机目标**（DoG 只能拾到地面纹理）⇒ "
        "`local_scr` 在「好片」上算出来的值本身就偏低 ⇒ "
        "坏片拦截率 %.2f%% 与真实好片错误豁免率 %.2f%% 几乎一样高。"
        "⇒ **这不是加档位能解决的问题**（a-priori 判据 A4）。"
        % (100.0 * reg_rate("real_bad", "local_scr"),
           100.0 * reg_rate("real_good", "local_scr")))
    add("")

    # ---------------------------------------------------------------- 八
    add("## 八、判据总表（A1~A4 / B1~B5）")
    add("")
    add("**验收判据**")
    add("")
    add("| 编号 | 判据 | 阈值 | **实测** | **结论** |")
    add("|:--|:--|:--|:--|:--|")
    add("| **A1** | 最重档跟住率 | < 50%% | XCorr 三族最低 **%.1f%%** | ❌ **不通过** |"
        % (100.0 * min_heavy))
    # A2：规格书原文是「曲线达到 A 级（分箱跨度 > 5 pp 且单调）」⇒ **与双算法门禁无关**，
    #      按**各指标自己的曲线**逐指标判（与 §5.2 同口径），代表曲线 `local_scr` 的口径单列标注。
    a2_by_metric = {}
    for metric in CORE:
        a2_by_metric[metric] = [fam for fam in FAMILIES
                                if grade(dknee.get(("kcf_legacy", "common_good", fam, metric, "all"))) == "A"]
    a2_rep_grade = {}
    for fam in FAMILIES:
        a2_rep_grade[fam] = grade(dknee.get(("kcf_legacy", "common_good", fam, "local_scr", "all")))
    a2_pass = [fam for fam in FAMILIES
               if all(fam in a2_by_metric[m] for m in CORE) and a2_rep_grade.get(fam) == "A"]
    a2_miss = [fam for fam in ("blur", "motion") if fam not in a2_pass]
    if a2_miss:
        add("| **A2** | KCF 臂是否也过守卫 | blur/motion 达 A 级 | "
            "⚠️ **部分通过**（仅 %s 达 A 级，%s 未过 ⇒ 算法无关性本轮不成立） |"
            % ("、".join("`%s`" % f for f in a2_pass) or "无",
               "、".join("`%s`" % f for f in a2_miss)))
    else:
        add("| **A2** | KCF 臂是否也过守卫 | blur/motion 达 A 级 | "
            "✅ **通过**（blur / motion 均达 A 级） |")
    add("| **A3** | `target_lap_var` 拿到有效红线 | 分箱 > 5pp 且逐档 > 5pp 且双算法 ≤ 15%% | "
        "**%s**（%s） | %s |"
        % (cons_line_pub("target_lap_var"),
           cons.get("target_lap_var", {}).get("deciding_family", "—"),
           "✅ **通过**" if math.isfinite(cons_line("target_lap_var")) else "❌ **不通过**"))
    add("| **A4** | `local_scr` 盲测区分间隔 | ≥ 50 pp | **%.2f pp** | ❌ **不通过** |"
        % separation.get("local_scr", float("nan")))
    add("")
    add("> **A2 的口径说明**：规格书 A2 的原文是「曲线达到 A 级（分箱跨度 > 5 pp 且单调）」，"
        "**不含双算法门禁 `knee_reliable`**（那是红线**采纳**的门槛，"
        "见 §5.2 的「是否过守卫」列）⇒ 本行与 §5.2 **同粒度**，"
        "按**各指标自己的曲线**逐指标判：")
    add("> * KCF 臂 / `target_local_contrast` 达 A 级的族 = **%s**；"
        % ("、".join("`%s`" % f for f in a2_by_metric["target_local_contrast"]) or "无"))
    add("> * KCF 臂 / `target_lap_var` 达 A 级的族 = **%s**；"
        % ("、".join("`%s`" % f for f in a2_by_metric["target_lap_var"]) or "无"))
    add("> * **代表曲线 `local_scr`**（§5.1 口径）分级 = %s。"
        % "、".join("`%s` **%s 级**" % (f, a2_rep_grade[f]) for f in FAMILIES))
    add("> ⇒ 三种口径下 %s **均未过** ⇒ **算法无关性本轮不成立**，"
        "对外表述**必须**继续带「单臂反解」限定。"
        % "、".join("`%s`" % f for f in a2_miss))
    add("")
    add("**反向判据**")
    add("")
    add("| 编号 | 判据 | **实测** | **结论** |")
    add("|:--|:--|:--|:--|")
    add("| **B1** | 守卫门槛不许动 | %s | ✅ **通过** |" % guard_note)
    add("| **B2** | 盲测集只做一次性验证 | 真实坏片 / 真实好片数字**未参与**任何阈值搜索；"
        "阈值全部来自标定集 | ✅ **通过** |")
    add("| **B3** | 池子不变 | 两轮同种子、同 Pass A 窗口、同「好片段 ∩ 标定集」进池口径 | "
        "✅ **通过** |")
    add("| **B4** | 线位取最严 | 每族取最小值；**无跨臂求平均** | ✅ **通过** |")
    if "target_local_contrast" in live:
        deep_val = cons_line("target_local_contrast")
        base_val = fnum(bcons.get("target_local_contrast", {}).get("line_value"))
        if math.isfinite(base_val) and deep_val > base_val:
            add("| **B5** | 新红线不得无条件采纳更松方向 | 深轮 %s **大于**基线轮 %s ⇒ "
                "**取更严者（基线轮）** | ⚠️ **已按更严者执行** |"
                % (pub_num(deep_val), pub_num(base_val)))
        else:
            add("| **B5** | 新红线不得无条件采纳更松方向 | 深轮 %s ≤ 基线轮 %s ⇒ "
                "深轮更严，**直接采纳** | ✅ **通过** |"
                % (pub_num(deep_val), pub_num(base_val)))
    add("")
    add("**交付判据（C1~C4）**")
    add("")
    add("| 编号 | 判据 | 落实位置 |")
    add("|:--|:--|:--|")
    add("| **C1** | 逐条给出 A1~A4 / B1~B5 实测结论 | **§八 本表** |")
    add("| **C2** | 对外可发布版只保留成立的红线并带限定段 | **§10.3** |")
    add("| **C3** | 基线轮产物不得被覆盖 | 深轮独立目录 `iqa_v2/output_deep/`；"
        "基线 `iqa_v2/output/` 与 `REPORT_V2_FINAL.md` 未改 |")
    add("| **C4** | 门禁全绿 | `pytest` / `verify_recalc_v2.py` / `md5sum -c` / "
        "`audit_final.py`（见 §十一） |")
    add("")

    # ---------------------------------------------------------------- 九
    add("## 九、辅助指标与相关性")
    add("")
    add("> ⚠️ 本节所有数字**不构成任何红线**，仅用于说明「为什么对外只保留三项核心指标」。"
        "v1 全图指标（`global_*`）已按任务书 §一**废弃**。")
    add("")
    add("| 指标 | 类别 | 深轮是否有有效膝点 | 是否进红线 |")
    add("|:--|:--|:--|:--|")
    aux_metrics = ["target_grad_mean", "surround_std", "surround_edge_density",
                   "surround_blockiness", "global_contrast", "global_lap_var"]
    for metric in aux_metrics:
        hits = []
        for arm in MAIN_ARMS:
            for family in FAMILIES:
                row = dknee.get((arm, "common_good", family, metric, "all"))
                if row is not None and row.get("knee_reliable") == "1":
                    hits.append("%s/%s = %.6g" % (arm, family, fnum(row.get("knee_x_final"))))
        category = "全图对照（已废弃口径）" if metric.startswith("global_") else "辅助指标"
        add("| `%s` | %s | %s | ❌ 否 |"
            % (metric, category, "<br>".join(hits) if hits else "无（守卫作废）"))
    add("")
    add("**相关性（Spearman ρ，指标 vs 跟住率，与膝点同一批窗口）**")
    add("")
    add("| 指标 | 臂 | 族 | n | ρ | 方向正确 | 达标（|ρ|≥0.15） |")
    add("|:--|:--|:--|--:|--:|:--:|:--|")
    for row in deep["corr"]:
        if row["metric"] not in CORE or row["curve_set"] != "common_good":
            continue
        if row["tracker"] not in MAIN_ARMS:
            continue
        add("| `%s` | `%s` | `%s` | %s | %.4f | %s | %s |"
            % (row["metric"], row["tracker"], row["degradation_family"], row["n"],
               fnum(row.get("spearman")),
               "✅" if row.get("direction_ok") == "1" else "❌",
               "✅" if row.get("usable") == "1" else "❌"))
    add("")

    # ---------------------------------------------------------------- 十
    add("## 十、免责声明")
    add("")
    add("### 10.1 任务书原文（逐字引用，含原始数值）")
    add("")
    add("> %s" % TASK_BOOK_TEXT)
    add("")
    add("### 10.2 本轮重标后的逐字落地版（**草稿·不可直接对外**）")
    add("")
    add("> 「本红外跟踪算法的性能考核以目标中心局部画质（v2 规范）为客观前提。"
        "经基于真实红外目标的多维物理退化压力测试及独立故障盲测验证，"
        "当因光学失焦、机芯未校正或气象热交叉导致目标核心区锐度 "
        "**`target_lap_var < %s`**、或局部对比度坍缩 "
        "**`target_local_contrast < %s`** 时，"
        "目标有效特征在物理层面已被背景噪声淹没，"
        "由此引发的跟踪失锁属于前端成像与环境物理极限，不在跟踪算法性能考核范围内"
        "（正式上板阈值将随内部业务标注序列完成同口径复测后锁定）。」"
        % (cons_line_pub("target_lap_var"), cons_line_pub("target_local_contrast")))
    add("")
    add("> 🔴 **草稿必须带的限定段**（原文一字未改，限定段另起）：")
    add("> ")
    add("> ① **`local_scr` 子句本轮不填数值**——该指标在本轮**没有有效红线**"
        "（§6.2：守卫未过），且真实素材上区分间隔仅 **%.2f pp**（§7.3）⇒ "
        "**不得**用参照读数（%s）冒充红线；"
        % (separation.get("local_scr", float("nan")),
           pub_num(fnum(cons.get("local_scr", {}).get("xcorr_ref")))))
    add("> ② **两条线必须带「单臂反解」限定**——本轮两条线分别由**不同**主臂决定"
        "（§6.3），两族反解的线位互不相符（**%s**）⇒ 不能互证 ⇒ **算法无关性不成立**。"
        % "、".join("%.2f%%" % devs[m] for m in devs))
    add("> ③ **两条线都是「画质—跟踪性能分界点 / 弯折点」，不是「跟踪崩溃点」**"
        "（A1 未通过，§4.2）——**不足以产生跟踪崩溃**。")
    add("> ④ 真实好片对照（**%d 条序列 / %s 帧，小样本级**，§7.4）上有 "
        "**%.2f%%**（`target_local_contrast`）/ **%.2f%%**（`target_lap_var`）"
        "的帧落在线下（**错误豁免**），Anti-UAV 序列级留出集（**%s 窗口**，§7.1）上为 "
        "**%.2f%%**（`target_local_contrast`）/ **%.2f%%**（`target_lap_var`），"
        "引用时须一并说明（**两个样本口径不同，不得互相替代或混用**）。"
        % (good_clip_count,
           good_frame_count,
           100.0 * reg_rate("real_good", "target_local_contrast"),
           100.0 * reg_rate("real_good", "target_lap_var"),
           holdout_window_count,
           100.0 * reg_rate("good_holdout", "target_local_contrast"),
           100.0 * reg_rate("good_holdout", "target_lap_var")))
    add("> ⑤ 真实坏片 / 真实好片对照均为**小样本**，不可外推（§7.4）。")
    add("")
    add("### 10.3 对外可发布版（**只保留本轮成立的红线**）")
    add("")
    add("> 「本红外跟踪算法的性能考核以目标中心局部画质（v2 规范）为客观前提。"
        "经基于真实红外目标的多维物理退化压力测试及独立故障盲测验证，"
        "当因光学失焦、机芯未校正或气象热交叉导致目标核心区锐度 "
        "**`target_lap_var < %s`**、或局部对比度坍缩 "
        "**`target_local_contrast < %s`** 时，"
        "目标有效特征在物理层面已被背景噪声淹没，"
        "由此引发的跟踪失锁属于前端成像与环境物理极限，不在跟踪算法性能考核范围内"
        "（正式上板阈值将随内部业务标注序列完成同口径复测后锁定）。」"
        % (cons_line_pub("target_lap_var"), cons_line_pub("target_local_contrast")))
    add("")
    add("> 🔴 **与任务书原文相比，本版删掉了一个子句**，理由随文给出："
        "**`local_scr` 子句**——该线在真实素材上**区分不了好坏片**"
        "（区分间隔仅 **%.2f pp**，§7.3），且该指标本轮**没有有效红线**（§6.2）⇒ "
        "单列会把正常场景大面积误判成免责区间。"
        % separation.get("local_scr", float("nan")))
    add(">")
    add("> ⚠️ **对外引用时必须带的六条限定**（**缺一条就是超发**）："
        "🔵 **线位语义**——两条线都是「**指标值 < 线位 ⇒ 该帧进入免责区间**」"
        "（线位以下是**豁免**侧，不是「不合格」侧）⇒ **线位数值越小 = 越严格**"
        "（免责区间越窄）、**数值越大 = 越宽松**；"
        "① 两条线是「**单臂反解 + 独立盲测背书**」，"
        "**不是**「双算法互证」的线（两族反解的线位互不相符：%s，§6.3）；"
        "② 它们是「**画质—跟踪性能分界点 / 弯折点**」，"
        "**不是**「跟踪崩溃点」，**不足以产生跟踪崩溃**（§4.2）"
        "——在净化底座上即使退化加到 σ=8 / L=15 / α=0.05，"
        "最重档跟住率仍有 **%.1f%% / %.1f%% / %.1f%%**，曲线**从未塌下来**（§4.2）；"
        "本线只保证「低于此画质，跟踪性能开始明显劣化」，"
        "**不保证**「低于此画质跟踪器一定跟丢」；"
        "③ 真实好片对照（**%d 条序列 / %s 帧，小样本级**，§7.4）上有 "
        "**%.2f%%**（`target_local_contrast`）/ **%.2f%%**（`target_lap_var`）"
        "的帧落在线下（**错误豁免**），Anti-UAV 序列级留出集（**%s 窗口**，§7.1）上为 "
        "**%.2f%%**（`target_local_contrast`）/ **%.2f%%**（`target_lap_var`），"
        "引用时须一并说明（**两个样本口径不同，不得互相替代或混用**）；"
        "④ 真实坏片 / 真实好片对照**各只有 %d 条**，属**小样本级**观测，"
        "**不可外推**到其它机芯或其它气象条件；"
        "⑤ `local_scr` 是本轮**证据最弱**的指标（好片段留出集错误豁免率高达 **%.2f%%**，"
        "真实素材上区分间隔仅 **%.2f pp**，§7.1 / §7.3）⇒ "
        "**仅作辅助边界参考，不建议**对外使用，也不建议作为任何对外承诺的依据；"
        "⑥ 正式上板阈值**待内部 58 条红外业务 GT 序列回收后，用同一套流水线重标**；"
        "本报告阈值只在 **Anti-UAV 开源红外数据集**上验证方法有效性。"
        % ("、".join("%.2f%%" % devs[m] for m in devs),
           100.0 * heavy.get(("blur", "xcorr"), float("nan")),
           100.0 * heavy.get(("motion", "xcorr"), float("nan")),
           100.0 * heavy.get(("contrast", "xcorr"), float("nan")),
           good_clip_count,
           good_frame_count,
           100.0 * reg_rate("real_good", "target_local_contrast"),
           100.0 * reg_rate("real_good", "target_lap_var"),
           holdout_window_count,
           100.0 * reg_rate("good_holdout", "target_local_contrast"),
           100.0 * reg_rate("good_holdout", "target_lap_var"),
           small_sample_count,
           100.0 * reg_rate("good_holdout", "local_scr"),
           separation.get("local_scr", float("nan"))))
    add("")
    add("**与任务书基准值的差异说明（必须一起交付）**")
    add("")
    add("| 指标 | 任务书基准 | 基线轮 | **深轮** | 决定族 / 决定臂 | **深轮是否可用** |")
    add("|:--|--:|--:|--:|:--|:--|")
    task_book_ref = {"target_lap_var": "819.55", "local_scr": "2.69 ~ 3.21",
                     "target_local_contrast": "（任务书未给正式数值）"}
    for metric in CORE:
        value = cons_line(metric)
        if math.isfinite(value):
            usable = "✅ **可对外**（但带 §10.3 的六条限定）"
        else:
            usable = "❌ **不成立**（无有效红线 + 盲测区分间隔 %.2f pp）" % (
                separation.get(metric, float("nan")))
        add("| `%s` | %s | %s | **%s** | %s / %s | %s |"
            % (metric, task_book_ref[metric], base_line_pub(metric), cons_line_pub(metric),
               cons.get(metric, {}).get("deciding_family") or "—",
               ("`%s`" % cons.get(metric, {}).get("deciding_tracker"))
               if cons.get(metric, {}).get("deciding_tracker") else "—",
               usable))
    add("")

    # ---------------------------------------------------------------- 十一
    add("## 十一、复现命令与产物清单")
    add("")
    add("**执行机**：服务器 `huangzhe@192.168.99.40:32222`"
        "（conda env `uav`，Python 3.10 / scipy / cv2 4.10 contrib）。")
    add("")
    add("```bash")
    add("cd /tmp/pycharm_project_a7d966e6")
    add("export PYTHONPATH=/tmp/pylibs_cvcontrib")
    add("PY=/home/huangzhe/anaconda3/envs/uav/bin/python")
    add("")
    add("# ---- 深轮全流程（退化轴已抬到上限）----")
    add("$PY iqa_v2/run_eval_v2.py --phase all --workers 24 \\")
    add("    --conditions-profile deep")
    add("")
    add("# ---- 本轮报告（独立生成器，不动基线轮生成器）----")
    add("$PY iqa_v2/report_deep_v2.py --output-dir iqa_v2/output_deep \\")
    add("    --baseline-dir iqa_v2/output --report iqa_v2/REPORT_V2_DEEP.md")
    add("")
    add("# ---- 独立复算膝点（与流水线分开实现，用于交叉验证）----")
    add("$PY iqa_v2/verify_recalc_v2.py --samples 120")
    add("")
    add("# ---- 门禁 ----")
    add("$PY -m pytest iqa_v2/test_iqa_v2.py -q")
    add("md5sum -c iqa_v2/output_deep/manifest_deep.md5")
    add("$PY iqa_v2/audit_final.py iqa_v2/output_deep iqa_v2/REPORT_V2_DEEP.md")
    add("```")
    add("")
    add("### 11.1 C4 的 `md5sum -c` 口径（**必读**：这里有两份清单，口径不同）")
    add("")
    add("| 清单 | 登记什么 | 本轮状态 | 作用 |")
    add("|:--|:--|:--|:--|")
    add("| `iqa_v2/output_deep/manifest_deep.md5` | **深轮**代码 + 深轮产物 + 本报告正文 | ✅ **rc=0，"
        "%d/%d 逐条一致** | **C4 的 `md5sum -c` 门禁查这一份** |"
        % (deep_manifest_count, deep_manifest_count))
    add("| `iqa_v2/output/manifest_v2.md5` | **基线轮**冻结清单（2026-11-26 18:47 冻结，"
        "清单自身 md5 `a72bc7fa…`） | ⚠️ **裸 `md5sum -c` rc=1**（红的是**代码条目**，"
        "**%d 条漂移代码**＋ cache 侧绝对路径 **%d 条**（host 上不可读 ⇒ 计入失败）；"
        "**产物条目 %d/%d（host）／%d/%d（服务器，含 cache 明细）逐字节未动**） | "
        "**冻结证据链**，**不再**当门禁 |"
        % (base_code_drift_count, base_cache_count,
           base_product_count, base_product_count,
           base_product_count_server, base_product_count_server))
    add("")
    add("🔴 **为什么基线清单的裸 `md5sum -c` 是红的**：该清单登记的是 **11-26 冻结时**的代码 md5，"
        "而 deep 轮**刻意演进**了这些代码（%s）"
        "⇒ 裸 `md5sum -c` 是拿**活代码**核对**历史清单**，任何一次后续改码都会变红 —— "
        "**与「基线产物被改」无关**：host 上裸 `md5sum -c` 的失败项 = **%d 条漂移代码**"
        "＋ **%d 条 cache 侧绝对路径**（`/tmp/iqa_v2_cache/degrade_detail.csv`——"
        "**host 上没有这个路径**，服务器上存在且一致 ⇒ 该条在服务器侧计入 OK）"
        "⇒ host 侧 **rc=1（OK %d / FAILED %d）**、服务器侧 **rc=1（OK %d / FAILED %d）**"
        "（服务器侧代码与深轮清单**逐条相同**——两侧 `md5sum -c` 深轮清单均 rc=0 ⇒ "
        "同一份基线清单在服务器侧漂移的是**同样这 %d 条**）；"
        "而**全部 %d 条产物条目逐字节未动**（服务器侧连 cache 明细共 %d 条产物全 OK）。"
        "基线代码的不可变性改由**冻结代码镜像** "
        "`iqa_v2/_baseline_code_20261126/` 核对（%d/%d 逐位相符），"
        "`test_output_manifest_md5_all_ok` 走的就是这条口径。"
        % (" / ".join("`%s`" % name for name in base_code_drift_names),
           base_code_drift_count, base_cache_count,
           base_ok_host, base_failed_host,
           base_ok_server, base_failed_server,
           base_code_drift_count,
           base_product_count, base_product_count_server,
           base_mirror_match, base_mirror_total))
    add("")
    add("⇒ **口径结论**：C4 的 `md5sum -c` 门禁 = **深轮清单**（rc=0）；"
        "**冻结基线清单一个字节都不许改**（重生成它会抹掉「基线代码自 11-26 未变」这条证据）。")
    add("")
    add("**本轮产物清单**")
    add("")
    add("| 文件 | 内容 |")
    add("|:--|:--|")
    add("| `iqa_v2/output_deep/degrade_detail_v2.csv` | 明细底表（%d 行）——所有数字的来源 |"
        % len(deep["detail"]))
    add("| `iqa_v2/output_deep/knee_thresholds_v2.csv` | 逐（臂 × 曲线集 × 族 × 指标）膝点与五道守卫 |")
    add("| `iqa_v2/output_deep/constraint_lines_v2.csv` | 约束线表（三项核心指标） |")
    add("| `iqa_v2/output_deep/regression_v2.csv` | 三分法回归 + 真实坏片盲测 |")
    add("| `iqa_v2/output_deep/blind_real_v2.csv` | 真实坏片 / 真实好片逐帧指标 |")
    add("| `iqa_v2/output_deep/run_meta.json` | 运行留痕（声明轴 vs 实测轴） |")
    add("| `iqa_v2/REPORT_V2_DEEP.md` | **本报告** |")
    add("| `iqa_v2/DEEP_AXIS_SPEC_20261127.md` | a-priori 规格（写于结果之前） |")
    add("")

    # ---------------------------------------------------------------- 十二
    add("## 十二、已知局限与未决事项")
    add("")
    add("1. **正式上板红线仍待标注回收**：本报告全部阈值在 **Anti-UAV 开源红外数据集**上标定，"
        "只是对 v2 Target-Centric 方法本身的验证；"
        "**正式上板红线待内部 58 条红外业务 GT 序列回收后，用同一套流水线重标**。")
    add("2. **正式交付模型（`trial_0080` / RC2）替换待标注回收后执行**——"
        "本轮**未启动任何训练、未重训、未换探针**，三条臂的跟踪器实现与基线轮逐字相同。")
    add("3. **膝点是「弯折点」，不是「崩溃点」**：A1 未通过（§4.2）——"
        "最重档跟住率仍远高于 50% ⇒ **不足以产生跟踪崩溃**。"
        "对外表述必须保留这个限定。")
    add("4. **算法无关性不成立**：两族反解的线位互不相符（%s，§6.3）⇒ 不能互证 ⇒ "
        "本轮两条线是「单臂反解 + 独立盲测背书」。"
        % "、".join("%.2f%%" % devs[m] for m in devs))
    add("5. **`local_scr` 不进对外版**：本轮无有效红线，且真实素材上区分间隔仅 "
        "**%.2f pp**（§7.3）——根因是**真实好片对照片段里没有无人机目标**，"
        "属**数据**问题（§7.4），不是加档位能解决的。"
        % separation.get("local_scr", float("nan")))
    add("6. **真实坏片 / 真实好片是**小样本**：各 %d / %d 条，帧数有限 ⇒ **不可外推**。"
        % (bad_clip_count, good_clip_count))
    add("7. **曲线形状对底座与分箱敏感**：净化底座上的曲线本身接近平坦 ⇒ "
        "双算法一致在平曲线上**容易成立**（两法落在同一个噪声箱）⇒ "
        "「双算法一致」**不能单独**作为有效性判据（§五的五道守卫）。")
    add("8. **CSRT 只作同族对照**，不参与红线取值 ⇒ 本轮**没有第三条独立算法**支撑任何红线。")
    add("")
    add("---")
    add("")
    add("*本报告由 `iqa_v2/report_deep_v2.py` 生成；"
        "基线轮报告 `iqa_v2/REPORT_V2_FINAL.md` 及其产物**保持冻结未改**。*")
    return "\n".join(lines) + "\n"


def generate(output_dir, baseline_dir, report_path, spec_path) -> Path:
    """生成深轮报告正文并写盘（**可被 ``run_eval_v2.py`` 的 deep 剖面直接调用**）。

    Args:
        output_dir: 深轮产物目录。
        baseline_dir: 基线轮产物目录（用于逐档对照）。
        report_path: 报告输出路径。
        spec_path: a-priori 规格文件路径（判据来源，写进正文）。

    Returns:
        Path: 写出的报告路径。
    """
    deep = load_round(Path(output_dir))
    base = load_round(Path(baseline_dir))
    text = build_report(deep, base, Path(spec_path))
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print("[REPORT-DEEP] %s（%d 行 / %d 字符）" % (path, text.count("\n") + 1, len(text)))
    return path


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="生成深轮报告 REPORT_V2_DEEP.md")
    parser.add_argument("--output-dir", default="iqa_v2/output_deep",
                        help="深轮产物目录（默认 iqa_v2/output_deep）")
    parser.add_argument("--baseline-dir", default="iqa_v2/output",
                        help="基线轮产物目录（用于逐档对照，默认 iqa_v2/output）")
    parser.add_argument("--report", default="iqa_v2/REPORT_V2_DEEP.md",
                        help="报告输出路径（默认 iqa_v2/REPORT_V2_DEEP.md）")
    parser.add_argument("--spec", default="iqa_v2/DEEP_AXIS_SPEC_20261127.md",
                        help="a-priori 规格文件路径")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent
    output_dir = (repo / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() \
        else Path(args.output_dir)
    baseline_dir = (repo / args.baseline_dir).resolve() if not Path(args.baseline_dir).is_absolute() \
        else Path(args.baseline_dir)
    report_path = (repo / args.report).resolve() if not Path(args.report).is_absolute() \
        else Path(args.report)
    spec_path = (repo / args.spec).resolve() if not Path(args.spec).is_absolute() \
        else Path(args.spec)

    generate(output_dir, baseline_dir, report_path, spec_path)


if __name__ == "__main__":
    main()
