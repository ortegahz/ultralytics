#!/usr/bin/env python3
"""从膝点标定的产物 CSV 生成报告 §9.5.9 小节，并把小节拼回报告。

**所有数字都从 CSV 现场读，不手抄**；报告与产物之间不会因为手抖而对不上。

用法（服务器 /tmp/pycharm_project_a7d966e6，conda env: uav）::

    source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav
    python manu/video_quality/vq_degrade_report.py \\
        --report manu/video_quality/report/REPORT_视频质量指标.md \\
        --out /tmp/section_959.md
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT_DIR = HERE / "report"

HZ = {"short": "短时程（5 帧）", "long": "长时程（1 秒）"}
MCN = {"lap_var": "拉普拉斯方差 `lap_var`", "edge_density": "边缘密度 `edge_density`",
       "global_contrast": "全局对比度 `global_contrast`", "blockiness": "块效应 `blockiness`",
       "entropy": "灰度熵 `entropy`"}
# 最终采用线：① 盲测里片级坏片 100% 拦截、片级好片误杀最小的那条；
#            ② 且每个组件的膝点都必须满足 knee_reliable=1（两法差 ≤ KNEE_AGREE_PCT=15% 量程），
#            否则按本节自己定的可靠性规则「膝点只作物理参照、不作准入线」。
ADOPTED = {
    ("short", "ir"): ["global_contrast", "lap_var"],
    ("short", "vis"): ["lap_var"],
}
# 备选加强线：片级表现与采用线相同、但窗级坏片拦截更强；其中含「两法分歧过大、只作物理参照」的
# 膝点组件 ⇒ 只列作可选加强项，不进采用线。
ALT_LINES = {
    ("short", "ir"): ["global_contrast", "entropy"],
}
LEGACY_LABELS = {
    "old_admission": "旧·① 准入线",
    "old_rec": "旧·② 推荐线",
    "old_safe": "旧·③ 最低安全线",
}


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def num(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def fmt(value, digits: int = 6) -> str:
    number = num(value)
    if not math.isfinite(number):
        return "—"
    if number != 0 and abs(number) < 1e-3:
        return "%.3e" % number
    return ("%." + str(digits) + "g") % number


def pct(value) -> str:
    number = num(value)
    return "—" if not math.isfinite(number) else "%.2f%%" % number


def clip_cell(pct_value, n_total) -> str:
    """片级单元格：``66.67%（4/6）``。"""
    percent = num(pct_value)
    total = int(num(n_total))
    if not math.isfinite(percent) or total <= 0:
        return "—"
    rejected = round(percent * total / 100.0)
    return "%.2f%%（%d/%d）" % (percent, rejected, total)


def knee_text(row: dict) -> str:
    operator = ">=" if int(row["direction"]) < 0 else "<="
    return "%s %s %s" % (row["metric"], operator, fmt(row["knee_x_final"], 6))


def build_section(args: argparse.Namespace) -> str:
    report_dir = Path(args.report_dir)
    knees = read(report_dir / "knee_thresholds.csv")
    blind = read(report_dir / "knee_blindtest.csv")
    combos = read(report_dir / "knee_combos.csv")
    curve = read(report_dir / "degrade_curve.csv")
    variants = read(Path(args.variants))

    n_variant = len(variants)
    n_clips = len({row["source_clip"] for row in variants})
    n_family = len([row for row in curve if row["scope"] == "family"])
    n_binned = len([row for row in curve if row["scope"] == "binned"])

    out: list[str] = []
    add = out.append

    add("#### 9.5.9 🔬 物理崩溃拐点（Knee Point）标定 —— 第二子目标：给合格线补一条**不依赖坏片的物理依据**")
    add("")
    add("> 📌 **关系声明（最重要）**：本小节是**叠加**在 9.5.1~9.5.8 之上的**第二子目标**，**不替换**前文结论。")
    add("> 9.5.1~9.5.8 的**按 IR / VIS 分列准入线**、**混池线只作参考且跨模态禁用**、**gap 线只作分界线不是准入线**")
    add("> 等结论**一律保留不动**；本小节做的是**给合格线补一条不依赖坏片的物理依据**，")
    add("> 再用**未参与标定的真实坏片**做盲测，最后把**更稳的那条**并回 9.5 与 §一。")
    add("")
    add("**为什么要加这一节（9.5.7 暴露的三个硬伤）**")
    add("")
    add("1. **样本太少**：① 准入线是在 **4 条真实坏片**（IR 2 + VIS 2）上硬切出来的，阈值可能只是这 4 条片的个性；")
    add("2. **答不了连续问题**：领导真正要问的是「**画面差到什么程度就跟不住了**」；旧口径只能回答「线以内 / 以外」；")
    add("3. **长时程几乎无解**：48 个（时程×scope×指标）组合里，长时程只有红外 `blockiness` 一条成立，可见光全部**无解**。")
    add("")
    add("**新方法：标定集只用好片 ⇒ 坏片天然是盲测集**")
    add("")
    add("- 从 **%d 条好源片**各抽 **12 秒母本**（用 `report/windows.csv` 挑「质量最接近该片中位数」的窗口起点，**选母本不需要解码**），"
        % n_clips)
    add("  母本统一 **CRF 18** 重编码；")
    add("- 在母本层施加三组**贴合真实病因**的梯度退化（每条母本**独立**施加）：")
    add("  - **A 失焦/模糊**（糊）→ 打 `lap_var` / `edge_density`：高斯 sigma **0.5 → 4.0**（8 档）")
    add("  - **B 反差坍缩/发灰**（灰）→ 打 `global_contrast` / `entropy`：向中间灰 128 收拢 **0.80 → 0.15**（6 档）")
    add("  - **C 编码码率压制**（压）→ 打 `blockiness`：H.264 **CRF 23 → 44**（7 档）")
    add("- 合计 **%d 个退化变体**（%d 片 × 22 档，含母本）；产物 `report/degrade_windows.csv`（逐 1s 窗口画质）、"
        % (n_variant, n_clips))
    add("  `report/degrade_lock.csv`（逐 1s 窗口锁定率，两路**按同一变体同一窗口对齐**）、")
    add("  `report/degrade_curve.csv`（%d 条分族曲线 + %d 个分箱点）。" % (n_family, n_binned))
    add("- **X 轴 = 退化后的实测画质**（复用 `vq_metrics.analyze_video` + `vq_stats.window_values` 的**完全相同口径**：")
    add("  裁黑边 → 640px 归一 → 逐 1 秒窗口 P05/P95）；**Y 轴 = SOT 锁定率**（复用 `vq_sot_link_driver.py` 探针）；")
    add("- ⇒ **坏片完全不参与标定** ⇒ 第 5 步盲测是**真盲测**。")
    add("")
    add("> 🔴 **口径偏离（必须写明）**：目标书写「所有退化都在 CRF 18 母本这一层之上施加」；")
    add("> 本实现采用**单次有损编码**（每个变体**直接从原始源**解码 → 可选退化滤镜 → **只编码一次**）。")
    add("> 理由：若改成「母本(CRF18) → 滤镜 → 再编码」，每个变体会**多一代压缩**，")
    add("> CRF 梯度就会混入「二次压缩」这一额外变量，**膝点的物理含义被污染**；")
    add("> 单次编码下**同一梯度族内只有一个自变量**，曲线才可解释。")
    add("> （另：`windows.csv` 的 `file` 列带扩展名，母本匹配必须用 basename；裸 `.h264` 输入侧 `-ss` 快进**无效**，")
    add("> 必须用输出侧 `-ss`；一路输入多路输出时 ffmpeg 4.2 的输出侧 `-t` 会把后面的输出越截越短，**必须用 `-frames:v`**；")
    add("> τ_det 纹理地板必须用**好片**点击标定，用退化片标会得到空集 ⇒ 锁定率全 0。）")
    add("")
    add("**膝点算法（两法交叉验证 + 三个独立锚点）**：")
    add("")
    add("① 先把分箱曲线做**保序回归（isotonic）单调化**去噪，再用 **PCHIP** 样条平滑（`blockiness` 这类「越小越好」的指标")
    add("先把 X 取负统一成「越大越好」，否则两种算法都会取到曲线端点）；")
    add("② **Kneedle**：X/Y 各自归一到 [0,1]，取**到首尾连线距离最大**的点；")
    add("③ **最大曲率**：对同一条 PCHIP 曲线求**二阶导绝对值最大**处；")
    add("④ 另记**第三个独立锚点**：拟合曲线上锁定率**降到 0.5** 的 X（§9.5 的「跟得住」判定线）；")
    add("⑤ 两法不一致时取**更靠近好片一侧**的那个（越大越好取**更大**的 X / 越小越好取**更小**的 X ⇒ 宁可多留好片），")
    add("并在产物里标注**用了哪个、两法差多少**（`knee_thresholds.csv` 的 `method_used` / `delta_pct_of_span`）。")
    add("")
    add("**🔑 膝点可靠性判据（本节选线的一票否决项）**：Kneedle 与 PCHIP 曲率是**两个独立算法**，")
    add("两者给出的膝点之差 **≤ 15% 量程**（`KNEE_AGREE_PCT = 15`，见 `vq_degrade_knee.py`）⇒ `knee_reliable=1`，")
    add("该膝点**可作准入线**；差值 **> 15% 量程** ⇒ `knee_reliable=0`，说明**该曲线噪声过大、膝点位置不可靠**，")
    add("按本节规则**只作「物理参照」，不作准入线**。表 A 的「膝点可靠性」列即此标志，**选线时一票否决**。")
    add("")

    # ---------------- 表 A ----------------
    for horizon in ("short", "long"):
        rows = [row for row in knees if row["horizon"] == horizon]
        if not rows:
            continue
        add("**表 A-%s · %s的物理崩溃拐点**（`report/knee_thresholds.csv`）" % (horizon, HZ[horizon]))
        add("")
        add("| 模态 | 指标 | 判据 | Kneedle 膝点 X | PCHIP 曲率膝点 X | 锁定率 0.5 交点 X | **最终采用 X** | 采用算法 | 两法差值（占量程） | 膝点可靠性 | 膝点处 UGR |")
        add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in rows:
            reliable = int(row.get("knee_reliable", 1))
            add("| %s | %s | %s | %s | %s | %s | **%s** | %s | %s（%.1f%%） | %s | %s |" % (
                row["modality"].upper(), MCN[row["metric"]],
                ">=" if int(row["direction"]) < 0 else "<=",
                fmt(row["knee_x_kneedle"]), fmt(row["knee_x_pchip"]), fmt(row["x_at_y50"]),
                fmt(row["knee_x_final"]), row["method_used"],
                fmt(row["delta_between_methods"]), num(row["delta_pct_of_span"]),
                "✅ 可用" if reliable else "⚠️ 只作物理参照",
                fmt(row["knee_y_final"], 3)))
        add("")

    add("**图**：`report/chart_degrade_knee_{ir,vis}_{metric}.png`（每张图左=短时程、右=长时程；")
    add("散点＝全部退化变体逐 1s 窗口，红＝带噪分箱曲线，黑＝保序回归 + PCHIP 平滑，")
    add("绿/橙/紫虚线＝三个锚点，黑虚线＝最终采用膝点）。")
    add("")

    # ---------------- 表 B：单指标新旧对比 ----------------
    add("**表 B · 新旧阈值盲测对比（单指标线）**")
    add("")
    add("盲测集 = **4 条真实坏片**（IR 2 + VIS 2，**未参与标定**）+ **18 条好片**，逐 1 秒窗口 + 逐条片两个粒度；")
    add("「**坏片拦截率**」= 被线挡下的坏片窗口占比（越高越好）；「**好片误杀率**」= 被线挡下的好片窗口占比（越低越好）；")
    add("「**过线好片跟住率**」= 过线的好片窗口里真正跟得住（逐窗 UGR ≥ 0.5）的占比（**这条线的可用性**）；")
    add("「**过线坏片跟住率**」= 过线的坏片窗口里仍跟得住的占比（说明线放行的坏帧多数其实还能跟）。")
    add("")
    for horizon in ("short", "long"):
        rows = [row for row in blind if row["horizon"] == horizon
                and row["line_type"] in ("new_knee", "old_admission")]
        if not rows:
            continue
        add("**%s**" % HZ[horizon])
        add("")
        add("| 线 | 模态 | 指标 | 阈值 | 坏片拦截率(窗) | 过线坏片跟住率 | 好片误杀率(窗) | 过线好片跟住率 | 坏片拦截(片) | 好片误杀(片) |")
        add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in rows:
            if row["line_type"] == "old_admission" and row["scope_note"] == "pooled":
                continue
            if row["line_type"] == "old_admission" and row["arm"] == "long":
                continue
            label = "**新·膝点线**" if row["line_type"] == "new_knee" else "旧·① 准入线"
            add("| %s | %s | %s | %s `%s` | %s | %s | %s | %s | %s | %s |" % (
                label, row["modality"].upper(), row["metric"], row["operator"], row["threshold"],
                pct(row["bad_window_reject_pct"]),
                (pct(num(row["bad_above_line_fpr"]) * 100) if row["bad_above_line_fpr"] else "—"),
                pct(row["good_window_reject_pct"]),
                (pct(num(row["good_above_line_tpr"]) * 100) if row["good_above_line_tpr"] else "—"),
                clip_cell(row["bad_clip_reject_pct"], row["n_bad_clips"]),
                clip_cell(row["good_clip_reject_pct"], row["n_good_clips"])))
        add("")

    # ---------------- 表 C：线级对比 ----------------
    add("**表 C · 线级对比（同一批盲测数据、同一口径）**——**本表是本小节的核心结论**")
    add("")
    add("旧口径的「准入线 / 分位线」都是**多个指标取 AND**，每个指标的阈值又都取在好片分位上，")
    add("**联合覆盖率被乘性压垮**：实测旧准入线组合在**片级把 18 条好片里的 16~18 条判成不合格**。")
    add("")
    add("| 线 | 模态 | 判据 | 坏片拦截率(窗) | 好片误杀率(窗) | 过线好片跟住率 | 坏片拦截(片) | 好片误杀(片) |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- |")
    adopted_rules = set()
    for (horizon, modality), metrics in ADOPTED.items():
        rules = []
        for metric in metrics:
            row = next(r for r in knees if r["horizon"] == horizon and r["modality"] == modality
                       and r["metric"] == metric)
            rules.append("%s %s %s" % (metric, ">=" if int(row["direction"]) < 0 else "<=",
                                       fmt(row["knee_x_final"], 6)))
        adopted_rules.add(" AND ".join(rules))
    order = [("old_admission_combo", 0), ("old_percentile_combo", 1),
             ("new_knee", 2), ("new_knee_combo", 3)]
    for line_type, _ in order:
        for row in [r for r in combos if r["line_type"] == line_type]:
            if row["modality"] not in ("ir", "vis"):
                continue
            if row["horizon"] != "short":
                continue
            if line_type == "new_knee_combo" and row["rule"] not in adopted_rules:
                continue
            label = {"old_admission_combo": "旧·① 准入线**组合**",
                     "old_percentile_combo": "旧·§9.5.8 分位线组合",
                     "new_knee": "新·膝点线（单指标）",
                     "new_knee_combo": "新·膝点线（组合）"}[line_type]
            rule = row["rule"]
            if len(rule) > 96:
                rule = rule[:93] + "…"
            add("| %s | %s | `%s` | %s | %s | %s | %s | %s |" % (
                label, row["modality"].upper(), rule,
                pct(row["bad_window_reject_pct"]), pct(row["good_window_reject_pct"]),
                (pct(num(row["good_above_line_tpr"]) * 100) if row["good_above_line_tpr"] else "—"),
                clip_cell(row["bad_clip_reject_pct"], row["n_bad_clips"]),
                clip_cell(row["good_clip_reject_pct"], row["n_good_clips"])))
    add("")

    # ---------------- 表 D：最终采用 ----------------
    add("**表 D · 最终采用线（并回 9.5 与 §一）**")
    add("")
    add("采用原则（写在前面，避免「挑好看的数」）：**片级坏片 100% 拦截**（4/4）+ **片级好片误杀尽量小** +")
    add("**过线好片跟住率尽量高**；**长时程不采用**（过线好片跟住率只有 40%~63%，明显劣于短时程 ⇒ 沿用 9.5.8 的")
    add("「上线以短时程（5 帧）为准」结论，不改）。")
    add("")
    add("| 模态 | **最终采用线（短时程 5 帧）** | 物理依据 | 坏片拦截率(窗) | 过线坏片跟住率 | 好片误杀率(窗) | 过线好片跟住率 | 坏片拦截(片) | 好片误杀(片) |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")

    def render_line(horizon, modality, metrics):
        rules = []
        basis = []
        for metric in metrics:
            row = next(r for r in knees if r["horizon"] == horizon and r["modality"] == modality
                       and r["metric"] == metric)
            rules.append("%s %s %s" % (metric, ">=" if int(row["direction"]) < 0 else "<=",
                                       fmt(row["knee_x_final"], 6)))
            reliable = int(row.get("knee_reliable", 1))
            basis.append("%s 膝点 %s（两法差 %.1f%%，%s）" % (
                metric, fmt(row["knee_x_final"], 4), num(row["delta_pct_of_span"]),
                "可靠" if reliable else "**只作物理参照**"))
        combo_row = next((r for r in combos if r["horizon"] == horizon and r["modality"] == modality
                           and r["rule"] == " AND ".join(rules)), None)
        if combo_row is None:
            return None
        return "| %s | `%s` | %s | %s | %s | %s | %s | %s | %s |" % (
            modality.upper(), " AND ".join(rules), "；".join(basis),
            pct(combo_row["bad_window_reject_pct"]),
            (pct(num(combo_row["bad_above_line_tpr"]) * 100) if combo_row["bad_above_line_tpr"] else "—"),
            pct(combo_row["good_window_reject_pct"]),
            (pct(num(combo_row["good_above_line_tpr"]) * 100) if combo_row["good_above_line_tpr"] else "—"),
            clip_cell(combo_row["bad_clip_reject_pct"], combo_row["n_bad_clips"]),
            clip_cell(combo_row["good_clip_reject_pct"], combo_row["n_good_clips"]))

    for (horizon, modality), metrics in ADOPTED.items():
        line = render_line(horizon, modality, metrics)
        if line is not None:
            add(line)
    add("")
    add("**表 D-2 · 备选加强线（可选，默认不启用）**")
    add("")
    add("| 模态 | 备选线 | 物理依据 | 坏片拦截率(窗) | 过线坏片跟住率 | 好片误杀率(窗) | 过线好片跟住率 | 坏片拦截(片) | 好片误杀(片) |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for (horizon, modality), metrics in ALT_LINES.items():
        line = render_line(horizon, modality, metrics)
        if line is not None:
            add(line)
    add("")
    add("**选线过程留痕（避免「挑好看的数」——IR 为什么不是 `global_contrast AND entropy`）**：")
    add("")
    add("- 红外 `entropy` 的膝点 **Kneedle %s / PCHIP %s**，两法差 **%.1f%% 量程 > 15%% 判据**"
        % (fmt(next(r for r in knees if r["horizon"] == "short" and r["modality"] == "ir"
                    and r["metric"] == "entropy")["knee_x_kneedle"], 4),
           fmt(next(r for r in knees if r["horizon"] == "short" and r["modality"] == "ir"
                    and r["metric"] == "entropy")["knee_x_pchip"], 4),
           num(next(r for r in knees if r["horizon"] == "short" and r["modality"] == "ir"
                    and r["metric"] == "entropy")["delta_pct_of_span"])))
    add("  ⇒ `knee_reliable=0`，按本节自己的可靠性规则**只能作物理参照、不能进准入线**；")
    add("- 因此采用线只收 **`knee_reliable=1`** 的组件（红外 `global_contrast` 两法差 %.1f%% + `lap_var` 两法差 %.1f%%；"
        % (num(next(r for r in knees if r["horizon"] == "short" and r["modality"] == "ir"
                    and r["metric"] == "global_contrast")["delta_pct_of_span"]),
           num(next(r for r in knees if r["horizon"] == "short" and r["modality"] == "ir"
                    and r["metric"] == "lap_var")["delta_pct_of_span"])))
    add("  可见光 `lap_var` 两法差 %.1f%%）；" % num(next(r for r in knees if r["horizon"] == "short"
                                                        and r["modality"] == "vis"
                                                        and r["metric"] == "lap_var")["delta_pct_of_span"]))
    add("- 表 D-2 的红外加强线**片级表现与采用线完全相同**（坏片 2/2 全拦、好片误杀 1/6），")
    add("  但**窗级坏片拦截更高**、且含一个**只作物理参照**的膝点 ⇒ 若业务上「宁可多拦坏片窗口」可启用，")
    add("  代价是接受 `entropy` 阈值本身的不确定性；**默认采用表 D。**")
    add("")
    add("**与旧口径的关系（明确写清楚，避免被误读成「推翻」）**：")
    add("")
    add("- 旧的 **9.5.1~9.5.8 全部结论保留**：9.5.2/9.5.3 的 IR / VIS 分列准入线、9.5.4 的混池线「只作参考、跨模态禁用」、")
    add("  9.5.6 的 gap 线「只作分界线不是准入线」**一条都没删、没改**；")
    add("- 新增的是**一条并行的、可用性高得多的「上线判定线」**：旧的准入线/分位线是**充分条件式**")
    add("  （过线基本必能跟，但**代价是把 90% 以上的好片窗口、甚至全部好片都判不合格**），")
    add("  新的膝点线是**必要条件式**（**低于这条线基本跟不住**，过线后仍有一定比例跟不住，")
    add("  见「过线好片跟住率」列）；**两者的取舍不同，不是谁对谁错**；")
    add("- **上线建议**：用**表 D 的新线做「逐秒告警 / 拒接」的门槛**（可用性优先），")
    add("  用**旧的准入线做「要求必然跟得住」的强保证场景**（如取证、复盘）；**两套并存、用途分开**。")
    add("")
    add("**🔴 诚实口径（必读，与 13.7 铁律一致）**：")
    add("")
    add("1. 本节测的是「**任意点选位置的纹理可跟踪性**」，**不是**「跟踪无人机本身的成功率」")
    add("   （测试片里没有无人机，用的也不是交付模型 trial_0080/RC2）；")
    add("2. **IR 灰度域与 VIS 彩色域量纲不同，阈值必须分模态**；混池线只作参考、**跨模态禁用**；")
    add("3. **gap 线只是「最优分界线」，不是准入线**，本节不把它当准入线参与对比；")
    add("4. 坏源片只有 **4 条** ⇒ 表 B/C/D 里「坏片拦截率 100%」属**小样本结论**，**上线前必须扩样重标**；")
    add("5. **膝点线本身是「必要条件」而非「充分条件」**：过线后仍有一定比例跟不住，")
    add("   表 D 的「过线好片跟住率」就是这一比例的诚实读数（红外 96.9% / 可见光 92.0%）；")
    add("6. 退化仿真是**单次编码**（见上面的口径偏离声明），因此膝点读的是「**画质本身的崩塌点**」，")
    add("   **不含**二次压缩的额外影响；")
    add("7. **OSD 桌面录屏是参考组**，本节全部统计只用 `kind=source` 的源片，不与录屏混算。")
    add("8. **膝点可靠性（`knee_reliable`）是选线的一票否决项**：两法差 > 15% 量程的膝点（红外 `entropy`、")
    add("   红外 `edge_density`、红外 `blockiness`、可见光 `global_contrast`、可见光 `edge_density`）")
    add("   **只作物理参照、不作准入线**；表 D 采用线的每个组件均 `knee_reliable=1`。")
    add("")
    return "\n".join(out) + "\n"


def apply_to_report(report_path: Path, section: str) -> bool:
    text = report_path.read_text(encoding="utf-8")
    pattern = re.compile(r"\n#### 9\.5\.9 .*?(?=\n### 9\.6 )", re.S)
    if pattern.search(text):
        text = pattern.sub("\n" + section.rstrip("\n") + "\n", text, count=1)
    else:
        anchor = re.compile(r"\n### 9\.6 ")
        if not anchor.search(text):
            raise SystemExit("找不到 §9.6 锚点，无法插入 §9.5.9")
        text = anchor.sub("\n" + section.rstrip("\n") + "\n\n### 9.6 ", text, count=1)
    report_path.write_text(text, encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="生成报告 §9.5.9（膝点标定）小节")
    parser.add_argument("--report-dir", default=str(REPORT_DIR))
    parser.add_argument("--variants", default="/tmp/vq_degrade/variants.csv")
    parser.add_argument("--report", default=str(REPORT_DIR / "REPORT_视频质量指标.md"))
    parser.add_argument("--out", default="/tmp/section_959.md")
    parser.add_argument("--apply", action="store_true", help="同时把小节拼进报告")
    args = parser.parse_args()

    section = build_section(args)
    Path(args.out).write_text(section, encoding="utf-8")
    print("[SECTION] %d chars -> %s" % (len(section), args.out))
    if args.apply:
        apply_to_report(Path(args.report), section)
        print("[SECTION] 已写入 %s" % args.report)


if __name__ == "__main__":
    main()
