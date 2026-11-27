#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""deep 轮（深化退化轴）**验收判定器**——逐条机器判定 A1~A4 / B1~B5 / C1~C4。

为什么要有这个文件
------------------
本轮是「**先定规格、后看结果**」的 a-priori 复算：判据写死在
``iqa_v2/DEEP_AXIS_SPEC_20261127.md``（写于任何 deep 结果产生之前）。
验收必须是**可复算**的，而不是「看完报告点个头」。

因此本文件**只吃 CSV / 源码常量**，**不采信报告里的任何数字**：
所有结论都从 ``degrade_detail_v2.csv`` / ``knee_thresholds_v2.csv`` /
``constraint_lines_v2.csv`` / ``regression_v2.csv`` / ``blind_real_v2.csv``
现场重算，再回头去和报告正文对照（对照结果作为 D 系列**交叉核验**输出）。

用法
----
    python3 iqa_v2/acceptance_deep.py \
        --deep-dir iqa_v2/output_deep \
        --baseline-dir iqa_v2/output \
        --report iqa_v2/REPORT_V2_DEEP.md

退出码：默认 0（验收报告模式，任何判定都只是**记录事实**）；
加 ``--strict`` 时，只要存在非 PASS 的判定就返回 1（门禁模式）。

纯标准库实现：host 侧没有 numpy/cv2 也能跑。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import defaultdict

# --------------------------------------------------------------------------
# 规格书写死的期望值（来自 DEEP_AXIS_SPEC_20261127.md §一 / §二）
# --------------------------------------------------------------------------
FROZEN_CONSTANTS = {
    "CURVE_MIN_SPAN": 0.05,        # 守卫门槛：分箱曲线跨度 / 逐档跨度下限
    "KNEE_AGREE_PCT": 15.0,         # 双算法（Kneedle vs PCHIP）膝点偏差门限
    "N_BINS": 12,                  # 衰减曲线分箱数
    "GOOD_CLE_PX": 3.0,            # 三分法「好片段」CLE 门限
    "CALIB_RATIO": 0.7,            # 序列级 7:3 切分
    "SPLIT_SEED": 20261126,        # 切分种子
}
FROZEN_BLUR = (0.6, 1.0, 1.5, 2.2, 3.0, 4.0)
FROZEN_MOTION = (3, 5, 7, 9, 11)
FROZEN_CONTRAST = (0.80, 0.65, 0.50, 0.35, 0.25, 0.15)
#: 规格书 §1.1 写死的 **deep** 档位表（a-priori，改档位必须改规格书）
FROZEN_DEEP_BLUR = (0.6, 1.0, 1.5, 2.2, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
FROZEN_DEEP_MOTION = (3, 5, 7, 9, 11, 13, 15)
FROZEN_DEEP_CONTRAST = (0.80, 0.65, 0.50, 0.35, 0.25, 0.15, 0.10, 0.05)

CORE_METRICS = ("target_local_contrast", "target_lap_var", "local_scr")
MAIN_ARMS = ("xcorr", "kcf_legacy")          # 两条主臂
CONTROL_ARM = "csrt"                          # 同族对照臂，只作参照
ALL_ARMS = ("xcorr", "kcf_legacy", "csrt")

# A1：deep 轴下最重档跟住率必须「显著低于 50%」才算曲线真的塌了
A1_COLLAPSE_PCT = 50.0
# A4：盲测区分间隔（坏片拦截率 − 真实好片错误豁免率）门槛
A4_MIN_SEPARATION_PP = 50.0
# 退化族里「最重档」的方向（blur/motion 越大越重，contrast 越小越重）
FAMILY_HEAVIEST = {
    "blur": "max",
    "motion": "max",
    "contrast": "min",
}

PASS, FAIL, PARTIAL = "PASS", "FAIL", "PARTIAL"

#: **仓库根**（由脚本自身位置推导，**不依赖 CWD**）——
#: 判定器的所有相对路径（``iqa_v2/output_deep`` 等）都锚定到这里 ⇒
#: 在任何目录下调用都得到同一结果（否则换个 CWD 就「清单条目缺失」，门禁假红）。
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_path(path, base=None):
    """把相对路径锚定到**仓库根**（绝对路径原样返回）。

    Args:
        path (str): 待解析路径（相对仓库根，或已是绝对路径）。
        base (str | None): 锚定基准目录（默认 :data:`REPO_ROOT`）。

    Returns:
        str: 绝对路径。
    """
    if os.path.isabs(path):
        return path
    return os.path.join(base or REPO_ROOT, path)


# --------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------
def fnum(value):
    """把 CSV 里的字符串转成 float；空串 / nan / 不可解析一律回 nan。"""
    if value is None:
        return float("nan")
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none", "null"):
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def is_finite(value):
    return isinstance(value, float) and math.isfinite(value)


def fmt(value, digits=6):
    if not is_finite(value):
        return "nan"
    return ("%." + str(digits) + "g") % value


def read_csv(path):
    if not os.path.exists(path):
        return None
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def md5_of(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_source_constant(path, name):
    """从一个 .py 源文件里读出模块级常量（只支持数值字面量）。"""
    if not os.path.exists(path):
        return None
    pattern = re.compile(r"^%s\s*=\s*([0-9]*\.?[0-9]+)" % re.escape(name), re.M)
    match = pattern.search(open(path, encoding="utf-8").read())
    return float(match.group(1)) if match else None


def read_source_tuple(path, name):
    """读出模块级元组常量（浮点 / 整数列表）。"""
    if not os.path.exists(path):
        return None
    pattern = re.compile(r"^%s\s*=\s*\(([^)]*)\)" % re.escape(name), re.M)
    match = pattern.search(open(path, encoding="utf-8").read())
    if not match:
        return None
    items = []
    for token in match.group(1).split(","):
        token = token.strip()
        if not token:
            continue
        items.append(float(token))
    return tuple(items)


# --------------------------------------------------------------------------
# 上下文：把两轮的 CSV 读进来，并现场重算所有验收用到的量
# --------------------------------------------------------------------------
class Context:
    """所有判定的唯一数据源。**不读报告数字**（报告只用于 D 系列对照）。"""

    def __init__(self, deep_dir, baseline_dir, code_dir, frozen_code_dir=None,
                 baseline_code_dir=None):
        self.repo_root = REPO_ROOT
        self.deep_dir = resolve_path(deep_dir)
        self.baseline_dir = resolve_path(baseline_dir)
        self.code_dir = resolve_path(code_dir)
        self.frozen_code_dir = (resolve_path(frozen_code_dir) if frozen_code_dir
                                else os.path.join(self.code_dir, "server_out_final", "files"))
        #: **冻结基线轮代码镜像**（基线轮代码的逐字节拷贝）⇒ C3 的代码条目在这里核对。
        self.baseline_code_dir = (resolve_path(baseline_code_dir) if baseline_code_dir
                                  else os.path.join(self.code_dir, "_baseline_code_20261126"))

        self.deep_detail = read_csv(os.path.join(self.deep_dir, "degrade_detail_v2.csv")) or []
        self.base_detail = read_csv(os.path.join(self.baseline_dir, "degrade_detail_v2.csv")) or []
        self.deep_knee = read_csv(os.path.join(self.deep_dir, "knee_thresholds_v2.csv")) or []
        self.base_knee = read_csv(os.path.join(self.baseline_dir, "knee_thresholds_v2.csv")) or []
        self.deep_lines = read_csv(os.path.join(self.deep_dir, "constraint_lines_v2.csv")) or []
        self.base_lines = read_csv(os.path.join(self.baseline_dir, "constraint_lines_v2.csv")) or []
        self.deep_reg = read_csv(os.path.join(self.deep_dir, "regression_v2.csv")) or []
        self.base_reg = read_csv(os.path.join(self.baseline_dir, "regression_v2.csv")) or []
        self.deep_blind = read_csv(os.path.join(self.deep_dir, "blind_real_v2.csv")) or []
        self.base_blind = read_csv(os.path.join(self.baseline_dir, "blind_real_v2.csv")) or []
        self.deep_split = read_csv(os.path.join(self.deep_dir, "split_v2.csv")) or []
        self.base_split = read_csv(os.path.join(self.baseline_dir, "split_v2.csv")) or []
        self.deep_seg = read_csv(os.path.join(self.deep_dir, "segment_class_v2.csv")) or []
        self.base_seg = read_csv(os.path.join(self.baseline_dir, "segment_class_v2.csv")) or []
        self.deep_window = read_csv(os.path.join(self.deep_dir, "windows.csv")) or []
        self.base_window = read_csv(os.path.join(self.baseline_dir, "windows.csv")) or []
        self.deep_meta = read_csv  # 占位，真正的 meta 走 json
        self.meta = {}
        meta_path = os.path.join(self.deep_dir, "run_meta.json")
        if os.path.exists(meta_path):
            self.meta = json.load(open(meta_path, encoding="utf-8"))

        self._index()

    # -- 索引 ---------------------------------------------------------------
    def _index(self):
        self.class_index = {}
        for row in self.deep_seg:
            key = (row["seq_id"], int(row["frame_id"]))
            self.class_index[key] = row
        self.base_class_index = {}
        for row in self.base_seg:
            self.base_class_index[(row["seq_id"], int(row["frame_id"]))] = row

        # 基线轮自己的 common_good 池（两轮三分法行序不同，必须各自建索引）
        self.base_common_good = {
            key for key, row in self.base_class_index.items()
            if all(row.get("class_%s" % arm) == "good" for arm in ALL_ARMS)
        }

        # 三臂共同好片段（common_good）——曲线的跨臂可比池
        self.common_good = {
            key for key, row in self.class_index.items()
            if all(row.get("class_%s" % arm) == "good" for arm in ALL_ARMS)
        }
        # 进仿真池（标定集 ∩ 三分法好片段 class_ds == good）
        self.pool_ds = {
            key for key, row in self.class_index.items()
            if row.get("class_ds") == "good"
        }

        self.knee_index = {}
        for row in self.deep_knee:
            key = (row["tracker"], row["curve_set"], row["stratum"],
                   row["degradation_family"], row["metric"])
            self.knee_index[key] = row
        self.base_knee_index = {}
        for row in self.base_knee:
            key = (row["tracker"], row["curve_set"], row["stratum"],
                   row["degradation_family"], row["metric"])
            self.base_knee_index[key] = row

        self.line_index = {row["metric"]: row for row in self.deep_lines}
        self.base_line_index = {row["metric"]: row for row in self.base_lines}

        self.reg_index = {(row["set"], row["metric"]): row for row in self.deep_reg}
        self.base_reg_index = {(row["set"], row["metric"]): row for row in self.base_reg}

    # -- 现场重算：逐档跟住率（common_good 池）------------------------------
    def level_rates(self, detail=None, pool=None):
        """按 (族, 档位) 重算三臂跟住率，口径与 run_eval_v2._level_success_rates 一致。"""
        detail = self.deep_detail if detail is None else detail
        pool = self.common_good if pool is None else pool
        table = defaultdict(lambda: defaultdict(lambda: {"n": 0, "ok": defaultdict(int)}))
        for row in detail:
            if row.get("condition_family") == "clean":
                continue
            key = (row["seq_id"], int(row["frame_id"]))
            if key not in pool:
                continue
            family = row["condition_family"]
            level = fnum(row["condition_level"])
            if not is_finite(level):
                continue
            bucket = table[family][level]
            bucket["n"] += 1
            for arm in ALL_ARMS:
                bucket["ok"][arm] += int(row.get("%s_success" % arm) or 0)
        return table

    def pool_of(self, detail=None):
        """明细表里真正被仿真过的窗口集合（非 clean 行）。"""
        detail = self.deep_detail if detail is None else detail
        return {(row["seq_id"], int(row["frame_id"]))
                for row in detail if row.get("condition_family") != "clean"}

    # -- 曲线分级（复刻 §4.2 的四道守卫）-------------------------------------
    def grade(self, tracker, family, metric, curve_set="common_good", stratum="all"):
        row = self.knee_index.get((tracker, curve_set, stratum, family, metric))
        if row is None:
            return None
        curve_span = fnum(row.get("curve_y_span"))
        level_span = fnum(row.get("level_y_span"))
        mono = str(row.get("curve_monotone")).strip() in ("1", "True", "true")
        mmono = str(row.get("metric_monotone")).strip() in ("1", "True", "true")
        threshold = FROZEN_CONSTANTS["CURVE_MIN_SPAN"]
        # 与 run_eval_v2.curve_grade 同口径：**跟住率非单调 或 指标侧非单调 ⇒ C 级**
        if not mono or not mmono:
            level = "C"
        elif curve_span > threshold and level_span > threshold:
            level = "A"
        else:
            level = "B"
        return {
            "row": row, "curve_span": curve_span, "level_span": level_span,
            "monotone": mono, "metric_monotone": mmono, "grade": level,
            "knee_x": fnum(row.get("knee_x_final")),
            "reliable": str(row.get("knee_reliable")).strip() in ("1", "True", "true"),
            "n_windows": row.get("n_windows"),
        }

    def candidates(self, metric, tracker, curve_set="common_good", stratum="all"):
        """某指标 / 某臂通过全部守卫的候选（族 → 膝点）。"""
        out = {}
        for family in FAMILY_HEAVIEST:
            info = self.grade(tracker, family, metric, curve_set, stratum)
            if not info:
                continue
            if info["grade"] != "A" or not info["reliable"]:
                continue
            if not is_finite(info["knee_x"]):
                continue
            out[family] = info
        return out


# --------------------------------------------------------------------------
# 判定结果容器
# --------------------------------------------------------------------------
class Verdict:
    def __init__(self, ident, status, requirement, evidence, detail=None):
        self.ident = ident
        self.status = status
        self.requirement = requirement
        self.evidence = evidence
        self.detail = detail or []

    def line(self):
        return "%-8s %-8s %s —— %s" % (self.ident, self.status, self.requirement, self.evidence)


def section_text(markdown, marker):
    """截取报告里某个标题之后、下一个同级标题之前的正文。"""
    index = markdown.find(marker)
    if index < 0:
        return ""
    tail = markdown[index + len(marker):]
    stop = re.search(r"\n#{1,4} ", tail)
    return tail[: stop.start()] if stop else tail


def blockquote(markdown):
    """抽出正文里第一段连续的行引用（> 开头）。"""
    out = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith(">"):
            out.append(stripped.lstrip(">").strip())
        elif out:
            break
    return "\n".join(out)


# ==========================================================================
# A 系列：主判据（曲线是否真的塌 / 各指标是否拿到红线）
# ==========================================================================
def founded_metric(ctx, metric):
    """本轮该指标是否**真正成立**（＝C2 里「真正成立的红线」的唯一口径）。

    与 A3 同源：必须有有限线位 ＋ **决定曲线过全部守卫**（A 级 + knee_reliable=1）；
    `local_scr` 另外要求盲测区分间隔 ≥ 50 pp（A4）。
    ⇒ 「有线位」**不等于**「成立」——这是 C2 不能用「line_value 有限」当判据的原因。
    """
    row = ctx.line_index.get(metric)
    if not row:
        return False, "约束线表无该指标"
    line = fnum(row.get("line_value"))
    if not is_finite(line):
        return False, "本轮无有效红线（line_value = nan）"
    tracker = row.get("deciding_tracker")
    family = row.get("deciding_family")
    if not tracker or not family:
        return False, "无线位"
    info = ctx.grade(tracker, family, metric,
                      row.get("curve_set") or "common_good",
                      row.get("stratum") or "all")
    if not info:
        return False, "决定曲线在 knee_thresholds 里查不到"
    if info["grade"] != "A" or not info["reliable"]:
        return False, ("决定曲线未过守卫（%s 级 / knee_reliable=%d）"
                       % (info["grade"], int(info["reliable"])))
    if metric == "local_scr":
        bad = ctx.reg_index.get(("real_bad", metric))
        good = ctx.reg_index.get(("real_good", metric))
        # 🔴 缺回归行 ⇒ **无法**做盲测门禁 ⇒ 判**不成立**（不得因缺数据而放行）
        if not (bad and good):
            return False, "缺盲测回归行（real_bad / real_good）⇒ 无法核验盲测门禁"
        sep = (100.0 * fnum(bad.get("rate_below_line"))
               - 100.0 * fnum(good.get("rate_below_line")))
        if sep < A4_MIN_SEPARATION_PP:
            return False, "盲测区分间隔仅 %.2f pp（< 50 pp）" % sep
    return True, ("线位 %s（决定 %s/%s，过全部守卫）"
                  % (fmt(line, 10), family, tracker))


def a1_collapse_rate(ctx):
    """A1 的真值：**主臂**（xcorr / kcf_legacy）× 三族的最重档跟住率最低值。

    规格书 §三.3：CSRT 只是**同族对照臂**，不参与判据取值 ⇒ A1 只看两条主臂。
    """
    rates = ctx.level_rates()
    worst = None
    for family in sorted(rates):
        levels = sorted(rates[family])
        if not levels:
            continue
        heaviest = (max(levels) if FAMILY_HEAVIEST[family] == "max" else min(levels))
        bucket = rates[family][heaviest]
        for arm in MAIN_ARMS:
            pct = 100.0 * bucket["ok"][arm] / max(1, bucket["n"])
            if worst is None or pct < worst[0]:
                worst = (pct, family, arm)
    return worst


def check_a1(ctx):
    """A1：deep 轴下最重档跟住率是否显著低于 50%（曲线真的塌了）。"""
    rates = ctx.level_rates()
    detail = []
    collapsed = []
    for family in sorted(rates):
        levels = sorted(rates[family])
        heaviest = (max(levels) if FAMILY_HEAVIEST[family] == "max" else min(levels))
        bucket = rates[family][heaviest]
        row = []
        for arm in ALL_ARMS:
            pct = 100.0 * bucket["ok"][arm] / max(1, bucket["n"])
            row.append("%s %.1f%%" % (arm, pct))
            # 🔴 只认**主臂**：csrt 是同族对照臂，塌不塌不作为判据（规格书 §三.3）
            if arm in MAIN_ARMS and pct < A1_COLLAPSE_PCT:
                collapsed.append("%s/%s" % (family, arm))
        detail.append("%-9s 最重档 %-6s n=%-5d %s" %
                      (family, fmt(heaviest, 4), bucket["n"], "  ".join(row)))
    if collapsed:
        verdict = Verdict("A1", PASS, "deep 轴最重档跟住率显著低于 50%（曲线真塌）",
                          "以下组合跌破 50%%：%s" % ", ".join(collapsed), detail)
    else:
        worst = []
        for family in sorted(rates):
            levels = sorted(rates[family])
            heaviest = (max(levels) if FAMILY_HEAVIEST[family] == "max" else min(levels))
            bucket = rates[family][heaviest]
            for arm in MAIN_ARMS:
                worst.append((100.0 * bucket["ok"][arm] / max(1, bucket["n"]),
                              family, arm))
        worst.sort()
        top = "；".join("%s/%s %.1f%%" % (fam, arm, pct) for pct, fam, arm in worst[:3])
        verdict = Verdict("A1", FAIL, "deep 轴最重档跟住率显著低于 50%（曲线真塌）",
                          "无一组合跌破 50%%（主臂最低三档：%s）⇒ 曲线**未塌**，"
                          "膝点只是弯折点不是崩溃点；按规格书须保留"
                          "「本方法在净化底座上无法造出崩溃点」的措辞" % top, detail)
    return verdict


def check_a2(ctx):
    """A2：KCF 臂在 deep 轴下是否也过守卫（算法无关性能否成立）。"""
    detail = []
    # 规格书 A2 的字面口径：「KCF 的 blur/motion 曲线达到 A 级（分箱跨度 > 5 pp 且单调）」。
    # 「代表曲线」选哪个指标规格书没写 ⇒ **两个口径都判**，并说明结论是否与口径无关。
    carried = [m for m in CORE_METRICS
               if is_finite(fnum((ctx.line_index.get(m) or {}).get("line_value")))]
    scope = carried or list(CORE_METRICS)
    per_metric = {}
    for metric in scope:
        # 规格书 A2 的字面口径是「达到 A 级」，不含双算法门禁 ⇒ 用 grade 判
        a_families = [f for f in FAMILY_HEAVIEST
                      if (ctx.grade("kcf_legacy", f, metric) or {}).get("grade") == "A"]
        per_metric[metric] = a_families
        detail.append("KCF 臂 / %-24s 达到 A 级的族：%s"
                      % (metric, "、".join(a_families) or "**无**"))
    rep = {}
    for family in ("blur", "motion", "contrast"):
        info = ctx.grade("kcf_legacy", family, "local_scr")
        if not info:
            rep[family] = "缺失"
            continue
        rep[family] = info["grade"]
        detail.append("参考口径（代表曲线 local_scr，报告 §4.2 同口径）kcf_legacy/%-8s "
                      "分箱跨度 %5.2f pp  逐档跨度 %5.2f pp  单调 %s ⇒ %s 级"
                      % (family, 100 * info["curve_span"], 100 * info["level_span"],
                         "是" if info["monotone"] else "否", info["grade"]))

    need = ("blur", "motion")
    miss = []
    got = []
    for metric, families in per_metric.items():
        for family in need:
            tag = "%s/%s" % (metric, family)
            if family in families:
                got.append(tag)
            else:
                miss.append(tag)
    rep_miss = [f for f in need if rep.get(f) != "A"]
    detail.append("结论与口径无关性：按各指标自己的曲线 ⇒ 过 %s，未过 %s；"
                  "按 local_scr 代表曲线 ⇒ 未过 %s"
                  % ("、".join(got) or "无", "、".join(miss) or "无",
                     "、".join(rep_miss) or "无"))
    if not miss and not rep_miss:
        return Verdict("A2", PASS, "KCF 臂 blur/motion 曲线均达 A 级",
                       "两种口径下 blur/motion 均达 A 级", detail)
    if got:
        return Verdict("A2", PARTIAL,
                       "KCF 臂 blur/motion 曲线均达 A 级",
                       "仅部分族过守卫（过：%s；未过：%s）⇒ **算法无关性本轮不成立**，"
                       "对外表述必须继续带「单臂反解」限定"
                       % ("、".join(got), "、".join(miss) or "、".join(rep_miss)), detail)
    return Verdict("A2", FAIL, "KCF 臂 blur/motion 曲线均达 A 级",
                   "KCF 两族均未达 A 级 ⇒ 算法无关性不成立", detail)


def check_a3(ctx):
    """A3：target_lap_var 是否拿到有效红线。"""
    row = ctx.line_index.get("target_lap_var")
    if not row:
        return Verdict("A3", FAIL, "target_lap_var 拿到有效红线", "约束线表无该指标行", [])
    line = fnum(row.get("line_value"))
    tracker = row.get("deciding_tracker")
    family = row.get("deciding_family")
    if not is_finite(line):
        return Verdict("A3", FAIL, "target_lap_var 拿到有效红线",
                       "line_value = nan（本轮无有效红线）", [])
    curve_set = row.get("curve_set") or "common_good"
    stratum = row.get("stratum") or "all"
    info = ctx.grade(tracker, family, "target_lap_var", curve_set, stratum)
    detail = ["线位 %s，决定族/臂 %s/%s，有效候选 %s 个，跨臂偏差 %s%%"
              % (fmt(line, 10), family, tracker, row.get("n_candidates"),
                 fmt(fnum(row.get("cross_arm_dev_pct")), 4))]
    if info:
        detail.append("决定曲线：分箱跨度 %.2f pp（门槛 5）  逐档跨度 %.2f pp（门槛 5）  "
                      "单调 %s  knee_reliable=%s"
                      % (100 * info["curve_span"], 100 * info["level_span"],
                         "是" if info["monotone"] else "否", int(info["reliable"])))
    ok_span = bool(info) and info["curve_span"] > FROZEN_CONSTANTS["CURVE_MIN_SPAN"] \
        and info["level_span"] > FROZEN_CONSTANTS["CURVE_MIN_SPAN"]
    ok_mono = bool(info) and info["monotone"]
    ok_knee = bool(info) and info["reliable"]
    if ok_span and ok_mono and ok_knee:
        return Verdict("A3", PASS, "target_lap_var 拿到有效红线",
                       "线位 %s：分箱跨度、逐档跨度均 > 5pp，指标单调，"
                       "双算法门禁 knee_reliable=1" % fmt(line, 10), detail)
    reasons = []
    if not ok_span:
        reasons.append("跨度不足")
    if not ok_mono:
        reasons.append("跟住率非单调")
    if not ok_knee:
        reasons.append("双算法门禁未过")
    return Verdict("A3", FAIL, "target_lap_var 拿到有效红线",
                   "线位 %s 但 %s ⇒ 锐度维度继续不给数值（不得放宽门槛）"
                   % (fmt(line, 10), "、".join(reasons)), detail)


def check_a4(ctx):
    """A4：local_scr 盲测区分间隔 ≥ 50pp。"""
    detail = []
    separations = {}
    for metric in CORE_METRICS:
        bad = ctx.reg_index.get(("real_bad", metric))
        good = ctx.reg_index.get(("real_good", metric))
        if not bad or not good:
            continue
        bad_pct = 100.0 * fnum(bad.get("rate_below_line"))
        good_pct = 100.0 * fnum(good.get("rate_below_line"))
        separations[metric] = bad_pct - good_pct
        detail.append("%-24s 阈值 %-12s 坏片拦截 %6.2f%%  真实好片错误豁免 %6.2f%%  ⇒ 间隔 %6.2f pp"
                      % (metric, fmt(fnum(bad.get("threshold")), 8), bad_pct, good_pct,
                         bad_pct - good_pct))
    row = ctx.line_index.get("local_scr")
    line = fnum(row.get("line_value")) if row else float("nan")
    separation = separations.get("local_scr", float("nan"))
    detail.append("local_scr 有效红线：%s" % fmt(line, 10))
    if not is_finite(line):
        return Verdict("A4", FAIL, "local_scr 盲测区分间隔 ≥ 50pp",
                       "本轮 local_scr **无有效红线**（line_value=nan）⇒ 无法进入对外"
                       "免责声明；参照阈值下区分间隔仅 %.2f pp（< 50pp）"
                       % separation, detail)
    if separation >= A4_MIN_SEPARATION_PP:
        return Verdict("A4", PASS, "local_scr 盲测区分间隔 ≥ 50pp",
                       "区分间隔 %.2f pp ≥ 50pp" % separation, detail)
    return Verdict("A4", FAIL, "local_scr 盲测区分间隔 ≥ 50pp",
                   "区分间隔仅 %.2f pp < 50pp ⇒ 继续不进对外免责声明" % separation, detail)


# ==========================================================================
# B 系列：反向判据（防止「越做越松」）
# ==========================================================================
def check_b1(ctx):
    """B1：守卫门槛 / 三分法 / 切分口径逐位未动。"""
    detail = []
    problems = []
    sources = {
        "CURVE_MIN_SPAN": os.path.join(ctx.code_dir, "run_eval_v2.py"),
        "KNEE_AGREE_PCT": os.path.join(ctx.code_dir, "run_eval_v2.py"),
        "N_BINS": os.path.join(ctx.code_dir, "run_eval_v2.py"),
        "GOOD_CLE_PX": os.path.join(ctx.code_dir, "regression_v2.py"),
        "CALIB_RATIO": os.path.join(ctx.code_dir, "regression_v2.py"),
        "SPLIT_SEED": os.path.join(ctx.code_dir, "regression_v2.py"),
    }
    for name, expected in FROZEN_CONSTANTS.items():
        actual = read_source_constant(sources[name], name)
        ok = actual is not None and abs(actual - expected) < 1e-12
        detail.append("%-16s 期望 %-12s 实测 %-12s %s"
                      % (name, expected, actual, "✅" if ok else "❌"))
        if not ok:
            problems.append("%s 变了（%s → %s）" % (name, expected, actual))

    # 容差口径 max(5, 0.5·max(w,h))
    tol_re = re.compile(r"max\(\s*5(?:\.0)?\s*,\s*0\.5\s*\*\s*max\(")
    tol_ok = False
    for path in (os.path.join(ctx.code_dir, "run_eval_v2.py"),
                 os.path.join(ctx.code_dir, "regression_v2.py")):
        if os.path.exists(path) and tol_re.search(open(path, encoding="utf-8").read()):
            tol_ok = True
            break
    detail.append("%-16s 期望 max(5, 0.5·max(w,h))  实测 %s"
                  % ("跟住容差", "一致" if tol_ok else "未找到"))
    if not tol_ok:
        problems.append("跟住容差口径未找到")

    # 基线档位表必须逐位不变（deep 只能**加**档位）
    deg = os.path.join(ctx.code_dir, "degradation_4d.py")
    frozen_deg = os.path.join(ctx.frozen_code_dir, "degradation_4d.py")
    for name, expected in (("BLUR_SIGMAS", FROZEN_BLUR),
                           ("MOTION_LENGTHS", FROZEN_MOTION),
                           ("CONTRAST_ALPHAS", FROZEN_CONTRAST)):
        actual = read_source_tuple(deg, name)
        frozen = read_source_tuple(frozen_deg, name)
        ok = actual == expected and (frozen is None or frozen == expected)
        detail.append("%-16s 期望 %-38s 实测 %-38s %s"
                      % (name, expected, actual, "✅" if ok else "❌"))
        if not ok:
            problems.append("%s 与冻结基线不一致" % name)

    if problems:
        return Verdict("B1", FAIL, "守卫门槛 / 三分法 / 切分口径不许动",
                       "；".join(problems), detail)
    return Verdict("B1", PASS, "守卫门槛 / 三分法 / 切分口径不许动",
                   "全部冻结常量逐位一致（深轮只加了档位，没动任何门槛）", detail)


def check_b2(ctx):
    """B2：盲测集只做一次性验证（真实素材数字不参与阈值搜索）。"""
    detail = []
    problems = []
    deep_blind = os.path.join(ctx.deep_dir, "blind_real_v2.csv")
    base_blind = os.path.join(ctx.baseline_dir, "blind_real_v2.csv")
    if os.path.exists(deep_blind) and os.path.exists(base_blind):
        d_md5, b_md5 = md5_of(deep_blind), md5_of(base_blind)
        same = d_md5 == b_md5
        detail.append("blind_real_v2.csv 两轮 md5：deep=%s baseline=%s ⇒ %s"
                      % (d_md5[:12], b_md5[:12], "逐字节一致" if same else "不一致"))
        if not same:
            problems.append("盲测产物两轮不一致（素材或口径被改）")
    else:
        problems.append("找不到 blind_real_v2.csv")

    # 约束线必须**只**由合成侧（knee_thresholds）可复算 ⇒ 真实素材没进搜索
    for metric in CORE_METRICS:
        row = ctx.line_index.get(metric)
        if not row:
            continue
        line = fnum(row.get("line_value"))
        # 🔴 复算必须用**约束线行自己声明**的曲线池（与 A3 口径一致），
        #    否则会在错的池上复核出「可复算」的假象
        recomputed, arms = recompute_line(ctx, metric,
                                          row.get("curve_set") or "common_good",
                                          row.get("stratum") or "all")
        if not is_finite(line):
            detail.append("%-24s 本轮无线位（n_candidates=%s）" % (metric, row.get("n_candidates")))
            continue
        ok = is_finite(recomputed) and abs(recomputed - line) <= 1e-9 * max(1.0, abs(line))
        detail.append("%-24s 约束线 %-22s 仅用 knee_thresholds（合成侧）复算 = %-22s %s"
                      % (metric, fmt(line, 10), fmt(recomputed, 10), "✅" if ok else "❌"))
        detail.append("    主臂线位：%s" % "  ".join("%s=%s" % (a, fmt(v, 10)) for a, v in arms.items()))
        if not ok:
            problems.append("%s 线位无法由合成侧复算（疑有盲测参与）" % metric)

    if problems:
        return Verdict("B2", FAIL, "盲测集只做一次性验证",
                       "；".join(problems), detail)
    return Verdict("B2", PASS, "盲测集只做一次性验证",
                   "盲测产物两轮逐字节一致，且约束线**只**由合成侧 knee_thresholds 可复算"
                   "⇒ 真实素材数字未参与阈值搜索", detail)


def recompute_line(ctx, metric, curve_set="common_good", stratum="all"):
    """只用 knee_thresholds 复算某指标的主臂线位（每族取最小，再跨臂取最小）。"""
    arm_lines = {}
    for arm in MAIN_ARMS:
        cands = ctx.candidates(metric, arm, curve_set, stratum)
        if not cands:
            continue
        arm_lines[arm] = min(info["knee_x"] for info in cands.values())
    if not arm_lines:
        return float("nan"), arm_lines
    return min(arm_lines.values()), arm_lines


def check_b3(ctx):
    """B3：仿真池与基线轮同池同窗口。"""
    detail = []
    problems = []
    d_split = os.path.join(ctx.deep_dir, "split_v2.csv")
    b_split = os.path.join(ctx.baseline_dir, "split_v2.csv")
    if os.path.exists(d_split) and os.path.exists(b_split):
        same = md5_of(d_split) == md5_of(b_split)
        detail.append("split_v2.csv 两轮 md5 %s ⇒ 序列级 7:3 切分 %s"
                      % ("一致" if same else "不一致", "逐字节相同" if same else "**已变**"))
        if not same:
            problems.append("序列级切分与基线不一致")

    # 三分法分类：行序无关的全集比对
    def seg_set(rows):
        return {(r["seq_id"], int(r["frame_id"]), r.get("class_xcorr"),
                 r.get("class_kcf_legacy"), r.get("class_csrt"), r.get("class_ds"))
                for r in rows}
    d_seg, b_seg = seg_set(ctx.deep_seg), seg_set(ctx.base_seg)
    same_seg = d_seg == b_seg
    detail.append("segment_class_v2.csv：deep %d 行 / baseline %d 行 ⇒ 全集 %s"
                  % (len(ctx.deep_seg), len(ctx.base_seg),
                     "逐行一致（仅行序不同）" if same_seg else "**不一致**"))
    if not same_seg:
        problems.append("三分法分类与基线不一致（%d vs %d 行）" % (len(d_seg), len(b_seg)))

    # 仿真池（被实际仿真的窗口集合）逐个比对
    d_pool, b_pool = ctx.pool_of(ctx.deep_detail), ctx.pool_of(ctx.base_detail)
    same_pool = d_pool == b_pool
    detail.append("仿真池窗口：deep %d / baseline %d ⇒ %s"
                  % (len(d_pool), len(b_pool),
                     "同一批窗口" if same_pool else "**不同**"))
    if not same_pool:
        problems.append("仿真池与基线不同（deep %d vs baseline %d）" % (len(d_pool), len(b_pool)))

    # 留出集窗口不得出现在退化明细里
    calib = {r["seq_id"] for r in ctx.deep_split if r.get("split") == "calib"}
    holdout_windows = {(r["seq_id"], int(r["frame_id"])) for r in ctx.deep_seg
                       if r["seq_id"] in {s["seq_id"] for s in ctx.deep_split
                                          if s.get("split") == "holdout"}}
    leaked = holdout_windows & d_pool
    detail.append("留出集窗口出现在退化明细中的条数：%d" % len(leaked))
    if leaked:
        problems.append("留出集泄漏进仿真池 %d 条" % len(leaked))

    # 档位阶梯：deep 必须是 baseline 的严格超集
    base_levels = set()
    deep_levels = set()
    for row in ctx.base_detail:
        if row.get("condition_family") != "clean":
            base_levels.add((row["condition_family"], fnum(row["condition_level"])))
    for row in ctx.deep_detail:
        if row.get("condition_family") != "clean":
            deep_levels.add((row["condition_family"], fnum(row["condition_level"])))
    missing = base_levels - deep_levels
    detail.append("档位阶梯：baseline %d 档 / deep %d 档 ⇒ baseline ⊆ deep：%s"
                  % (len(base_levels), len(deep_levels), "是" if not missing else "**否**"))
    if missing:
        problems.append("deep 少了 baseline 的档位：%s" % sorted(missing)[:5])

    # 明细行数
    clean_rows = sum(1 for r in ctx.deep_detail if r.get("condition_family") == "clean")
    deg_rows = len(ctx.deep_detail) - clean_rows
    expected_deg = len(d_pool) * len(deep_levels)
    detail.append("明细行数：clean %d + 退化 %d = %d（池 %d × 档位 %d = %d）"
                  % (clean_rows, deg_rows, len(ctx.deep_detail), len(d_pool),
                     len(deep_levels), expected_deg))
    if deg_rows != expected_deg:
        problems.append("退化明细行数 %d ≠ 池×档位 %d" % (deg_rows, expected_deg))
    if len(ctx.deep_detail) != clean_rows + expected_deg:
        problems.append("明细总行数 %d ≠ 规格书 §四 的 24817"
                        % len(ctx.deep_detail))

    # 两轮**共有档位**上的逐行数值必须逐位相同（同一底座、同一确定性退化 ⇒ 可逐点比对）
    compare_cols = ["local_scr", "target_local_contrast", "target_lap_var",
                    "target_grad_mean", "surround_std", "surround_edge_density",
                    "surround_blockiness", "global_contrast", "global_lap_var",
                    "xcorr_success", "kcf_legacy_success", "csrt_success",
                    "xcorr_cle", "kcf_legacy_cle", "csrt_cle"]

    def _index(rows, mode):
        """mode: 'clean' 只取 clean 行 / 'degraded' 只取退化行 / 'all' 全取。"""
        out = {}
        for row in rows:
            is_clean = row.get("condition_family") == "clean"
            if mode == "clean" and not is_clean:
                continue
            if mode == "degraded" and is_clean:
                continue
            key = (row["seq_id"], int(row["frame_id"]),
                   row.get("condition_family"), row.get("condition_level"))
            out[key] = row
        return out

    for mode, label in (("degraded", "共有退化档位"),
                        ("clean", "clean 档（Pass A 三分法）")):
        d_idx = _index(ctx.deep_detail, mode)
        b_idx = _index(ctx.base_detail, mode)
        shared = set(d_idx) & set(b_idx)
        mismatched = 0
        first_bad = ""
        for key in shared:
            d_row, b_row = d_idx[key], b_idx[key]
            for col in compare_cols:
                if col not in d_row or col not in b_row:
                    continue
                d_val, b_val = fnum(d_row.get(col)), fnum(b_row.get(col))
                if is_finite(d_val) and is_finite(b_val):
                    if abs(d_val - b_val) > 1e-9 * max(1.0, abs(d_val)):
                        mismatched += 1
                        if not first_bad:
                            first_bad = "%s/%s %s: deep %s vs base %s" % (
                                key[0], key[1], col, fmt(d_val, 10), fmt(b_val, 10))
                elif d_val != b_val:
                    mismatched += 1
                    if not first_bad:
                        first_bad = "%s/%s %s: deep %r vs base %r" % (
                            key[0], key[1], col, d_val, b_val)
        detail.append("%s: 两轮共有 %d 行 x %d 个读数 => 不一致 %d 处%s"
                      % (label, len(shared), len(compare_cols), mismatched,
                         ("; 首个: " + first_bad) if first_bad else ""))
        if mismatched:
            problems.append("%s 有 %d 处数值不一致" % (label, mismatched))

    if problems:
        return Verdict("B3", FAIL, "仿真池与基线轮同池同窗口",
                       "；".join(problems), detail)
    return Verdict("B3", PASS, "仿真池与基线轮同池同窗口",
                   "切分逐字节一致、三分法一致、仿真池同批 %d 窗口、留出集 0 泄漏、"
                   "档位为基线严格超集" % len(d_pool), detail)


def check_b4(ctx):
    """B4：线位每族取最小值，绝不跨臂求平均。"""
    detail = []
    problems = []
    for metric in CORE_METRICS:
        row = ctx.line_index.get(metric)
        if not row:
            continue
        line = fnum(row.get("line_value"))
        arm_cols = {arm: fnum(row.get("%s_line" % arm)) for arm in ALL_ARMS}
        finite_arms = {arm: v for arm, v in arm_cols.items() if is_finite(v)}
        if not is_finite(line):
            detail.append("%-24s 本轮无线位" % metric)
            continue
        main = [arm_cols[arm] for arm in MAIN_ARMS if is_finite(arm_cols[arm])]
        expect_min = min(main) if main else float("nan")
        expect_avg = sum(main) / len(main) if main else float("nan")
        is_min = abs(expect_min - line) <= 1e-9 * max(1.0, abs(line))
        is_avg = (len(main) >= 2 and is_finite(expect_avg)
                  and abs(expect_avg - line) <= 1e-9 * max(1.0, abs(line)))
        # 每族取最小：复核主臂线位确实是**该臂声明曲线池上**各族候选的最小
        curve_set = row.get("curve_set") or "common_good"
        stratum = row.get("stratum") or "all"
        per_arm_ok = True
        for arm in MAIN_ARMS:
            cands = ctx.candidates(metric, arm, curve_set, stratum)
            if cands:
                fam_min = min(i["knee_x"] for i in cands.values())
                if abs(fam_min - arm_cols[arm]) > 1e-9 * max(1.0, abs(arm_cols[arm])):
                    per_arm_ok = False
        detail.append("%-24s 线位 %-22s 主臂 %s ⇒ 跨臂最小 %-22s 平均 %-22s %s"
                      % (metric, fmt(line, 10),
                         " ".join("%s=%s" % (a, fmt(arm_cols[a], 8)) for a in MAIN_ARMS),
                         fmt(expect_min, 10), fmt(expect_avg, 10),
                         "=最小 ✅" if is_min else ("=平均 ❌" if is_avg else "**既非最小也非平均 ❌**")))
        detail.append("    对照臂 csrt = %s（只作参照，不参与取值）" % fmt(arm_cols.get("csrt"), 8))
        detail.append("    每族取最小复核：%s" % ("通过" if per_arm_ok else "**不通过**"))
        if not is_min:
            problems.append("%s 线位不是跨臂最小值" % metric)
        if is_avg:
            problems.append("%s 线位疑似跨臂求平均" % metric)
        if not per_arm_ok:
            problems.append("%s 某臂线位不是该臂各族候选的最小值" % metric)
    if problems:
        return Verdict("B4", FAIL, "线位每族取最小、绝不跨臂求平均",
                       "；".join(problems), detail)
    return Verdict("B4", PASS, "线位每族取最小、绝不跨臂求平均",
                   "各指标线位均等于主臂候选的**跨臂最小值**（非平均）；"
                   "对照臂 csrt 未参与取值", detail)


def stricter_line(ctx, metric):
    """两轮线位里的**更严者**（数值更小者）。

    Returns:
        tuple: (更严者, deep 线位, 基线线位)；两轮都无值时更严者为 nan。
    """
    d_line = fnum((ctx.line_index.get(metric) or {}).get("line_value"))
    b_line = fnum((ctx.base_line_index.get(metric) or {}).get("line_value"))
    if is_finite(d_line) and is_finite(b_line):
        return min(d_line, b_line), d_line, b_line
    if is_finite(d_line):
        return d_line, d_line, b_line
    if is_finite(b_line):
        return b_line, d_line, b_line
    return float("nan"), d_line, b_line


def outward_clauses(report_text):
    """解析对外可发布版（§10.3 首段引用）里的阈值子句。

    Returns:
        tuple[str, dict, list]: (正文, {指标: 数值}, 被**提到**的指标)
    """
    quote = blockquote(section_text(report_text, "### 10.3"))
    parsed = {}
    mentioned = []
    if quote:
        for metric in CORE_METRICS:
            if metric not in quote:
                continue
            mentioned.append(metric)
            nums = re.findall(r"`%s\s*<\s*([0-9.]+)`" % metric, quote)
            if nums:
                parsed[metric] = float(nums[0])
    return quote, parsed, mentioned


#: 规格书 B5 的补救措辞（deep 线位比基线更松时**必须**写明）
B5_MOVE_WORDING = "口径变化导致线位移动"

#: 判据**定义表**的行（`| **A1** | … |`）——它只是引用规格措辞，不是本轮事实
_CRITERION_ROW_RE = re.compile(r"^\|\s*\*\*[A-D]\d\*\*\s*\|")

#: 结论词（中文「通过 / 不通过 / 部分通过」与英文 PASS / FAIL）
CONCLUSION_WORDS = ("通过", "PASS", "FAIL")


def report_body(report_text):
    """报告正文（剔除判据**定义表**的行）——用于核对「本轮事实性」的措辞。"""
    out = []
    for line in report_text.splitlines():
        stripped = line.strip()
        if (_CRITERION_ROW_RE.match(stripped)
                and not any(word in stripped for word in CONCLUSION_WORDS)):
            continue
        out.append(line)
    return "\n".join(out)


def check_b5(ctx, report_text=""):
    """B5：新红线不得比基线更松；取两者更严者。"""
    detail = []
    problems = []
    _quote, published, _mentioned = outward_clauses(report_text)
    body = report_body(report_text)
    for metric in CORE_METRICS:
        stricter, d_line, b_line = stricter_line(ctx, metric)
        if is_finite(d_line) and is_finite(b_line):
            which = "deep" if d_line <= b_line else "baseline"
            # 🔴 对外版公布的线位必须**等于两轮更严者**（规格书 B5 的补救口径）
            pub = published.get(metric, d_line)
            ok = abs(pub - stricter) <= 1e-6 * max(1.0, abs(stricter))
            detail.append("%-24s deep %-22s baseline %-22s ⇒ 取更严者 = %-22s（%s）"
                          " 对外版 %-22s %s"
                          % (metric, fmt(d_line, 10), fmt(b_line, 10), fmt(stricter, 10),
                             which, fmt(pub, 10), "✅" if ok else "❌ 未取更严者"))
            if not ok:
                problems.append("%s 未取更严者（对外版 %s，应为 %s）"
                                % (metric, fmt(pub, 10), fmt(stricter, 10)))
            if d_line > b_line and B5_MOVE_WORDING not in body:
                problems.append("%s deep 线位比基线更松（%s > %s），报告未写明「%s」"
                                % (metric, fmt(d_line, 10), fmt(b_line, 10),
                                   B5_MOVE_WORDING))
        elif is_finite(d_line):
            detail.append("%-24s deep %-22s baseline %-22s ⇒ deep 为新线（基线无数值），"
                          "不构成放宽" % (metric, fmt(d_line, 10), fmt(b_line, 10)))
        elif is_finite(b_line):
            detail.append("%-24s deep %-22s baseline %-22s ⇒ deep 本轮无有效线位，"
                          "**不放宽**：基线线位仍有效" % (metric, fmt(d_line, 10), fmt(b_line, 10)))
        else:
            detail.append("%-24s deep %-22s baseline %-22s ⇒ 两轮均无有效线位"
                          % (metric, fmt(d_line, 10), fmt(b_line, 10)))
    if problems:
        return Verdict("B5", FAIL, "取两轮更严者，线位不得被放宽",
                       "；".join(problems), detail)
    stricter = []
    for metric in CORE_METRICS:
        _strict, d_line, b_line = stricter_line(ctx, metric)
        if is_finite(d_line) and is_finite(b_line):
            stricter.append("%s 取 %s %s（两轮更严者）"
                            % (metric, "deep" if d_line <= b_line else "基线",
                               fmt(min(d_line, b_line), 8)))
        elif is_finite(d_line):
            stricter.append("%s 取 deep %s（基线无有效红线，新线）" % (metric, fmt(d_line, 8)))
        elif is_finite(b_line):
            stricter.append("%s deep 无线位 ⇒ 不放宽（基线线位 %s 仅在形式上仍在，"
                            "其对外可用性另受 A4 约束）" % (metric, fmt(b_line, 8)))
    return Verdict("B5", PASS, "取两轮更严者，线位不得被放宽",
                   "；".join(stricter), detail)



def check_b6(ctx):
    """B6：deep 档位表必须**恰好**等于规格书 §1.1 写死的档位（不多不少）。"""
    detail = []
    problems = []
    meta = ctx.meta or {}
    # run_meta.json 有两种形态：早期扁平（blur_sigmas/…）与现在的
    # conditions_profile_declared{blur,motion,contrast} + detail_levels_detected{…}
    declared_map = meta.get("conditions_profile_declared") or {}
    detected_map = meta.get("detail_levels_detected") or {}
    declared = {}
    for family in ("blur", "motion", "contrast"):
        values = (declared_map.get(family)
                  or meta.get("%s_sigmas" % family)
                  or meta.get("%s_lengths" % family)
                  or meta.get("%s_alphas" % family)
                  or detected_map.get(family) or ())
        declared[family] = tuple(values)
    frozen = {
        "blur": FROZEN_DEEP_BLUR,
        "motion": FROZEN_DEEP_MOTION,
        "contrast": FROZEN_DEEP_CONTRAST,
    }
    # 实际跑出来的档位（从明细表反推）
    actual = {"blur": set(), "motion": set(), "contrast": set()}
    for row in ctx.deep_detail:
        family = row.get("condition_family")
        if family in actual:
            actual[family].add(round(fnum(row.get("condition_level")), 6))
    for family in ("blur", "motion", "contrast"):
        want = tuple(round(float(x), 6) for x in frozen[family])
        got_declared = tuple(round(float(x), 6) for x in declared[family])
        got_actual = tuple(sorted(actual[family]))
        ok_decl = got_declared == want
        ok_actual = got_actual == tuple(sorted(want))
        detail.append("%-9s 规格 %-42s run_meta %-42s 实跑 %s"
                      % (family, str(want), str(got_declared),
                         "一致 ✅" if ok_actual else str(got_actual)))
        if not ok_decl:
            problems.append("%s 的 run_meta 声明档位与规格书不一致" % family)
        if not ok_actual:
            problems.append("%s 的实际档位与规格书不一致" % family)
    if problems:
        return Verdict("B6", FAIL, "deep 档位表恰好等于规格书 §1.1",
                       "；".join(problems), detail)
    return Verdict("B6", PASS, "deep 档位表恰好等于规格书 §1.1",
                   "三族档位与规格书逐位一致（σ≤8 / L≤15 / α≤0.05，共 25 档）", detail)


def _extract_numbers(text):
    """从一段文字里抽出所有数字（含小数、百分数）。"""
    out = []
    for token in re.findall(r"-?\d+(?:\.\d+)?", text or ""):
        try:
            out.append(float(token))
        except ValueError:
            continue
    return out


#: A2 单元格里的「**无**族达 A 级」表述（窄口径：有族达 A 时不能这样写）
D7_NO_A_CLAIM = re.compile(r"族\s*[:：]?\s*\**无\**|无一(?:个)?族|(?:均|全部|都|全都)未过")
#: A2 单元格里的「某族达 A 级」**肯定**表述（无族达 A 时不能这样写）
D7_A_CLAIM = re.compile(r"`?(?:blur|motion|contrast)`?[^。\n]{0,6}达\s*A")
#: A2 单元格里的「没有族达 A 级」表述（宽口径）
D7_NO_A_ANY = re.compile(r"族\s*[:：]?\s*\**无\**|无一(?:个)?族|未过")


def check_d7(ctx, report_text):
    """D7：报告「判据总表」的 A1~A4 实测单元格 vs 现场重算。"""
    detail = []
    problems = []
    checked = 0
    tables = _markdown_tables(report_text)
    # A1 的真值：主臂（xcorr / kcf_legacy）三族最重档取最低（与 check_a1 同口径）
    worst = a1_collapse_rate(ctx)
    a1_true = worst[0] if worst else None
    a3_true = fnum((ctx.line_index.get("target_lap_var") or {}).get("line_value"))
    bad = ctx.reg_index.get(("real_bad", "local_scr"))
    good = ctx.reg_index.get(("real_good", "local_scr"))
    a4_true = (100.0 * fnum(bad.get("rate_below_line"))
               - 100.0 * fnum(good.get("rate_below_line"))) if (bad and good) else float("nan")

    for table in tables:
        header = table[0]
        col_meas = _col(header, "实测")
        if col_meas < 0:
            continue
        col_id = 0
        for cells in table[1:]:
            ident = cells[col_id].strip("*` ").upper()
            if ident not in ("A1", "A2", "A3", "A4"):
                continue
            cell = cells[col_meas]
            nums = _extract_numbers(cell)
            if ident == "A1":
                if a1_true is None:
                    problems.append("A1 真值算不出来（现场重算无数据）⇒ 无法对账")
                    continue
                if not any(abs(n - a1_true) <= 0.06 for n in nums):
                    problems.append("A1 实测单元格「%s」里没有 %.1f%%（真值）"
                                    % (cell, a1_true))
                else:
                    checked += 1
                    detail.append("A1 实测 %s ✅（真值 %.1f%%）" % (cell, a1_true))
            elif ident == "A3":
                # 🔴 现场真值缺失 ⇒ 不能当「核过了」（否则「全部一致」是假的）
                if not is_finite(a3_true):
                    problems.append("A3 真值算不出来（本轮 target_lap_var 无有效红线）"
                                    "⇒ 无法对账")
                    continue
                if not any(abs(n - a3_true) <= 1e-3 * max(1.0, abs(a3_true)) for n in nums):
                    problems.append("A3 实测单元格「%s」里没有线位 %s"
                                    % (cell, fmt(a3_true, 10)))
                else:
                    checked += 1
                    detail.append("A3 实测 %s ✅（真值 %s）" % (cell, fmt(a3_true, 10)))
            elif ident == "A4":
                if not is_finite(a4_true):
                    problems.append("A4 真值算不出来（缺盲测回归行）⇒ 无法对账")
                    continue
                if not any(abs(n - a4_true) <= 0.02 for n in nums):
                    problems.append("A4 实测单元格「%s」里没有区分间隔 %.2f pp"
                                    % (cell, a4_true))
                else:
                    checked += 1
                    detail.append("A4 实测 %s ✅（真值 %.2f pp）" % (cell, a4_true))
            elif ident == "A2":
                # KCF 是否有族达 A 级：**双向**核对（有的不能写「无」，没有的不能写「有」）
                a_families = set()
                for metric in CORE_METRICS:
                    for family in FAMILY_HEAVIEST:
                        info = ctx.grade("kcf_legacy", family, metric)
                        if info and info["grade"] == "A":
                            a_families.add(family)
                if a_families:
                    if D7_NO_A_CLAIM.search(cell):
                        problems.append("A2 实测单元格「%s」写「无族达 A 级」，"
                                        "但 KCF 臂有族达 A 级（%s）"
                                        % (cell, "、".join(sorted(a_families))))
                    else:
                        checked += 1
                        detail.append("A2 实测 %s ✅（KCF 达 A 级的族：%s）"
                                      % (cell, "、".join(sorted(a_families))))
                else:
                    if D7_A_CLAIM.search(cell):
                        problems.append("A2 实测单元格「%s」声称有族达 A 级，"
                                        "但 KCF 臂**无**族达 A 级" % cell)
                    elif not D7_NO_A_ANY.search(cell):
                        problems.append("A2 实测单元格「%s」未写明「无族达 A 级」" % cell)
                    else:
                        checked += 1
                        detail.append("A2 实测 %s ✅（KCF 无族达 A 级）" % cell)
    if checked == 0:
        return Verdict("D7", FAIL, "报告判据总表的实测单元格与重算一致",
                       "**未能定位到判据总表** —— 对账无效，不能判 PASS", detail)
    if problems:
        return Verdict("D7", FAIL, "报告判据总表的实测单元格与重算一致",
                       "；".join(problems[:6]), detail)
    return Verdict("D7", PASS, "报告判据总表的实测单元格与重算一致",
                   "共核对 %d 个判据单元格（A1/A2/A3/A4），全部一致" % checked, detail)


def check_d8(ctx, report_text):
    """D8：报告内部交叉引用可达 ＋ 对外版阈值精度（有效位数）。"""
    detail = []
    problems = []
    # ---- 8.1 悬空的「§X.Y」引用（任务书 / 规格书的引用不在此列） --------------
    present = set()
    for line in report_text.splitlines():
        head = re.match(r"^#{2,4}\s+([^\s]+)", line.strip())
        if head:
            present.add(head.group(1).strip("〇一二三四五六七八九十."))
    for match in re.finditer(r"§(\d+\.\d+)", report_text):
        token = match.group(1)
        # 豁免窗口取**整行**：引用外部规格书（如 `DEEP_AXIS_SPEC_20261127.md §1.1`）
        # 写在同一行里，不该被当成报告内部的悬空引用
        line_start = report_text.rfind("\n", 0, match.start()) + 1
        line_end = report_text.find("\n", match.start())
        window = report_text[line_start:line_end if line_end >= 0 else len(report_text)]
        if ("任务书" in window or "规格书" in window or "GOAL" in window
                or "DEEP_AXIS" in window):
            continue
        if token in present:
            continue
        line_no = report_text[: match.start()].count("\n") + 1
        problems.append("L%-5d 悬空交叉引用「§%s」（报告里没有这一节）" % (line_no, token))
    detail.append("报告里存在的章节号：%s" % "、".join(sorted(present)))
    # ---- 8.2 对外版阈值的有效位数 -------------------------------------------
    section = section_text(report_text, "### 10.3")
    quote = blockquote(section)
    if quote:
        for match in re.finditer(r"`[a-z_]+\s*<\s*([0-9.]+)`", quote):
            literal = match.group(1)
            digits = literal.replace(".", "").lstrip("0")
            if len(digits) > 7:
                problems.append("对外版阈值「%s」写了 %d 位有效数字（应对外取 7 位）"
                                % (literal, len(digits)))
    else:
        problems.append("未能从 §10.3 抽出对外版正文")
    if problems:
        return Verdict("D8", FAIL, "报告交叉引用可达且对外阈值精度合规",
                       "；".join(problems[:6]), detail)
    return Verdict("D8", PASS, "报告交叉引用可达且对外阈值精度合规",
                   "无悬空 §X.Y 引用；对外版阈值有效位数 ≤ 7", detail)


# ==========================================================================
# C 系列：交付判据
# ==========================================================================
def criteria_with_conclusion(text, marks):
    """在**表格行**里同时出现「判据编号」与「结论词」的判据编号集合。

    只在正文里提到编号（如「这正是 A1 要用实测回答的问题」）**不算**给出了结论；
    只有「编号 + 结论」同处一行才算逐条给出了实测结论。
    """
    found = set()
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        if not any(word in stripped for word in CONCLUSION_WORDS):
            continue
        for mark in marks:
            if re.search(r"(?<![A-Za-z0-9])%s(?![0-9])" % mark, stripped):
                found.add(mark)
    return found


def check_c1(ctx, report_text, acceptance_text=""):
    """C1：报告必须逐条给出 A1~A4 与 B1~B5 的实测结论。"""
    marks = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "B5"]
    in_report = sorted(criteria_with_conclusion(report_text, marks))
    in_acceptance = sorted(criteria_with_conclusion(acceptance_text, marks))
    detail = ["REPORT_V2_DEEP.md 中**带结论**的判据编号：%s"
              % ("、".join(in_report) or "（无）")]
    if acceptance_text:
        detail.append("验收记录中带结论的判据编号：%s"
                      % ("、".join(in_acceptance) or "（无）"))
    # 规格书原文：「**报告**必须逐条给出 A1~A4 与 B1~B5 的实测结论」
    # ⇒ 判据编号必须出现在**被验收的报告**里；验收记录只是补充，不能替代。
    if len(in_report) == len(marks):
        return Verdict("C1", PASS, "逐条给出 A1~A4 / B1~B5 的实测结论",
                       "报告正文逐条给出了 A1~A4 与 B1~B5 的实测结论", detail)
    missing = [m for m in marks if m not in in_report]
    return Verdict("C1", FAIL, "逐条给出 A1~A4 / B1~B5 的实测结论",
                   "报告正文缺判据编号/结论：%s（验收记录里有也不能替代报告本身）"
                   % "、".join(missing), detail)


def check_c2(ctx, report_text, acceptance_text=""):
    """C2：对外可发布版只保留本轮真正成立的红线，并带齐限定段。"""
    section = section_text(report_text, "### 10.3")
    quote, published, mentioned = outward_clauses(report_text)
    detail = []
    if not quote:
        return Verdict("C2", FAIL, "对外版只保留成立的红线并带齐限定段",
                       "未能从 §10.3 抽出对外发布版正文", [])
    detail.append("对外版正文（§10.3 首段引用）：")
    detail.append("    " + quote[:400])
    detail.append("对外版包含的子句：%s"
                  % ("、".join(sorted(mentioned)) or "（无）"))
    # 哪些指标本轮**真正成立**（与 A3/A4 同口径：不是「有线位」就算成立）
    founded = {}
    founded_reason = {}
    for metric in CORE_METRICS:
        ok, reason = founded_metric(ctx, metric)
        founded_reason[metric] = reason
        if ok:
            founded[metric] = fnum(ctx.line_index[metric].get("line_value"))
    detail.append("本轮**真正成立**的红线：%s"
                  % ("、".join("%s<%s" % (m, fmt(v, 8)) for m, v in founded.items()) or "（无）"))
    for metric in CORE_METRICS:
        if metric not in founded:
            detail.append("    %-24s 不成立：%s" % (metric, founded_reason[metric]))
    detail.append("对照臂 csrt / 已作废参照是否混入对外版：%s"
                  % ("否" if "359.85" not in quote and "141.77" not in quote else "**是**"))

    problems = []
    for metric in mentioned:
        if metric not in published:
            # 🔴 子句提到了阈值却**解析不出数值**（如写成 `≤`）⇒ 记问题，
            #    绝不能把 None 带进后面的算术里（那会让判定器整体崩掉）
            problems.append("对外版 %s 无可解析数值" % metric)
            continue
        value = published[metric]
        if metric not in founded:
            problems.append("对外版含**未成立**的 %s 子句" % metric)
            continue
        # 对外版的门槛取**两轮更严者**（规格书 B5）——deep 更松时不得原样对外
        reference = stricter_line(ctx, metric)[0]
        if abs(value - reference) > 1e-6 * max(1.0, abs(value)):
            problems.append("对外版 %s 数值 %s ≠ 实测线位 %s" % (metric, value, reference))
    omitted = [m for m in founded if m not in mentioned]
    if omitted:
        problems.append("本轮成立但对外版**未收录**：%s" % "、".join(omitted))

    # 限定段
    qualifier_hits = {
        "单臂反解": "单臂反解" in section,
        "画质—跟踪性能分界点（非崩溃点）": ("分界点" in section and "崩溃点" in section),
        "真实好片错误豁免比例": ("错误豁免" in section),
    }
    detail.append("限定段命中：%s"
                  % "；".join("%s=%s" % (k, "✅" if v else "❌") for k, v in qualifier_hits.items()))
    missing_q = [k for k, v in qualifier_hits.items() if not v]
    if missing_q:
        problems.append("对外版缺限定段：%s" % "、".join(missing_q))

    if problems and not omitted:
        return Verdict("C2", FAIL, "对外版只保留成立的红线并带齐限定段",
                       "；".join(problems), detail)
    if problems:
        return Verdict("C2", PARTIAL, "对外版只保留成立的红线并带齐限定段",
                       "未成立的红线与限定段均无问题；但 %s" % "；".join(problems), detail)
    return Verdict("C2", PASS, "对外版只保留成立的红线并带齐限定段",
                   "子句、数值、限定段全部对齐", detail)


def check_c3(ctx):
    """C3：基线轮产物不得被覆盖（deep 走独立目录）。"""
    detail = []
    problems = []
    manifest = os.path.join(ctx.baseline_dir, "manifest_v2.md5")
    if not os.path.exists(manifest):
        return Verdict("C3", FAIL, "基线轮产物不得被覆盖",
                       "找不到基线清单 %s" % manifest, [])
    base_products = []
    base_code = []
    for line in open(manifest, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        digest, path = parts[0], parts[1].strip()
        is_abs = os.path.isabs(path)
        # 🔴 清单里的仓库内条目按**仓库根**解析（清单登记名相对仓库根）⇒ 与 CWD 无关。
        target = path if is_abs else os.path.join(ctx.repo_root, path)
        # 🔴 基线**代码**条目必须用**冻结基线代码镜像**核对（与 ``test_iqa_v2.py``
        #    的 ``_verify_manifest(..., BASELINE_CODE_DIR)`` 同口径）：
        #    工作区的基线代码已**刻意演进**到 deep 轮 ⇒ 拿**活代码**核对一份**历史**
        #    清单必然假失败 ⇒ C3 会被永久判成 PARTIAL ⇒ 这条门禁名存实亡。
        via = ""
        if path.startswith("iqa_v2/") and path.endswith(".py"):
            frozen = os.path.join(ctx.baseline_code_dir, os.path.basename(path))
            if os.path.isfile(frozen):
                target = frozen
                via = "（冻结代码镜像核对）"
        if not os.path.exists(target):
            if is_abs:
                # 清单里登记的服务器侧绝对路径（明细 CSV 不入库），host 上无法核对
                base_code.append("%s ⚠️ 服务器侧路径，host 不可核对" % path)
            else:
                problems.append("清单条目缺失：%s" % path)
            continue
        actual = md5_of(target)
        ok = actual == digest
        entry = "✅" if ok else "❌ 期望 %s 实测 %s" % (digest[:10], actual[:10])
        if "/output/" in path or path.startswith("iqa_v2/output"):
            base_products.append("%s %s" % (path, entry))
        else:
            base_code.append("%s %s%s" % (path, entry, via))
    detail.append("基线**产物**核对（%d 条）：" % len(base_products))
    detail.extend("    " + item for item in base_products)
    bad_products = [item for item in base_products if "❌" in item]
    detail.append("基线**代码**核对（%d 条）：" % len(base_code))
    detail.extend("    " + item for item in base_code)
    bad_code = [item for item in base_code if "❌" in item]
    detail.append("⇒ 基线产物 %d/%d 一致；基线代码 %d/%d 一致"
                  % (len(base_products) - len(bad_products), len(base_products),
                     len(base_code) - len(bad_code), len(base_code)))

    # deep 产物必须落在独立目录
    detail.append("deep 产物目录：%s（独立于 %s）"
                  % (os.path.basename(ctx.deep_dir), os.path.basename(ctx.baseline_dir)))
    if os.path.abspath(ctx.deep_dir) == os.path.abspath(ctx.baseline_dir):
        problems.append("deep 与 baseline 指向同一目录")
    if bad_products:
        problems.append("基线**产物**被改动 %d 条" % len(bad_products))
    if bad_code:
        problems.append("基线**代码**与冻结清单不符 %d 条（%s，产物未动）"
                        % (len(bad_code),
                           "冻结代码镜像被改动 ⇒ 证据链断裂" if "（冻结代码镜像核对）" in "".join(base_code)
                           else "无冻结代码镜像 ⇒ 只能比工作区活代码"))
    status = PASS
    if problems:
        status = FAIL if bad_products else PARTIAL
    return Verdict("C3", status, "基线轮产物不得被覆盖（deep 走独立目录）",
                   "基线产物 %d/%d 逐字节一致；deep 独立目录 %s"
                   % (len(base_products) - len(bad_products), len(base_products),
                      os.path.basename(ctx.deep_dir))
                   + ("；但 %s" % "；".join(problems) if problems else ""), detail)


def check_c4(ctx, gates):
    """C4：门禁全绿（pytest / verify_recalc / md5sum -c / audit_final）。"""
    detail = []
    problems = []
    names = ["pytest", "verify_recalc", "md5sum", "audit_final"]
    for name in names:
        info = gates.get(name)
        if not info:
            detail.append("%-16s 未提供结果" % name)
            problems.append("%s 门禁无结果" % name)
            continue
        rc = info.get("rc")
        ok = rc == 0
        detail.append("%-16s rc=%s %s —— %s" % (name, rc, "✅" if ok else "❌",
                                                 info.get("summary", "")))
        if not ok:
            problems.append("%s 门禁未过（rc=%s）" % (name, rc))
    if problems:
        return Verdict("C4", FAIL, "门禁全绿", "；".join(problems), detail)
    return Verdict("C4", PASS, "门禁全绿", "四道门禁均通过", detail)


# ==========================================================================
# D 系列（附加）：把报告正文里的数字**回头**和 CSV 对账
# —— 目的不是复述报告，而是抓「表格是 deep 数据、段落还是基线轮遗留」的错位
# ==========================================================================
def _cells(line):
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _markdown_tables(section):
    """把一段 markdown 切成若干张表（每张表 = [表头行, 数据行…]）。"""
    tables = []
    current = []
    for line in section.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            if current:
                tables.append(current)
                current = []
            continue
        if set(stripped) <= set("|-: "):
            continue
        current.append(_cells(stripped))
    if current:
        tables.append(current)
    return tables


def _pick_table(tables, *needles):
    """按表头关键字挑一张表（全部关键字都要出现在表头行里）。"""
    for table in tables:
        header = table[0] if table else []
        if all(any(needle in cell for cell in header) for needle in needles):
            return table
    return []


def _markdown_rows(section):
    """兼容旧调用：返回本段所有表格的所有行。"""
    out = []
    for table in _markdown_tables(section):
        out.extend(table)
    return out


def _col(header, *needles):
    """按表头关键字找列号；找不到返回 -1。"""
    for index, cell in enumerate(header):
        if all(needle in cell for needle in needles):
            return index
    return -1


def _num(text):
    text = (text or "").replace("**", "").replace("pp", "")
    text = text.replace("✅", "").replace("❌", "").replace("nan", "nan")
    text = text.replace("无有效红线", "nan").replace("（", " ").replace("）", " ")
    text = text.replace("%", "").replace(",", "").strip()
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _strength_order(family, levels):
    """退化强度递增时的档位顺序（contrast 的 α 越小越重）。"""
    return sorted(levels) if FAMILY_HEAVIEST[family] == "max" else sorted(levels, reverse=True)


def _tables_with_context(text):
    """把整篇 markdown 切成表格，并记住每张表**前面最近**的「……臂」小标题。

    报告里逐档跟住率表按臂分块（**XCorr 臂** / **KCF（legacy） 臂**），
    表头本身不带臂名 ⇒ 必须靠上下文认领。
    """
    out = []
    arm = None
    heading = ""
    current = []
    for line in text.splitlines():
        stripped = line.strip()
        arm_heading = re.search(r"\*\*(?:XCorr|KCF)[^*]{0,20}臂\*\*", stripped)
        if arm_heading:
            arm = "xcorr" if "XCorr" in arm_heading.group(0) else "kcf_legacy"
        if stripped.startswith("#") or (stripped and not stripped.startswith("|")
                                        and not stripped.startswith(">")):
            heading = stripped
        if stripped.startswith("|"):
            if set(stripped) <= set("|-: "):
                continue
            current.append(_cells(stripped))
            continue
        if current:
            out.append((arm, heading, current))
            current = []
    if current:
        out.append((arm, heading, current))
    return out


def check_d1(ctx, report_text):
    """D1：报告里的逐档跟住率表 vs 现场重算。"""
    detail = []
    problems = []
    checked = 0
    tables = _tables_with_context(report_text)
    rates = ctx.level_rates()
    base_rates = ctx.level_rates(ctx.base_detail, ctx.base_common_good)

    def check_column(cells, col, family, arm, table_rates, label):
        """核对一列「族 × 档位」的跟住率读数，返回核对条数。"""
        parts = [x.strip() for x in cells[col].split("→")]
        levels = _strength_order(family, sorted(table_rates.get(family, {})))
        if len(parts) != len(levels):
            problems.append("%s/%s：报告给了 %d 个档位，明细里是 %d 个"
                            % (family, arm, len(parts), len(levels)))
            return 0
        count = 0
        for level, part in zip(levels, parts):
            bucket = table_rates.get(family, {}).get(level)
            if bucket is None:
                problems.append("明细里没有 %s 档位 %s" % (family, level))
                continue
            claimed = _num(part)
            actual = 100.0 * bucket["ok"][arm] / max(1, bucket["n"])
            count += 1
            # 🔴 报告写了但**解析不出数值**（如 `N/A`）⇒ 记问题，不能当无事发生
            if not is_finite(claimed):
                problems.append("%s/%s %s档位 %s：报告写的「%s」无法解析为数值"
                                % (family, arm, label, level, part))
                continue
            if abs(actual - claimed) > 0.06:
                problems.append("%s/%s 档位 %s：报告 %s%% vs 实测 %.2f%%"
                                % (family, arm, level, claimed, actual))
        return count

    for arm_hint, _heading, table in tables:
        header = table[0]
        col = _col(header, "深轮逐档跟住率")
        if col < 0:
            continue
        arm = arm_hint
        if arm is None:
            problems.append("逐档跟住率表无法判定所属臂（表头 %s）" % header)
            continue
        col_base = _col(header, "基线轮逐档跟住率")
        for cells in table[1:]:
            family = cells[0].strip("`")
            if family not in FAMILY_HEAVIEST:
                continue
            checked += check_column(cells, col, family, arm, rates, "深轮")
            # 🔴 基线轮那一列同样要对账（否则基线轮的遗留数字可以蒙混过关）
            if col_base >= 0:
                checked += check_column(cells, col_base, family, arm, base_rates, "基线轮")
            detail.append("%-12s %-9s 复核 %d 个档位"
                          % (arm, family, len(cells[col].split("→"))))
    if checked == 0:
        return Verdict("D1", FAIL, "报告逐档跟住率表与明细一致",
                       "**未能定位到逐档跟住率表**（检查了 %d 张表）——对账无效，"
                       "不能判 PASS" % len(tables), detail)
    if problems:
        return Verdict("D1", FAIL, "报告逐档跟住率表与明细一致",
                       "；".join(problems[:6]) + ("（共 %d 处）" % len(problems) if len(problems) > 6 else ""),
                       detail)
    return Verdict("D1", PASS, "报告逐档跟住率表与明细一致",
                   "共核对 %d 个「族×档位×臂」读数，全部与 degrade_detail 重算一致" % checked,
                   detail)


#: D2 的跨度容差（pp）——报告写两位小数 ⇒ 容差 = 半个末位 + 余量
D2_SPAN_TOL_PP = 0.0055


def check_d2(ctx, report_text):
    """D2：报告里的曲线有效性分级表 vs knee_thresholds。"""
    detail = []
    problems = []
    checked = 0
    tables = _markdown_tables(report_text)
    for table in tables:
        header = table[0]
        col_span = _col(header, "分箱跨度")
        col_level = _col(header, "逐档跨度")
        if col_span < 0 or col_level < 0:
            continue
        deep_span = _col(header, "深轮分箱跨度")
        deep_level = _col(header, "深轮逐档跨度")
        for cells in table[1:]:
            if len(cells) < 2:
                continue
            # 行语义：§5.1 = 臂 + 族（代表曲线 local_scr）；§5.2 = 指标 + 臂 + 族
            first = cells[0].strip("`")
            if first in CORE_METRICS and len(cells) > 2:
                metric, tracker, family = first, cells[1].strip("`"), cells[2].strip("`")
            else:
                metric = "local_scr"
                tracker, family = first, cells[1].strip("`")
            if tracker not in ALL_ARMS or family not in FAMILY_HEAVIEST:
                continue
            info = ctx.grade(tracker, family, metric)
            if not info:
                problems.append("knee_thresholds 无 %s/%s/%s 行" % (metric, tracker, family))
                continue
            span_col = deep_span if deep_span >= 0 else col_span
            level_col = deep_level if deep_level >= 0 else col_level
            claimed_span = _num(cells[span_col])
            claimed_level = _num(cells[level_col])
            actual_span = 100.0 * info["curve_span"]
            actual_level = 100.0 * info["level_span"]
            checked += 1
            # 🔴 报告写了但**解析不出数值** ⇒ 记问题（否则 `abs(x - nan)` 恒为 False ⇒ 假通过）
            if not is_finite(claimed_span) or not is_finite(claimed_level):
                problems.append("%s/%s/%s 跨度：报告「%s / %s」无法解析为数值"
                                % (metric, tracker, family, cells[span_col],
                                   cells[level_col]))
                continue
            if abs(actual_span - claimed_span) > D2_SPAN_TOL_PP:
                problems.append("%s/%s/%s 分箱跨度：报告 %.2fpp vs 实测 %.2fpp"
                                % (metric, tracker, family, claimed_span, actual_span))
            if abs(actual_level - claimed_level) > D2_SPAN_TOL_PP:
                problems.append("%s/%s/%s 逐档跨度：报告 %.2fpp vs 实测 %.2fpp"
                                % (metric, tracker, family, claimed_level, actual_level))
            detail.append("%-24s %-12s %-9s 分箱 %6.2f/%6.2f pp  逐档 %6.2f/%6.2f pp"
                          % (metric, tracker, family, claimed_span, actual_span,
                             claimed_level, actual_level))
    if checked == 0:
        return Verdict("D2", FAIL, "报告曲线分级表与 knee_thresholds 一致",
                       "**未能定位到曲线分级表** —— 对账无效，不能判 PASS", detail)
    if problems:
        return Verdict("D2", FAIL, "报告曲线分级表与 knee_thresholds 一致",
                       "；".join(problems[:6]), detail)
    return Verdict("D2", PASS, "报告曲线分级表与 knee_thresholds 一致",
                   "共核对 %d 条曲线跨度（各指标自己的曲线 + 代表曲线 local_scr），全部一致"
                   % checked, detail)


def check_d3(ctx, report_text):
    """D3：报告里的约束线表 vs constraint_lines_v2.csv。"""
    detail = []
    problems = []
    checked = 0
    tables = _markdown_tables(report_text)
    for table in tables:
        header = table[0]
        col_line = _col(header, "约束线")
        if col_line < 0:
            continue
        col_family = _col(header, "决定族")
        col_arm = _col(header, "决定臂")
        for cells in table[1:]:
            metric = cells[0].strip("`")
            if metric not in CORE_METRICS:
                continue
            row = ctx.line_index.get(metric)
            if not row:
                problems.append("约束线表无 %s" % metric)
                continue
            line = fnum(row.get("line_value"))
            claimed = _num(cells[col_line])
            checked += 1
            if not is_finite(claimed) and not is_finite(line):
                detail.append("%-24s 两边都无有效红线 ✅" % metric)
                continue
            if not is_finite(claimed) or not is_finite(line):
                problems.append("%s 线位：报告 %s vs 实测 %s"
                                % (metric, fmt(claimed, 8), fmt(line, 8)))
                continue
            if abs(claimed - line) > 1e-4 * max(1.0, abs(line)):
                problems.append("%s 线位：报告 %s vs 实测 %s"
                                % (metric, claimed, fmt(line, 10)))
            family = cells[col_family].strip("`") if col_family >= 0 else "?"
            arm = cells[col_arm].strip("`") if col_arm >= 0 else "?"
            if family != (row.get("deciding_family") or "—"):
                problems.append("%s 决定族：报告 %s vs 实测 %s"
                                % (metric, family, row.get("deciding_family")))
            if arm != (row.get("deciding_tracker") or "—"):
                problems.append("%s 决定臂：报告 %s vs 实测 %s"
                                % (metric, arm, row.get("deciding_tracker")))
            detail.append("%-24s 报告 %-12s 实测 %-20s 决定 %s/%s"
                          % (metric, claimed, fmt(line, 10),
                             row.get("deciding_family"), row.get("deciding_tracker")))
    if checked == 0:
        return Verdict("D3", FAIL, "报告约束线表与 constraint_lines 一致",
                       "**未能定位到约束线表** —— 对账无效，不能判 PASS", detail)
    if problems:
        return Verdict("D3", FAIL, "报告约束线表与 constraint_lines 一致",
                       "；".join(problems[:6]), detail)
    return Verdict("D3", PASS, "报告约束线表与 constraint_lines 一致",
                   "共核对 %d 条线位（含决定族/臂），全部一致" % checked, detail)


def check_d4(ctx, report_text):
    """D4：报告里的盲测数字 vs regression_v2.csv。"""
    detail = []
    problems = []
    checked = 0
    tables = _markdown_tables(report_text)
    for table in tables:
        header = table[0]
        col_bad = _col(header, "真实坏片")
        col_good = _col(header, "好片对照")
        col_sep = _col(header, "区分间隔")
        if col_bad < 0 or col_good < 0:
            continue
        for cells in table[1:]:
            metric = cells[0].strip("`")
            if metric not in CORE_METRICS:
                continue
            reg_bad = ctx.reg_index.get(("real_bad", metric))
            reg_good = ctx.reg_index.get(("real_good", metric))
            if not (reg_bad and reg_good):
                continue
            checked += 1
            claimed_bad = _num(cells[col_bad])
            claimed_good = _num(cells[col_good])
            actual_bad = 100.0 * fnum(reg_bad.get("rate_below_line"))
            actual_good = 100.0 * fnum(reg_good.get("rate_below_line"))
            for name, claimed, actual in (("坏片拦截", claimed_bad, actual_bad),
                                          ("真实好片错误豁免", claimed_good, actual_good)):
                # 🔴 解析不出数值 ⇒ 记问题（`abs(nan - x)` 恒 False ⇒ 假通过）
                if not is_finite(claimed):
                    problems.append("%s %s：报告写的「%s」无法解析为数值"
                                    % (metric, name, cells[col_bad if name == "坏片拦截"
                                                           else col_good]))
                    continue
                if abs(claimed - actual) > 0.02:
                    problems.append("%s %s：报告 %.2f%% vs 实测 %.2f%%"
                                    % (metric, name, claimed, actual))
            if col_sep >= 0:
                claimed_sep = _num(cells[col_sep])
                actual_sep = actual_bad - actual_good
                if not is_finite(claimed_sep):
                    problems.append("%s 区分间隔：报告写的「%s」无法解析为数值"
                                    % (metric, cells[col_sep]))
                elif abs(claimed_sep - actual_sep) > 0.02:
                    problems.append("%s 区分间隔：报告 %.2fpp vs 实测 %.2fpp"
                                    % (metric, claimed_sep, actual_sep))
            detail.append("%-24s 坏片 %6.2f%%  好片 %6.2f%%  间隔 %6.2f pp"
                          % (metric, claimed_bad, claimed_good, claimed_bad - claimed_good))
    if checked == 0:
        return Verdict("D4", FAIL, "报告盲测数字与 regression 一致",
                       "**未能定位到盲测表** —— 对账无效，不能判 PASS", detail)
    if problems:
        return Verdict("D4", FAIL, "报告盲测数字与 regression 一致",
                       "；".join(problems[:6]), detail)
    return Verdict("D4", PASS, "报告盲测数字与 regression 一致",
                   "共核对 %d 组盲测数字，全部一致" % checked, detail)


def check_d6(ctx, report_text):
    """D6：报告里的「最重档跟住率」表 vs 现场重算（专抓「取成了最轻档」的错误）。"""
    detail = []
    problems = []
    checked = 0
    tables = _tables_with_context(report_text)
    rates = ctx.level_rates()
    base_rates = ctx.level_rates(ctx.base_detail, ctx.base_common_good)
    for _arm, heading, table in tables:
        header = table[0]
        if not any("最重档" in cell for cell in header) and "最重档" not in heading:
            continue
        for cells in table[1:]:
            family = cells[0].strip("`")
            if family not in FAMILY_HEAVIEST:
                continue
            levels = sorted(rates.get(family, {}))
            if not levels:
                continue
            heaviest = (max(levels) if FAMILY_HEAVIEST[family] == "max" else min(levels))
            lightest = (min(levels) if FAMILY_HEAVIEST[family] == "max" else max(levels))
            bucket_h = rates[family][heaviest]
            bucket_l = rates[family][lightest]
            base_levels = sorted(base_rates.get(family, {}))
            if not base_levels:
                continue
            base_heaviest = (max(base_levels) if FAMILY_HEAVIEST[family] == "max"
                             else min(base_levels))
            base_lightest = (min(base_levels) if FAMILY_HEAVIEST[family] == "max"
                             else max(base_levels))
            base_h = base_rates[family][base_heaviest]
            base_l = base_rates[family][base_lightest]
            for index, cell in enumerate(cells):
                # 该列属于哪条臂？表头同名
                col_head = header[index] if index < len(header) else ""
                arm = None
                for tracker in ALL_ARMS:
                    token = {"xcorr": "xcorr", "kcf_legacy": "kcf", "csrt": "csrt"}[tracker]
                    if token in col_head.lower():
                        arm = tracker
                        break
                if arm is None:
                    continue
                claimed = _num(cell)
                if "基线" in col_head:
                    actual_h = 100.0 * base_h["ok"][arm] / max(1, base_h["n"])
                    actual_l = 100.0 * base_l["ok"][arm] / max(1, base_l["n"])
                else:
                    actual_h = 100.0 * bucket_h["ok"][arm] / max(1, bucket_h["n"])
                    actual_l = 100.0 * bucket_l["ok"][arm] / max(1, bucket_l["n"])
                checked += 1
                # 🔴 报告写了但**解析不出数值** ⇒ 记问题（不能静默跳过）
                if not is_finite(claimed):
                    problems.append("%s / %s：报告写的「%s」无法解析为数值"
                                    % (family, arm, cell))
                    continue
                if abs(claimed - actual_h) <= 0.06:
                    continue
                if abs(claimed - actual_l) <= 0.06:
                    problems.append("%s / %s：报告写的 %.1f%% 其实是**最轻档**的读数，"
                                    "最重档真值是 %.1f%%" % (family, arm, claimed, actual_h))
                else:
                    problems.append("%s / %s：报告 %.1f%% vs 最重档实测 %.1f%%"
                                    % (family, arm, claimed, actual_h))
            detail.append("%-9s 最重档 %-6s xcorr %.1f%%  kcf %.1f%%  csrt %.1f%%"
                          % (family, fmt(heaviest, 4),
                             100.0 * bucket_h["ok"]["xcorr"] / max(1, bucket_h["n"]),
                             100.0 * bucket_h["ok"]["kcf_legacy"] / max(1, bucket_h["n"]),
                             100.0 * bucket_h["ok"]["csrt"] / max(1, bucket_h["n"])))
    if checked == 0:
        return Verdict("D6", FAIL, "报告「最重档跟住率」表与明细一致",
                       "**未能定位到最重档跟住率表** —— 对账无效，不能判 PASS", detail)
    if problems:
        return Verdict("D6", FAIL, "报告「最重档跟住率」表与明细一致",
                       "；".join(problems[:6]) + ("（共 %d 处）" % len(problems) if len(problems) > 6 else ""),
                       detail)
    return Verdict("D6", PASS, "报告「最重档跟住率」表与明细一致",
                   "共核对 %d 个读数，全部等于对应族的**最重档**实测值" % checked, detail)


#: 深轮报告里**不该**再出现的基线轮遗留措辞
#: （逐条对应「基线轮成立、deep 轮已不成立」的事实；命中即报告自相矛盾）
STALE_PATTERNS = [
    (r"若要得到更硬的崩溃拐点，需要\*\*加大退化档位上限\*\*",
     "deep 轮**已经**抬高了上限（σ→8 / L→15 / α→0.05），这段是基线轮遗留"),
    (r"KCF 与 CSRT 的曲线跨度全部 ≤ 5pp",
     "同表 `kcf_legacy/blur` 分箱跨度 10.97 pp（A 级），自相矛盾"),
    (r"只有 XCorr 臂的 `blur` 与 `motion` 两条曲线达到 A 级",
     "同表 `kcf_legacy/blur` 也是 A 级"),
    (r"本轮成立的两条线都只由 XCorr 一条臂支撑",
     "`target_local_contrast` 的决定臂其实是 **kcf_legacy/blur**"),
    (r"只有 XCorr 一条臂通过守卫",
     "应为「两条线各自只由一条臂支撑，且两臂线位差 53.99% / 38.69%」"),
    (r"档位（退化强度递增）",
     "档位列若是 8→0.6，强度是**递减**；表头写反"),
]

#: 「把 15% 当成**跨臂门禁**」的错误说法（必须真的在拿 15% 卡跨臂才算命中）
STALE_CROSS_ARM_GATE = (
    r"跨臂[^。；\n]{0,10}(?:偏差|线位差|差异)[^。；\n]{0,4}(?:15\s*%|<=|≤|<)",
    r"15\s*%[^。；\n]{0,6}跨臂[^。；\n]{0,8}(?:门禁|阈值|合格|达标)",
)

#: D5 关注的「基线轮参照读数」指标（基线轮的 target_lap_var 膝点）
D5_BASELINE_METRIC = "target_lap_var"


def stale_patterns(ctx):
    """按**本轮实测**生成遗留模式清单。"""
    patterns = list(STALE_PATTERNS)
    # 「本轮没有有效红线」只有在本轮**真的**拿到红线时才是遗留
    lap_line = fnum((ctx.line_index.get("target_lap_var") or {}).get("line_value"))
    if is_finite(lap_line):
        patterns.append((
            r"`target_lap_var` 本轮没有有效红线|`target_lap_var` 子句本轮「无数值可填」",
            "deep 轮 `target_lap_var` 已拿到有效红线 %s（A3 PASS）" % fmt(lap_line, 7)))
    patterns.extend(
        (pattern, "仓库里**没有**「跨臂 15% 门禁」：15% 是 Kneedle vs PCHIP 的双算法门禁")
        for pattern in STALE_CROSS_ARM_GATE)
    return patterns


def baseline_knee_values(ctx, metric=D5_BASELINE_METRIC):
    """基线轮某指标的膝点读数（现场从基线 CSV 取，不写死数值）。"""
    out = set()
    for row in ctx.base_knee:
        if row.get("metric") != metric:
            continue
        for col in ("knee_x_candidate", "knee_x_final"):
            value = fnum(row.get(col))
            if is_finite(value):
                out.add(value)
    return out


def check_d5(ctx, report_text):
    """D5：深轮报告里是否还残留基线轮的措辞 / 数字（表与段落错位）。"""
    detail = []
    hits = []
    patterns = stale_patterns(ctx)
    for pattern, why in patterns:
        for found in re.finditer(pattern, report_text):
            line_no = report_text[: found.start()].count("\n") + 1
            hits.append((line_no, pattern, why))
            detail.append("L%-5d %s" % (line_no, why))
    # 基线轮的膝点读数：现场取基线 CSV 的值逐个比对（单个数字命中即算遗留）
    stale_values = baseline_knee_values(ctx)
    for found in re.finditer(r"-?\d+(?:\.\d+)?", report_text):
        value = float(found.group(0))
        for stale in stale_values:
            if abs(value - stale) <= 1e-4 * max(1.0, abs(stale)):
                line_no = report_text[: found.start()].count("\n") + 1
                hits.append((line_no, "基线轮参照读数", "这是**基线轮** %s 的膝点读数 %s"
                             % (D5_BASELINE_METRIC, found.group(0))))
                detail.append("L%-5d 这是**基线轮** %s 的膝点读数 %s"
                              % (line_no, D5_BASELINE_METRIC, found.group(0)))
                break
    if not hits:
        return Verdict("D5", PASS, "报告正文无基线轮遗留措辞",
                       "逐条扫描 %d 个遗留模式 + %d 个基线轮参照读数，全部未命中"
                       % (len(patterns), len(stale_values)), detail)
    return Verdict("D5", FAIL, "报告正文无基线轮遗留措辞（表与段落错位）",
                   "命中 %d 处基线轮遗留（详见下面清单）—— 这些段落与 deep 轮的表格数据"
                   "自相矛盾，对外发布前必须改写" % len(hits), detail)


# ==========================================================================
# 主流程
# ==========================================================================
def build_parser():
    parser = argparse.ArgumentParser(
        description="deep 轮验收判定器（a-priori 判据 A1~A4 / B1~B5 / C1~C4）")
    parser.add_argument("--deep-dir", default="iqa_v2/output_deep",
                        help="deep 轮产物目录")
    parser.add_argument("--baseline-dir", default="iqa_v2/output",
                        help="基线轮产物目录")
    parser.add_argument("--code-dir", default="iqa_v2",
                        help="源码目录（用于读冻结常量）")
    parser.add_argument("--frozen-code-dir", default=None,
                        help="冻结基线代码目录（默认 <code-dir>/server_out_final/files）")
    parser.add_argument("--baseline-code-dir", default=None,
                        help="冻结基线轮代码镜像目录（默认 <code-dir>/_baseline_code_20261126）"
                             "—— C3 的代码条目在这里核对（活代码已演进到 deep 轮）")
    parser.add_argument("--report", default="iqa_v2/REPORT_V2_DEEP.md",
                        help="deep 轮报告（用于 C1/C2 与 D 系列对账）")
    parser.add_argument("--acceptance", default="iqa_v2/REPORT_V2_DEEP_ACCEPTANCE.md",
                        help="本验收记录（用于 C1 自检）")
    parser.add_argument("--gates-json", default=None,
                        help="外部门禁结果 JSON：{\"pytest\":{\"rc\":0,\"summary\":\"...\"}, ...}")
    parser.add_argument("--json-out", default=None, help="把判定结果写成 JSON")
    parser.add_argument("--strict", action="store_true",
                        help="存在非 PASS 判定时返回退出码 1")
    return parser


#: 22 条判据的编号（上下文都建不起来时逐条判不通过）
CRITERION_IDS = ("A1", "A2", "A3", "A4",
                 "B1", "B2", "B3", "B4", "B5", "B6",
                 "C1", "C2", "C3", "C4",
                 "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8")


def run_checks(ctx, report_text, acceptance_text, gates):
    """逐条跑判据。

    🔴 **任何一条判据抛异常都只让那一条判不通过**（异常文本写进证据），
    绝不允许一条判据的异常把整轮判定带崩（那样就一条结论都出不来）。
    """
    checks = (
        ("A1", lambda: check_a1(ctx)),
        ("A2", lambda: check_a2(ctx)),
        ("A3", lambda: check_a3(ctx)),
        ("A4", lambda: check_a4(ctx)),
        ("B1", lambda: check_b1(ctx)),
        ("B2", lambda: check_b2(ctx)),
        ("B3", lambda: check_b3(ctx)),
        ("B4", lambda: check_b4(ctx)),
        ("B5", lambda: check_b5(ctx, report_text)),
        ("B6", lambda: check_b6(ctx)),
        ("C1", lambda: check_c1(ctx, report_text, acceptance_text)),
        ("C2", lambda: check_c2(ctx, report_text, acceptance_text)),
        ("C3", lambda: check_c3(ctx)),
        ("C4", lambda: check_c4(ctx, gates)),
        ("D1", lambda: check_d1(ctx, report_text)),
        ("D2", lambda: check_d2(ctx, report_text)),
        ("D3", lambda: check_d3(ctx, report_text)),
        ("D4", lambda: check_d4(ctx, report_text)),
        ("D5", lambda: check_d5(ctx, report_text)),
        ("D6", lambda: check_d6(ctx, report_text)),
        ("D7", lambda: check_d7(ctx, report_text)),
        ("D8", lambda: check_d8(ctx, report_text)),
    )
    verdicts = []
    for ident, run in checks:
        try:
            verdicts.append(run())
        except Exception as exc:
            verdicts.append(Verdict(ident, FAIL, "判据 %s 的自动判定" % ident,
                                    "判定器内部异常：%s: %s（该判据判为不通过，"
                                    "其余判据不受影响）" % (type(exc).__name__, exc), []))
    return verdicts


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        ctx = Context(args.deep_dir, args.baseline_dir, args.code_dir,
                      args.frozen_code_dir, args.baseline_code_dir)
        # 🔴 报告/验收记录/门禁 JSON 的路径也锚定到**仓库根** ⇒ 判定器整体与 CWD 无关。
        report_text = ""
        if os.path.exists(resolve_path(args.report)):
            report_text = open(resolve_path(args.report), encoding="utf-8").read()
        acceptance_text = ""
        if os.path.exists(resolve_path(args.acceptance)):
            acceptance_text = open(resolve_path(args.acceptance), encoding="utf-8").read()
        gates = {}
        if args.gates_json and os.path.exists(resolve_path(args.gates_json)):
            gates = json.load(open(resolve_path(args.gates_json), encoding="utf-8"))
    except Exception as exc:
        # 连上下文都建不起来 ⇒ 22 条逐条判不通过（而不是崩掉整轮判定）
        verdicts = [Verdict(ident, FAIL, "判据 %s 的自动判定" % ident,
                            "上下文构建失败：%s: %s" % (type(exc).__name__, exc), [])
                    for ident in CRITERION_IDS]
    else:
        verdicts = run_checks(ctx, report_text, acceptance_text, gates)

    print("=" * 100)
    print("deep 轮验收判定（判据来源：iqa_v2/DEEP_AXIS_SPEC_20261127.md，先定规格后看结果）")
    print("deep 产物：%s" % os.path.abspath(args.deep_dir))
    print("基线产物：%s" % os.path.abspath(args.baseline_dir))
    print("=" * 100)
    for verdict in verdicts:
        print(verdict.line())
        for item in verdict.detail:
            print("         · %s" % item)
    counts = defaultdict(int)
    for verdict in verdicts:
        counts[verdict.status] += 1
    print("-" * 100)
    print("合计：PASS %d  PARTIAL %d  FAIL %d（共 %d 条）"
          % (counts[PASS], counts[PARTIAL], counts[FAIL], len(verdicts)))

    if args.json_out:
        payload = [{
            "id": v.ident, "status": v.status, "requirement": v.requirement,
            "evidence": v.evidence, "detail": v.detail,
        } for v in verdicts]
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        print("JSON 已写出：%s" % args.json_out)

    if args.strict and (counts[FAIL] or counts[PARTIAL]):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
