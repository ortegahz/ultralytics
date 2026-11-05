"""从产物 JSON 生成"外挂 vs 原逻辑"对照表（Markdown）。

用法::

    python manu/tracking/lookahead/make_report_table.py \
        --runs runs/lookahead/ref_original.json runs/lookahead/arms_rolling.json \
        --labels "原代码(同帧口径)" "外挂·rolling" --per-sequence

只读产物 JSON，不重跑推理；所有指标直接取自产物（不重新计算）。

两把尺子：

* 尺子1（历史帧等价性）= ``source_frame``：源帧 f 的历史估计对 GT_f 的评测。外挂是纯后处理、
  绝不回写历史估计 ⇒ 同一产物内所有臂的尺子1 F1 必须逐位相等（Δ = 0.000000pp）。
* 尺子2（伺服有效性）= ``current_frame`` 的 F1/P/R/FAR + ``servo`` 的 CLE/命中率/抖动/脱靶：
  评测"物理时刻 t 上报给转台的坐标"对"当前帧真值 GT_t"的误差。

总体（池化）行的 CLE 分位数取**全帧池化**口径 = ``servo.overall.cle_pooled_*``
（把各序列的逐帧 CLE 样本拼接后计算）；产物里的 ``servo.overall.cle_p*`` 是
**逐序列最差值**口径（``worst_over_sequences``），只在逐序列明细表里用。

旧产物可能缺字段（如没有 ``servo``），缺项一律显示 ``—``，不崩。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MISSING = "—"

# 臂名 → 显示名；主表行序按此排列，未列出的臂按名字排在后面
ARM_LABELS = {
    "original": "原逻辑(同帧口径)",
    "stale": "stale(硬吃Δ滞后)",
    "kinematic": "kinematic(纯前向外推)",
    "replay": "replay(视觉重推)",
}
ARM_ORDER = ["original", "stale", "kinematic", "replay"]

# 报告里出现的「物理滞后」毫秒数；由 main() 按 --delta 覆盖（Δ 帧 × 40ms @25Hz）。
LATENCY_MS = 320

# (键, 名称, 数值格式, 差值格式, 方向)；方向 +1 = 越高越好，-1 = 越低越好
# cle_* 五项的数值取自 servo 的**全帧池化**字段（cle_pooled_*，缺池化样本时回退 cle_*）
METRICS = [
    ("f1", "尺子2 F1", "{:.4f}%", "{:+.4f}pp", 1),
    ("precision", "Precision", "{:.4f}%", "{:+.4f}pp", 1),
    ("recall", "Recall", "{:.4f}%", "{:+.4f}pp", 1),
    ("far", "FAR/帧", "{:.4f}", "{:+.4f}", -1),
    ("hit5", "hit@5px", "{:.2f}%", "{:+.2f}pp", 1),
    ("hit_inbox", "hit@inbox", "{:.2f}%", "{:+.2f}pp", 1),
    ("cle_p50", "CLE p50(池)", "{:.2f}px", "{:+.2f}px", -1),
    ("cle_p90", "CLE p90(池)", "{:.2f}px", "{:+.2f}px", -1),
    ("cle_p95", "CLE p95(池)", "{:.2f}px", "{:+.2f}px", -1),
    ("cle_p99", "CLE p99(池)", "{:.2f}px", "{:+.2f}px", -1),
    ("cle_max", "CLE max(池)", "{:.2f}px", "{:+.2f}px", -1),
    ("max_jitter", "max_jitter", "{:.1f}px", "{:+.1f}px", -1),
    ("p99_jitter", "p99_jitter", "{:.1f}px", "{:+.1f}px", -1),
    ("fragmentation", "fragmentation", "{:.0f}", "{:+.0f}", -1),
    ("lost_frames", "lost_frames", "{:.0f}", "{:+.0f}", -1),
    ("coverage", "coverage", "{:.2f}%", "{:+.2f}pp", 1),
]
METRIC_DIR = {key: (name, fmt, diff_fmt, direction) for key, name, fmt, diff_fmt, direction in METRICS}

# 总体行的 CLE 分位数（取全帧池化口径 cle_pooled_<分位>，缺池化样本时回退逐序列最差值 cle_p<分位>）
CLE_QUANTILES = ("p50", "p90", "p95", "p99", "max")


def _dig(doc, *keys):
    """按路径取值，任一层缺失返回 None（容忍旧产物缺字段）。"""
    cur = doc
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _num(value, scale=1.0):
    """取数值并换算单位（比例→百分数）；非数值返回 None。"""
    return None if not isinstance(value, (int, float)) or isinstance(value, bool) else float(value) * scale


def _pooled_cle(servo, quantile):
    """取 servo 的**全帧池化** CLE 分位数（cle_pooled_*）；产物没有池化样本时返回 None。"""
    samples = servo.get("cle_pooled_samples")
    if isinstance(samples, (int, float)) and not isinstance(samples, bool) and samples <= 0:
        return None
    return _num(servo.get(f"cle_pooled_{quantile}"))


def _cle(servo, quantile):
    """总体 CLE 分位数：优先全帧池化口径；老产物/无池化样本时回退逐序列最差值。"""
    pooled = _pooled_cle(servo, quantile)
    return pooled if pooled is not None else _num(servo.get(f"cle_{quantile}"))


class Row:
    """主表的一行 = (产物, 臂)。无 ``arms`` 的产物退化成单行"原脚本产物"。"""

    def __init__(self, run_label, arm, arm_label, doc):
        self.run_label = run_label
        self.arm = arm
        self.arm_label = arm_label
        self.doc = doc
        arm_doc = (doc.get("arms") or {}).get(arm) or {}
        # 尺子1：源帧 f 的历史估计（同一产物内所有臂必须逐位相等）
        self.ruler1_f1 = _num(_dig(arm_doc, "source_frame", "overall", "f1"))
        # 无臂产物只有顶层 overall（就是原脚本的同帧输出）；臂的指标只认自己的 current_frame，缺了显示 —
        overall = (
            _dig(arm_doc, "current_frame", "overall") if arm else _dig(doc, "overall")
        ) or {}
        servo = _dig(arm_doc, "servo", "overall") or {}
        # CLE 五项走**全帧池化**口径（cle_pooled_*）；无池化样本时回退逐序列最差值（cle_p*）
        self.cle_pooled = any(_pooled_cle(servo, q) is not None for q in CLE_QUANTILES)
        self.metrics = {
            "f1": _num(overall.get("f1"), 100.0),
            "precision": _num(overall.get("precision"), 100.0),
            "recall": _num(overall.get("recall"), 100.0),
            "far": _num(overall.get("far_per_frame")),
            "hit5": _num(servo.get("hit_rate_5px"), 100.0),
            "hit_inbox": _num(servo.get("hit_rate_inbox"), 100.0),
            "cle_p50": _cle(servo, "p50"),
            "cle_p90": _cle(servo, "p90"),
            "cle_p95": _cle(servo, "p95"),
            "cle_p99": _cle(servo, "p99"),
            "cle_max": _cle(servo, "max"),
            "max_jitter": _num(servo.get("max_jitter")),
            "p99_jitter": _num(servo.get("p99_jitter")),
            "fragmentation": _num(servo.get("fragmentation")),
            "lost_frames": _num(servo.get("lost_frames")),
            "coverage": _num(servo.get("coverage"), 100.0),
        }
        self.percentile_aggregation = servo.get("percentile_aggregation")

    @property
    def name(self):
        """主表"方案"列：产物名 · 臂名。"""
        return f"{self.run_label} · {self.arm_label}"

    def seq_metrics(self, seq):
        """取本行在单序列上的 (F1%, hit@5px%, CLE p50(逐序), max_jitter)；缺项为 None。"""
        arm_doc = (self.doc.get("arms") or {}).get(self.arm) or {}
        cf = (
            _dig(arm_doc, "current_frame", "per_sequence", seq)
            if self.arm
            else _dig(self.doc, "per_sequence", seq)
        ) or {}
        sv = _dig(arm_doc, "servo", "per_sequence", seq) or {}
        return (
            _num(cf.get("f1"), 100.0),
            _num(sv.get("hit_rate_5px"), 100.0),
            _num(sv.get("cle_p50")),
            _num(sv.get("max_jitter")),
        )


def _load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _build_rows(runs, labels):
    """按 (产物, 臂) 展开成行；产物无 arms 时退化成一行"原脚本产物"。

    缺失/损坏的产物只跳过并记一条警告，不让整张表崩掉。
    """
    rows = []
    skipped = []
    for path, label in zip(runs, labels):
        try:
            doc = _load(path)
        except (OSError, ValueError) as exc:
            skipped.append((path, f"{type(exc).__name__}: {exc}"))
            continue
        arms = doc.get("arms") or {}
        if not arms:
            rows.append(Row(label, "", "原脚本产物", doc))
            continue
        ordered = [a for a in ARM_ORDER if a in arms] + [a for a in arms if a not in ARM_ORDER]
        for arm in ordered:
            rows.append(Row(label, arm, ARM_LABELS.get(arm, arm), doc))
    return rows, skipped


def _ruler1_reference(rows):
    """每个产物的尺子1 参照值：该产物的 original 臂；没有则取该产物内尺子1 F1 的最小值。"""
    refs = {}
    for row in rows:
        if row.ruler1_f1 is None:
            continue
        refs.setdefault(row.run_label, []).append(row.ruler1_f1)
    out = {}
    for label, values in refs.items():
        originals = [r.ruler1_f1 for r in rows if r.run_label == label and r.arm == "original"]
        out[label] = originals[0] if originals else min(values)
    return out


def _fmt(value, fmt):
    return MISSING if value is None else fmt.format(value)


def _delta_text(base, value, diff_fmt):
    """绝对差文本：带符号的差值。"""
    if base is None or value is None:
        return MISSING
    return diff_fmt.format(value - base)


def _relative_text(base, value):
    """相对变化文本；基线为 0 或缺项时无意义。"""
    if base is None or value is None or not base:
        return MISSING
    return f"{(value - base) / base * 100.0:+.2f}%"


def _pick_baseline(rows, baseline_arg):
    """定位基线行：--baseline 可给臂名、臂显示名或产物名；默认取第一个 original 行。"""
    candidates = [r for r in rows if r.arm and r.arm == baseline_arg]
    if not candidates:
        candidates = [r for r in rows if r.arm_label == baseline_arg]
    if not candidates:
        candidates = [r for r in rows if r.run_label == baseline_arg]
    if not candidates:
        candidates = [r for r in rows if r.arm == "original"]
    if not candidates:
        return None
    # 伺服类指标只有挂了外挂的产物才有，优先选带 servo 的那一行
    with_servo = [r for r in candidates if r.metrics["hit5"] is not None]
    return (with_servo or candidates)[0]


def _run_peers(rows):
    """按产物(run)索引同口径的 stale / original 臂。

    跨产物混用基线会算出无意义的「追回 %」（拿 24 序 stale 去比 8 序 replay），
    因此所有同口径比较必须落在**同一产物**内。
    """
    stale_by_run, original_by_run = {}, {}
    for row in rows:
        if row.arm == "stale":
            stale_by_run.setdefault(row.run_label, row)
        elif row.arm == "original":
            original_by_run.setdefault(row.run_label, row)
    return stale_by_run, original_by_run


def _framing_note(rows, baseline):
    """口径声明：讲清「零延迟天花板 / 交付基线 / 追回占比」三件事。

    没有这段声明，读者会把 `original`（离线零延迟、oracle 上界）当成交付基线，
    于是把「外挂没能追回 320 ms 滞后的那部分」误读成「精度被做坏」——
    2026-11-03 用户正是因此判定「结果很差、没办法交付」。数字全部取自行，不写死。
    """
    out = ["### 口径声明（先读这一段，否则结论会做反）", ""]
    out.append("- `original` 臂 = **零延迟天花板（oracle 上界）**：离线可以把检测器跑在每一帧上，"
               f"**没有 {LATENCY_MS} ms 物理滞后** ⇒ 它是运行时**不可能达到**的上界，**不是交付基线**。")
    out.append("- **交付基线 = `stale` 臂**（老逻辑：把第 t−Δ 帧的旧框原样甩给转台）——"
               "这才是今天物理上在跑的行为。")
    out.append("- 因此「退化」一节里的负号 = **未能追回的滞后代价**，**不是「精度被做坏」**。")
    out.append("- 判定「能不能交付」只看两件事：**尺子1 Δ = 0.000000pp**（外挂没动既有跟踪资产）"
               f"＋ 下一节「相对硬吃 {LATENCY_MS}ms 滞后的收益」的**追回占比**。")
    out.append("- 总体表的 **CLE 列 = 全帧池化分位数**（把各序列的逐帧 CLE 样本拼接后计算，"
               "对应产物里的 `cle_pooled_p50/p90/p95/p99/max`）；产物自带的 `worst_over_sequences`"
               "（**逐序列最差值**）口径**不用于总体表**，另见逐序列明细表。")
    out.append(f"- ⚠️ **Δ = {LATENCY_MS} ms 是「假设/标称」延时**（{LATENCY_MS // 40} 帧 @25 Hz），"
               "**不是实测链路延时**；下表所有 Δ 结论都在这个假设下成立。")
    out.append("- **交付三门**（判定能不能交付）：**F1 ≥ 0.88 / Recall ≥ 0.85 / FAR ≤ 0.030**。")
    # ---- 出框门口径：**从本表产物现算**（不写死 ⇒ 产物变了口径也不会说谎）----
    _fovs = sorted({float(((r.doc.get("lookahead") or {}).get("replay_fov_margin_px", 0.0)) or 0.0) for r in rows})
    if len(_fovs) == 1:
        _m = _fovs[0]
        out.append(
            f"- ⚠️ **出框门口径**：本表产物的 `--replay-fov-margin-px` = **{_m:g}**"
            + (
                "（**裸配置，未开出框门**）⇒ 交付推荐档（m=6）的数字**不在本表**，见 `FINAL_REPORT.md` §9.4 / §12。"
                if _m == 0.0
                else "（**含出框门**）⇒ 与裸配置（m=0）的数字不同，裸配置见 `FINAL_REPORT.md` §2.9。"
            )
        )
    else:
        out.append(f"- ⚠️ **出框门口径**：本表产物**混合**了 `--replay-fov-margin-px` = {_fovs} ⇒ 逐行看表头标注。")
    out.append("- ⚠️ **8 序 ≠ 24 序**：8 序 gate 集只覆盖 24 序官方口径的一小部分 GT ⇒ "
               "**8 序只做机理筛选 / A/B，不能给交付档排序**；本表若同时含 8 序与 24 序行，**交付判据以 24 序为准**；"
               "**精确覆盖率每次重算**（见 `runs/lookahead/fi_nogmc/report.md` §0 纪律 6 与 §7 表头）。")
    out.append("")

    # ---- 以下三条**全部从本表产物现算**（不写死任何覆盖率 / 红线结论）----
    _seq_gt: dict[str, dict[str, int]] = {}
    for r in rows:
        arm_doc = (r.doc.get("arms") or {}).get(r.arm) or {}
        ps = (arm_doc.get("current_frame") or {}).get("per_sequence") or r.doc.get("per_sequence") or {}
        if ps:
            _seq_gt.setdefault(r.run_label, {k: (v or {}).get("gt", 0) for k, v in ps.items()})
    _seen_gt_sets: set[tuple] = set()
    for _lbl, _ps in _seq_gt.items():
        _tot = sum(_ps.values())
        if not _tot:
            continue
        _top = max(_ps.items(), key=lambda kv: kv[1])
        _key = (_top[0], _top[1], _tot)
        if _key in _seen_gt_sets:      # 同一 GT 集（不同模型包）只印一次
            continue
        _seen_gt_sets.add(_key)
        out.append(
            f"- 🔴 **GT 集中度（`{_lbl}` 等同一 GT 集，从产物现算）**：逐序 GT 最多的一序 **`{_top[0]}`** 占该集 GT 的 "
            f"**{_top[1] / _tot * 100:.2f}%**（{_top[1]} / {_tot}）⇒ **任何单序列级的效果头条都必须先做逐序列归因**。"
        )
    _replay_fars = [(r.run_label, r.arm, r.metrics.get("far")) for r in rows if r.arm == "replay"]
    _replay_fars = [(lbl, arm, f) for lbl, arm, f in _replay_fars if f is not None]
    if _replay_fars:
        _worst = max(_replay_fars, key=lambda x: x[2])
        _ok = _worst[2] <= 0.030
        out.append(
            f"- 🔴 **FAR 红线（≤ 0.030）核对（从产物现算）**：本表 `replay` 臂 FAR 最大值 = "
            f"**{_worst[2]:.6f}**（`{_worst[0]}`）⇒ **{'✅ 过线' if _ok else '❌ 破线'}**；"
            f"**本表只覆盖 Δ={LATENCY_MS // 40} 档**，其它 Δ 的 FAR 红线结论以 24 序官方口径为准"
            f"（见 `runs/lookahead/fi_nogmc/report.md` §7 与 `FINAL_REPORT.md` §7/§11）。"
        )
    out.append("")

    stale_by_run, original_by_run = _run_peers(rows)
    original = next((r for r in rows if r.arm == "original"), None)
    stale = next((r for r in rows if r.arm == "stale"), None)
    plugins = [r for r in rows if r.arm not in ("", "original", "stale")]
    if original is None or stale is None or not plugins:
        out.append("> （本产物缺 `original` / `stale` 臂，无法给出追回占比。）")
        return out + [""]

    def _f1(row):
        return row.metrics["f1"]

    def _hit5(row):
        return row.metrics["hit5"]

    out.append("| 方案（产物 · 臂） | 含义 | 尺子2 F1 | hit@5px |")
    out.append("|---|---|---|---|")
    out.append(f"| {original.name} | 零延迟天花板（oracle 上界，**非交付基线**） | "
               f"{_fmt(_f1(original), '{:.4f}%')} | {_fmt(_hit5(original), '{:.2f}%')} |")
    out.append(f"| {stale.name} | **交付基线**：老逻辑硬吃 Δ 滞后 | "
               f"{_fmt(_f1(stale), '{:.4f}%')} | {_fmt(_hit5(stale), '{:.2f}%')} |")
    for row in plugins:
        out.append(f"| {row.name} | 外挂臂 | {_fmt(_f1(row), '{:.4f}%')} | "
                   f"{_fmt(_hit5(row), '{:.2f}%')} |")
    out.append("")

    for row in plugins:
        # 基线与天花板必须取**同一产物**的臂，否则「追回 %」是跨集混算的数字。
        base_stale = stale_by_run.get(row.run_label, stale)
        ceiling_row = original_by_run.get(row.run_label, original)
        cells = [f"**{row.name}**"]
        for label, base_row, key in (("F1", base_stale, "f1"), ("hit@5px", base_stale, "hit5")):
            base_value = base_row.metrics[key]
            value = row.metrics[key]
            ceiling = ceiling_row.metrics[key]
            if base_value is None or value is None or ceiling is None:
                cells.append(MISSING)
                continue
            lost = ceiling - base_value
            back = value - base_value
            pct = (back / lost * 100.0) if lost else None
            cells.append(
                f"{base_value:.4f} → {value:.4f}（{back:+.4f}）"
                + (f"，追回 {pct:.1f}%" if pct is not None else "")
            )
        out.append(f"- {' / '.join(cells)}")
    out.append("")

    # 尾部效应：只看 p50 会被骗，必须同时看 p95 —— 纯运动学外推的典型失效模式（池化口径）。
    tail_bits = []
    for row in plugins:
        base_stale = stale_by_run.get(row.run_label, stale)
        for key, label in (("cle_p50", "p50(池)"), ("cle_p95", "p95(池)")):
            base_value = base_stale.metrics[key]
            value = row.metrics[key]
            if base_value is None or value is None:
                continue
            better = value < base_value
            tail_bits.append((row.name, label, base_value, value, better))
    if tail_bits:
        out.append("**尾部效应（务必同时看 p50 与 p95）**")
        out.append("")
        for arm_label, label, base_value, value, better in tail_bits:
            verdict = "改善" if better else "**恶化**"
            out.append(f"- {arm_label} 的 CLE {label}：stale {base_value:.2f}px → {value:.2f}px（{verdict}）")
        out.append("")

    out.append(f"> 读法：`stale → 外挂` 的绝对提升就是**把 {LATENCY_MS} ms 滞后造成的脱靶追回了多少**；"
               "`追回 %` 是它相对天花板的总账里拿回的比例。")
    return out + [""]


def _main_table(rows, refs):
    """主表：行 = (产物, 臂)，列 = 两把尺子的全部指标。"""
    out = [
        "| 方案 | 尺子1 F1(Δ) | 尺子2 F1 | P | R | FAR/帧 | hit@5px | hit@inbox | CLE p50(池) | CLE p90(池) "
        "| CLE p95(池) | CLE p99(池) | CLE max(池) "
        "| max_jitter | p99_jitter | frag | lost | coverage |",
        "|---" * 18 + "|",
    ]
    for row in rows:
        if row.ruler1_f1 is None:
            ruler1 = MISSING
        else:
            ref = refs.get(row.run_label)
            diff = None if ref is None else (row.ruler1_f1 - ref) * 100.0
            ruler1 = f"{row.ruler1_f1 * 100.0:.4f}%（Δ{diff:+.6f}pp）"
        cells = [row.name, ruler1]
        cells += [_fmt(row.metrics[key], METRIC_DIR[key][1]) for key, *_ in METRICS]
        out.append("| " + " | ".join(cells) + " |")
    return out + [""]


def _degradation_table(rows, baseline):
    """「退化」小节：基线 → 各外挂臂，逐指标的绝对差与相对变化。

    基线与被比臂必须来自**同一产物**（8 序 replay 不能拿 24 序 stale 当基线），
    因此这里按产物分组，每个产物出一张表。
    """
    plugins = [r for r in rows if r.arm and r.arm not in ("original",) and r is not baseline]
    if baseline is None or not plugins:
        return ["### 退化（无可对照的外挂臂）", ""]
    stale_by_run, _ = _run_peers(rows)
    groups = {}
    for row in plugins:
        groups.setdefault(row.run_label, []).append(row)
    out = []
    for run_label, group in groups.items():
        base = stale_by_run.get(run_label)
        if base is None and baseline.run_label == run_label:
            base = baseline
        if base is None:
            continue
        header = "| 指标 | 方向 | 基线 |"
        sep = "|---|---|---|"
        for row in group:
            header += f" {row.name} | 变化 |"
            sep += "---|---|"
        if base.arm == "original":
            section = (f"### 未追回的滞后账（基线：{base.name} ＝ **零延迟天花板 / oracle 上界，"
                       f"非交付基线**）")
        else:
            section = f"### 退化（基线：{base.name}）"
        out += [
            section,
            "",
            "箭头按指标方向判定：**↑ = 朝更好的方向走**（F1/P/R/命中率/coverage 越高越好；"
            "FAR/CLE/抖动/碎片/脱靶越低越好），**↓ = 恶化**，**→ = 持平**。",
            "",
            header,
            sep,
        ]
        for key, name, fmt, diff_fmt, direction in METRICS:
            base_value = base.metrics[key]
            cells = [name, "越高越好" if direction > 0 else "越低越好", _fmt(base_value, fmt)]
            for row in group:
                value = row.metrics[key]
                if base_value is None or value is None:
                    cells += [_fmt(value, fmt), MISSING]
                    continue
                diff = value - base_value
                if abs(diff) < 1e-12:
                    arrow, verdict = "→", "持平"
                elif (diff > 0) == (direction > 0):
                    arrow, verdict = "↑", "改善"
                else:
                    arrow, verdict = "↓", "恶化"
                delta = _delta_text(base_value, value, diff_fmt)
                cells += [_fmt(value, fmt), f"{arrow} {verdict}（{delta}，{_relative_text(base_value, value)}）"]
            out.append("| " + " | ".join(cells) + " |")
        out.append("")
    return out or ["### 退化（无可对照的外挂臂）", ""]


def _gain_section(rows):
    """"相对硬吃 Δ 帧滞后的收益"小节：外挂臂 vs 同产物的 stale 臂。"""
    plugins = [r for r in rows if r.arm not in ("", "original", "stale")]
    if not plugins:
        return [f"### 相对硬吃 {LATENCY_MS}ms 滞后的收益（无外挂臂）", ""]
    out = [
        f"### 相对硬吃 {LATENCY_MS}ms 滞后的收益",
        "",
        "| 方案 | F1(臂) | F1(stale) | 追回(pp) | 追回占比 | hit@5px(臂) | hit@5px(stale) | 提升(pp) "
        "| CLE p95(池,臂) | CLE p95(池,stale) | 降幅(px) | 降幅(%) |",
        "|---" * 12 + "|",
    ]
    for row in plugins:
        stale = next((r for r in rows if r.run_label == row.run_label and r.arm == "stale"), None)
        original = next((r for r in rows if r.run_label == row.run_label and r.arm == "original"), None)
        if stale is None:
            out.append(f"| {row.name} | " + " | ".join([MISSING] * 11) + " |")
            continue
        lost = None
        if original is not None and original.metrics["f1"] is not None and stale.metrics["f1"] is not None:
            lost = original.metrics["f1"] - stale.metrics["f1"]
        back = None
        if row.metrics["f1"] is not None and stale.metrics["f1"] is not None:
            back = row.metrics["f1"] - stale.metrics["f1"]
        back_pct = None
        if back is not None and lost:
            back_pct = back / lost * 100.0
        hit_back = None
        if row.metrics["hit5"] is not None and stale.metrics["hit5"] is not None:
            hit_back = row.metrics["hit5"] - stale.metrics["hit5"]
        cle_drop = None
        if row.metrics["cle_p95"] is not None and stale.metrics["cle_p95"] is not None:
            cle_drop = stale.metrics["cle_p95"] - row.metrics["cle_p95"]
        cle_pct = None
        if cle_drop is not None and stale.metrics["cle_p95"]:
            cle_pct = cle_drop / stale.metrics["cle_p95"] * 100.0
        cells = [
            row.name,
            _fmt(row.metrics["f1"], "{:.4f}%"),
            _fmt(stale.metrics["f1"], "{:.4f}%"),
            MISSING if back is None else f"{back:+.4f}pp",
            MISSING if back_pct is None else f"{back_pct:+.1f}%",
            _fmt(row.metrics["hit5"], "{:.2f}%"),
            _fmt(stale.metrics["hit5"], "{:.2f}%"),
            MISSING if hit_back is None else f"{hit_back:+.2f}pp",
            _fmt(row.metrics["cle_p95"], "{:.2f}px"),
            _fmt(stale.metrics["cle_p95"], "{:.2f}px"),
            MISSING if cle_drop is None else f"{cle_drop:+.2f}px",
            MISSING if cle_pct is None else f"{cle_pct:+.1f}%",
        ]
        out.append("| " + " | ".join(cells) + " |")
    for label in dict.fromkeys(r.run_label for r in plugins):
        original = next((r for r in rows if r.run_label == label and r.arm == "original"), None)
        stale = next((r for r in rows if r.run_label == label and r.arm == "stale"), None)
        if original is None or stale is None:
            continue
        if original.metrics["f1"] is None or stale.metrics["f1"] is None:
            continue
        loss = original.metrics["f1"] - stale.metrics["f1"]  # 正数 = 丢掉的总账
        out.append(f"- {label} 的滞后代价：original F1 {original.metrics['f1']:.4f}% → stale F1 {stale.metrics['f1']:.4f}%"
                   f"（**−{loss:.4f}pp**，即外挂要追回的总账）")
    return out + [""]


def _per_sequence_table(rows):
    """逐序列明细：行 = 序列，列组 = (产物 × 臂) × {F1, hit@5px, CLE p50(逐序), max_jitter}。"""
    seqs = []
    for row in rows:
        for seq in (_dig(row.doc, "arms", row.arm, "current_frame", "per_sequence") or
                    _dig(row.doc, "per_sequence") or {}):
            if seq not in seqs:
                seqs.append(seq)
    if not seqs:
        return ["### 逐序列明细（产物里没有序列信息）", ""]
    header = "| 序列 |"
    sep = "|---|"
    for row in rows:
        for field in ("F1", "hit@5px", "CLE p50(逐序)", "max_jitter"):
            header += f" {row.name} {field} |"
            sep += "---|"
    out = ["### 逐序列明细（F1 / hit@5px / CLE p50(逐序) / max_jitter）", "", header, sep]
    for seq in seqs:
        cells = [seq]
        for row in rows:
            f1, hit5, p50, jitter = row.seq_metrics(seq)
            cells += [
                MISSING if f1 is None else f"{f1:.2f}%",
                MISSING if hit5 is None else f"{hit5:.1f}%",
                MISSING if p50 is None else f"{p50:.2f}",
                MISSING if jitter is None else f"{jitter:.1f}",
            ]
        out.append("| " + " | ".join(cells) + " |")
    return out + [""]


def main() -> None:
    ap = argparse.ArgumentParser(description="从产物 JSON 生成外挂 vs 原逻辑对照表（Markdown）")
    ap.add_argument("--runs", nargs="+", required=True, help="产物 JSON 路径")
    ap.add_argument("--labels", nargs="+", required=True, help="与 --runs 一一对应的显示名")
    ap.add_argument("--title", default="总体对照", help="表格标题")
    ap.add_argument("--delta", type=int, default=8, help="检测滞后 Δ（帧），仅用于标题")
    ap.add_argument("--model", default="RC2/trial_0080/epoch3", help="模型标识，仅用于标题")
    ap.add_argument("--per-sequence", action="store_true", help="附带逐序列明细表")
    ap.add_argument("--baseline", default="原逻辑(同帧口径)", help="退化/收益小节的基线标签（臂名或产物名）")
    ap.add_argument("--out-md", default="", help="可选：额外把 Markdown 写到该文件")
    args = ap.parse_args()
    assert len(args.runs) == len(args.labels), "--runs 与 --labels 数量必须一致"

    global LATENCY_MS
    LATENCY_MS = args.delta * 40

    rows, skipped = _build_rows(args.runs, args.labels)
    refs = _ruler1_reference(rows)
    baseline = _pick_baseline(rows, args.baseline)

    lines = [
        f"## {args.title}（Δ={args.delta} 帧 = **标称** {args.delta * 40} ms @25 Hz（**假设值**），{args.model}）",
        "",
        "> **尺子1（历史帧等价性）** = 源帧 f 的历史估计对 GT_f 的评测；外挂是纯后处理、绝不回写历史估计 "
        "⇒ 同一产物内所有臂的尺子1 F1 必须逐位相等（Δ = 0.000000pp）。",
        "> **尺子2（伺服有效性）** = 物理时刻 t 上报给转台的坐标对当前帧真值 GT_t 的误差；"
        "含 current_frame 的 F1/P/R/FAR 与 servo 的 CLE/命中率/抖动/脱靶。",
        "> 所有数字直接取自产物 JSON（未重算）；产物缺字段显示 `—`。",
        "",
    ]
    for path, reason in skipped:
        lines.append(f"> ⚠ 产物不可用，已跳过：`{path}`（{reason}）")
    if not rows:
        lines.append("> 没有可用产物，无法出表。")
    if skipped:
        lines.append("")
    if rows:
        lines += _framing_note(rows, baseline)
        lines += _main_table(rows, refs)
        lines += _degradation_table(rows, baseline)
        lines += _gain_section(rows)
        if args.per_sequence:
            lines += _per_sequence_table(rows)

    agg = {r.percentile_aggregation for r in rows if r.percentile_aggregation}
    if agg:
        lines.append(f"> 注：产物里 `cle_p*` 的聚合方式 = `{sorted(agg)[0]}`（跨序列按**逐序列最差**聚合，非全局分位）；"
                     "总体表的 CLE 列已改用 `cle_pooled_*`（**全帧池化**）口径。")
    fallback = [r.name for r in rows if not r.cle_pooled]
    if fallback:
        lines.append(f"> ⚠ 下列行产物里没有池化样本，CLE 列回退为**逐序列最差值**口径：{'、'.join(fallback)}。")
    lines.append("> 尺子1 Δ 的参照系：同一产物内的 `original` 臂（该产物没有 `original` 臂时取该产物内尺子1 F1 的最小值）。")

    text = "\n".join(lines) + "\n"
    print(text, end="")
    if args.out_md:
        path = Path(args.out_md)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
