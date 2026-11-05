#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""24 序官方口径 Fast Infer 严格对照：把 arms.json 汇成机器可核对的对照表。

读 ``<root24>/<model>/d<delta>/arms.json``（由
``manu/tracking/lookahead/server_overnight/run_fastinfer_24seq_nogmc24_sweep.sh`` 产出），
渲染四张表：

  T1 总体表       —— model × Δ × 臂（original/stale/kinematic/replay）
  T2 Δ 敏感性     —— model × Δ，只看 replay 臂（小延时是否提精度）
  T3 换模型包对照 —— rc2 vs cs_t0043 同 Δ（no-GMC vs GMC 的净收益）
  T4 交付三门     —— F1≥0.88 / Recall≥0.85 / FAR≤0.030 的逐项判定

口径铁律（与 README §5.18.1、manu/memory/rules.md §五一致）：
  * 交付基线臂 = ``stale``；``original`` 是**零延迟天花板**，不是基线。
  * 臂自身的检测指标 = ``arms[arm].current_frame.overall``（源帧窗口另在 ``source_frame``）。
  * 伺服指标 = ``arms[arm].servo.overall``；CLE 总体一律用**全帧池化** ``cle_pooled_*``
    （``cle_p*`` 是逐序列最差值，两者不可混用）。
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from make_fastinfer_nogmc_report import HIST_24SEQ_DELIVERY_D8  # noqa: E402

DELIVERY_GATES = {"f1": 0.88, "recall": 0.85, "far_per_frame": 0.030}
ARMS = ("original", "stale", "kinematic", "replay")
MODEL_LABELS = {
    "rc2": "RC2 trial_0080ep3 / GMC 语义",
    "cs_t0043": "cs_t0043 / no-GMC 语义",
}
# 只有这三个目录算「模型包」；fi_nogmc24 下别的一律不读（防止变体/别的实验污染对照）
MODEL_WHITELIST = ("rc2", "cs_t0043", "trial_0049")
# 本轮的目标 Δ（报告主口径）
TARGET_DELTA = 3

# 实测单帧处理成本（源：manu/memory/nogmc_rebuild_20261027.md §六「成本侧」+ §七.1）
#   带 GMC：183.32 ms/帧；删 GMC（W≡I）：61.78 ms/帧 ⇒ 省 121.54 ms/帧（66.3%），加速 2.97×
#   残留 61.78 ms 里 61.71 ms（99.9%）是 21 帧全图 median（大头是 21 次 warp）
#   帧周期：25 Hz ⇒ 40 ms/record（本项目 Δ 的口径，README §2）
LATENCY = {
    "gmc_ms_per_frame": 183.32,
    "nogmc_ms_per_frame": 61.78,
    "saved_ms": 121.54,
    "saved_ratio": 0.663,
    "speedup": 2.97,
    "frame_period_ms": 40.0,
}


def _num(value):
    """把可能是 None / 字符串的值转成 float，取不到返回 None。"""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_arm(arm_blob: dict) -> dict:
    """从一个臂的 blob 里抽出检测指标 + 伺服指标。"""
    cur = (arm_blob.get("current_frame") or {}).get("overall") or arm_blob.get("overall") or {}
    servo = (arm_blob.get("servo") or {}).get("overall") or {}
    return {
        "f1": _num(cur.get("f1")),
        "recall": _num(cur.get("recall")),
        "precision": _num(cur.get("precision")),
        "far_per_frame": _num(cur.get("far_per_frame")),
        "tp": cur.get("tp"),
        "fp": cur.get("fp"),
        "fn": cur.get("fn"),
        "gt": cur.get("gt"),
        "hit5": _num(servo.get("hit_rate_5px")),
        "hit_inbox": _num(servo.get("hit_rate_inbox")),
        "cle50_pooled": _num(servo.get("cle_pooled_p50")),
        "cle95_pooled": _num(servo.get("cle_pooled_p95")),
        "coverage": _num(servo.get("coverage")),
    }


def load_product(root24: Path) -> dict:
    """扫 <root24>/<model>/d<delta>/arms.json，返回 {(model, delta): payload}。

    只认**白名单里的模型包**（rc2 / cs_t0043 / trial_0049）：``fi_nogmc24`` 下可能混进
    变体目录（``p3knobs``）或别的实验目录，把它们当「模型包」读走会污染对照。
    """
    out = {}
    for model_dir in sorted(p for p in root24.glob("*") if p.is_dir()):
        if model_dir.name not in MODEL_WHITELIST:
            print(f"[SKIP] {model_dir.name}/ 不在模型包白名单 {MODEL_WHITELIST} ⇒ 不计入对照", file=sys.stderr)
            continue
        for arms_file in sorted(model_dir.glob("d*/arms.json")):
            try:
                payload = json.loads(arms_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                print(f"[WARN] 跳过 {arms_file}：{exc}", file=sys.stderr)
                continue
            delta = payload.get("lookahead", {}).get("delta")
            if delta is None:
                try:
                    delta = int(arms_file.parent.name.lstrip("d"))
                except ValueError:
                    delta = None
            per_seq = payload.get("per_sequence") or {}
            out[(model_dir.name, delta)] = {
                "arms": {a: extract_arm(b) for a, b in payload.get("arms", {}).items()},
                "main_arm": payload.get("lookahead", {}).get("main_arm"),
                "path": str(arms_file),
                "elapsed_seconds": payload.get("elapsed_seconds"),
                "gt_total": (payload.get("overall") or {}).get("gt"),
                "gt_per_seq": {k: v.get("gt") for k, v in per_seq.items() if isinstance(v, dict)},
                "seq_names": sorted(per_seq.keys()),
            }
    return out


def gates(metrics: dict) -> dict:
    """交付三门逐项判定（**fail-closed**：指标取不到 ⇒ UNKNOWN，整体不得判 PASS）。"""
    verdict = {}
    for key, limit in DELIVERY_GATES.items():
        value = metrics.get(key)
        if value is None:
            verdict[key] = ("UNKNOWN", None)
            continue
        ok = value >= limit if key in ("f1", "recall") else value <= limit
        verdict[key] = ("PASS" if ok else "FAIL", value)
    overall = "PASS" if all(v[0] == "PASS" for v in verdict.values()) else "FAIL"
    verdict["overall"] = (overall, None)
    return verdict


def _pct(value):
    return "—" if value is None else f"{value * 100:.4f}%"


def _pp(value):
    return "—" if value is None else f"{value * 100:+.4f}pp"


def _f(value, digits=6):
    return "—" if value is None else f"{value:.{digits}f}"


def _delta(new, old, scale=100.0, digits=4):
    if new is None or old is None:
        return "—"
    return f"{(new - old) * scale:+.{digits}f}"


def _replay(products: dict, model: str, delta):
    """取某模型某 Δ 的 replay 臂指标；缺产物返回 None。"""
    entry = products.get((model, delta))
    return entry["arms"].get("replay") if entry else None


def _original(products: dict, model: str, delta=TARGET_DELTA):
    """取某模型在 **target Δ** 的天花板（original 臂）指标；缺失则回退到最小 Δ。"""
    entry = products.get((model, delta))
    if entry and entry["arms"].get("original"):
        return entry["arms"]["original"]
    for d in sorted({x for m, x in products if m == model}, key=lambda x: (x is None, x)):
        entry = products.get((model, d))
        if entry and entry["arms"].get("original"):
            return entry["arms"]["original"]
    return None


def render_summary(products: dict) -> list:
    """§0 结论速览：四个问题全部**自动计算**，不人工抄写。"""
    lines = []
    add = lines.append
    models = sorted({m for m, _ in products})

    add("## §0 结论速览（数字全部由 arms.json 自动提取）")
    add("")

    add("### Q1 · 小延时（Δ=3 / **标称 120 ms**）是否比同模型包 Δ=8（标称 320 ms）精度更高？")
    add("")
    add("| 模型包 | F1(Δ3) | F1(Δ8) | ΔF1 | FAR(Δ3) | FAR(Δ8) | ΔFAR | hit(Δ3) | hit(Δ8) | Δhit | CLE p95(Δ3) | CLE p95(Δ8) |")
    add("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for model in models:
        r3, r8 = _replay(products, model, 3), _replay(products, model, 8)
        if not r3 or not r8:
            continue
        add(
            f"| {model} | {_pct(r3['f1'])} | {_pct(r8['f1'])} | {_delta(r3['f1'], r8['f1'])} | "
            f"{_f(r3['far_per_frame'])} | {_f(r8['far_per_frame'])} | "
            f"{_delta(r3['far_per_frame'], r8['far_per_frame'], 1.0, 6)} | "
            f"{_pct(r3['hit5'])} | {_pct(r8['hit5'])} | {_delta(r3['hit5'], r8['hit5'])} | "
            f"{_f(r3['cle95_pooled'], 3)} | {_f(r8['cle95_pooled'], 3)} |"
        )
    add("")
    add(
        "> 口径脚注：① Δ 是 **record 下标空间**的滞后；换算成物理时延按 record 间隔的**中位数**："
        "Δ=3 ⇒ 中位 **120 ms**（99.74% 的转移恰好 3 帧），Δ=8 ⇒ 中位 320 ms；长尾（个别序列上百帧）来自标注稀疏段。"
        "**两侧模型包共用同一 record 流 ⇒ 对比结论不受影响**。"
        "② 「历史交付档」只指 **rc2 Δ=8**（与 README §5.18.1 逐位一致）；cs_t0043 的 Δ=8 是本轮新产物。"
    )
    add("")

    add("### Q2 · 换成 no-GMC 模型包（cs_t0043）相对 rc2 / GMC 的净收益（同 Δ）")
    add("")
    if "rc2" in models and "cs_t0043" in models:
        add("| Δ | ΔF1(cs−rc2) | ΔFAR(cs−rc2) | Δhit(cs−rc2) | ΔCLE p95(cs−rc2) | F1 谁高 | FAR 谁低 |")
        add("|--:|--:|--:|--:|--:|:--|:--|")
        for delta in sorted({d for m, d in products}, key=lambda x: (x is None, x)):
            lm, rm = _replay(products, "rc2", delta), _replay(products, "cs_t0043", delta)
            if not lm or not rm:
                continue
            add(
                f"| {delta} | {_delta(rm['f1'], lm['f1'])} | "
                f"{_delta(rm['far_per_frame'], lm['far_per_frame'], 1.0, 6)} | "
                f"{_delta(rm['hit5'], lm['hit5'])} | "
                f"{_delta(rm['cle95_pooled'], lm['cle95_pooled'], 1.0, 3)} | "
                f"{'cs_t0043' if rm['f1'] > lm['f1'] else ('rc2' if lm['f1'] > rm['f1'] else '打平')} | "
                f"{'cs_t0043' if rm['far_per_frame'] < lm['far_per_frame'] else ('rc2' if lm['far_per_frame'] < rm['far_per_frame'] else '打平')} |"
            )
        add("")
    else:
        add("> ⚠️ cs_t0043 的 24 序产物尚未齐备 ⇒ 本表待 24 序 Δ 扫描出数后重跑。")
        add("")

    add("### Q3 · Δ=3 的 replay 臂过不过交付三门（F1≥0.88 / Recall≥0.85 / FAR≤0.030）？")
    add("")
    add("| 模型包 | F1 | Recall | FAR/帧 | 三门 | 相对 stale 的 F1 增益 | 相对天花板的 F1 缺口 |")
    add("|---|--:|--:|--:|:--:|--:|--:|")
    for model in models:
        r3 = _replay(products, model, 3)
        if not r3:
            continue
        entry = products.get((model, 3))
        stale = entry["arms"].get("stale") if entry else None
        ceiling = _original(products, model)
        g = gates(r3)
        add(
            f"| {model} | {_pct(r3['f1'])} | {_pct(r3['recall'])} | {_f(r3['far_per_frame'])} | {g['overall'][0]} | "
            f"{_delta(r3['f1'], stale['f1']) if stale else '—'}pp | "
            f"{_delta(r3['f1'], ceiling['f1']) if ceiling else '—'}pp |"
        )
    add("")

    add("### Q4 · Δ 假设的**时延预算依据**（**预算口径 · 非实测**）")
    add("")
    add("| 特征语义 | 单帧处理成本 | 相当于几个 40 ms 帧周期 | 预算口径下的推论 |")
    add("|:--|--:|--:|:--|")
    add(
        f"| GMC（rc2） | {LATENCY['gmc_ms_per_frame']} ms | "
        f"{LATENCY['gmc_ms_per_frame'] / LATENCY['frame_period_ms']:.2f} | "
        "单帧就吃掉 4 个多帧周期 ⇒ 历史上必须取 Δ=8（320 ms）才追得回 |"
    )
    add(
        f"| no-GMC（cs_t0043） | {LATENCY['nogmc_ms_per_frame']} ms | "
        f"{LATENCY['nogmc_ms_per_frame'] / LATENCY['frame_period_ms']:.2f} | "
        f"省 {LATENCY['saved_ms']} ms/帧（{LATENCY['saved_ratio'] * 100:.1f}%，加速 {LATENCY['speedup']}×）"
        " ⇒ 单帧预算上**支持**取 Δ=3（标称 120 ms） |"
    )
    add("")
    add(
        "> ⚠️ **免责**：本表是**单帧处理成本**的**预算口径**论证（常数来自 "
        "`manu/memory/nogmc_rebuild_20261027.md` §六 的实测：183.32 → 61.78 ms/帧），"
        "**不是本轮的实测时延**；真机时延本轮**未实测**（板卡侧见 `manu/memory/gmc_40ms_handles_rk3588.md`）。"
    )
    add(
        "> 「Δ=3 假设成立」的**精度收益**只由 §0 Q1 / T2 的 **24 序实测数**支撑；"
        "上表只能说明「no-GMC 付得起 Δ=3」这个**前提**，不能替代精度实测。"
    )
    if "cs_t0043" not in models:
        add(
            "> ⚠️ cs_t0043 的 24 序精度数**尚未出** ⇒ 本行目前只是预算参考，"
            "「换 no-GMC 后 Δ=3 到底赚多少精度」要等 Q1 补上 cs_t0043 行才能回答。"
        )
    add("")
    return lines


def render_uniqueness_check(products: dict) -> list:
    """§0.5 「唯一变量 = {模型包, Δ}」的**产物级自证**。

    记录集合 = 候选帧 ∪ GT 帧，候选帧来自**各自的缓存** ⇒ 两个模型包原则上可能评的不是同一批帧。
    所以必须逐位核对：① GT 总数 ② 逐序列 GT ③ FAR 的隐含分母（视频总帧数）。
    """
    md = []
    add = md.append
    add("## §0.5 「唯一变量 = {模型包, Δ}」的产物级自证")
    add("")
    have = {m for m, _ in products}  # products 的键是 (model, delta) 元组 ⇒ 不能直接 `in products`
    if "rc2" not in have or "cs_t0043" not in have:
        add("> ⚠️ 两个模型包的 24 序产物尚未齐备 ⇒ 本节待出数后自证。")
        add("")
        return md

    add("| Δ | rc2 GT总数 | cs GT总数 | GT 相同 | 逐序列 GT 相同 | 序列集合相同 | rc2 FAR 分母(帧) | cs FAR 分母(帧) | 分母相同 | 判定 |")
    add("|--:|--:|--:|:--:|:--:|:--:|--:|--:|:--:|:--:|")
    all_ok = True
    for delta in sorted({d for m, d in products}, key=lambda x: (x is None, x)):
        left, right = products.get(("rc2", delta)), products.get(("cs_t0043", delta))
        if not left or not right:
            continue
        gt_same = left.get("gt_total") == right.get("gt_total")
        seq_gt_same = left.get("gt_per_seq") == right.get("gt_per_seq")
        seq_set_same = left.get("seq_names") == right.get("seq_names")
        lf = left["arms"].get("replay") or {}
        rf = right["arms"].get("replay") or {}

        def _denom(m):
            fp, far = m.get("fp"), m.get("far_per_frame")
            if not fp or not far:
                return None
            return round(fp / far, 3)

        ld, rd = _denom(lf), _denom(rf)
        den_same = ld is not None and rd is not None and abs(ld - rd) < 1.0
        ok = gt_same and seq_gt_same and seq_set_same and den_same
        all_ok = all_ok and ok
        add(
            f"| {delta} | {left.get('gt_total')} | {right.get('gt_total')} | {'✅' if gt_same else '❌'} | "
            f"{'✅' if seq_gt_same else '❌'} | {'✅' if seq_set_same else '❌'} | {ld} | {rd} | "
            f"{'✅' if den_same else '❌'} | {'✅ 通过' if ok else '❌ 不通过'} |"
        )
    add("")
    add(
        "> **总判定：" + ("✅ 两侧评的是同一批帧（GT 总数 / 逐序列 GT / 序列集合 / FAR 分母全部逐位一致）⇒ 唯一变量确实只有 {模型包, Δ}。" if all_ok else "❌ 两侧的评测帧集合或 FAR 分母不一致 ⇒ 唯一变量论证不成立，必须先查清。")
    )
    add("")
    return md


def render(products: dict, anchor_check: dict) -> str:
    """渲染四张表（T1~T4）。"""
    models = sorted({m for m, _ in products})
    deltas = sorted({d for _, d in products}, key=lambda x: (x is None, x))
    lines = []
    add = lines.append

    add("# 24 序官方口径 · Fast Infer 严格对照（自动生成）")
    add("")
    add(f"生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}　执行机：{platform.node()}")
    add("")
    lines.extend(render_summary(products))
    lines.extend(render_uniqueness_check(products))

    # ---- T1 总体表 ----
    add("## T1 总体表（model × Δ × 臂）")
    add("")
    add("| 模型包 | Δ | 臂 | F1% | Recall% | Precision% | FAR/帧 | hit@5px% | CLE p50(池) | CLE p95(池) | coverage | 交付三门 |")
    add("|---|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|:--:|")
    for model in models:
        for delta in deltas:
            entry = products.get((model, delta))
            if not entry:
                continue
            for arm in ARMS:
                metrics = entry["arms"].get(arm)
                if not metrics:
                    continue
                g = gates(metrics)
                add(
                    f"| {model} | {delta} | {arm} | {_pct(metrics['f1'])} | {_pct(metrics['recall'])} | "
                    f"{_pct(metrics['precision'])} | {_f(metrics['far_per_frame'])} | {_pct(metrics['hit5'])} | "
                    f"{_f(metrics['cle50_pooled'], 3)} | {_f(metrics['cle95_pooled'], 3)} | "
                    f"{_pct(metrics['coverage'])} | {g['overall'][0]} |"
                )
    add("")

    # ---- T2 Δ 敏感性（replay 臂） ----
    add("## T2 Δ 敏感性（replay 臂；小延时是否提精度）")
    add("")
    add("| 模型包 | Δ | 延时(ms) | replay F1% | replay FAR/帧 | replay hit@5px% | replay CLE p95(池) | F1 vs stale(pp) | FAR vs stale | 三门 |")
    add("|---|--:|--:|--:|--:|--:|--:|--:|--:|:--:|")
    for model in models:
        for delta in deltas:
            entry = products.get((model, delta))
            if not entry:
                continue
            replay = entry["arms"].get("replay")
            stale = entry["arms"].get("stale")
            if not replay:
                continue
            g = gates(replay)
            add(
                f"| {model} | {delta} | {delta * 40} | {_pct(replay['f1'])} | {_f(replay['far_per_frame'])} | "
                f"{_pct(replay['hit5'])} | {_f(replay['cle95_pooled'], 3)} | "
                f"{_delta(replay['f1'], stale['f1']) if stale else '—'} | "
                f"{_delta(replay['far_per_frame'], stale['far_per_frame'], 1.0, 6) if stale else '—'} | "
                f"{g['overall'][0]} |"
            )
    add("")

    # ---- T3 换模型包核心对照（同 Δ 跨模型） ----
    add("## T3 换模型包核心对照（rc2 / GMC  vs  cs_t0043 / no-GMC，同 Δ）")
    add("")
    if "rc2" not in models or "cs_t0043" not in models:
        add("> ⚠️ 两边模型包尚未齐备（本表暂缺），等 24 序 cs_t0043 缓存/扫描出数后重跑。")
        add("")
    else:
        add("| Δ | 臂 | rc2 F1% | cs F1% | ΔF1(pp) | rc2 FAR | cs FAR | ΔFAR | rc2 hit% | cs hit% | Δhit(pp) | rc2 CLE p95 | cs CLE p95 |")
        add("|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
        for delta in deltas:
            left = products.get(("rc2", delta))
            right = products.get(("cs_t0043", delta))
            if not left or not right:
                continue
            for arm in ARMS:
                lm = left["arms"].get(arm)
                rm = right["arms"].get(arm)
                if not lm or not rm:
                    continue
                add(
                    f"| {delta} | {arm} | {_pct(lm['f1'])} | {_pct(rm['f1'])} | {_delta(rm['f1'], lm['f1'])} | "
                    f"{_f(lm['far_per_frame'])} | {_f(rm['far_per_frame'])} | {_delta(rm['far_per_frame'], lm['far_per_frame'], 1.0, 6)} | "
                    f"{_pct(lm['hit5'])} | {_pct(rm['hit5'])} | {_delta(rm['hit5'], lm['hit5'])} | "
                    f"{_f(lm['cle95_pooled'], 3)} | {_f(rm['cle95_pooled'], 3)} |"
                )
        add("")

    # ---- T4 交付三门 ----
    add("## T4 交付三门逐项（F1≥0.88 / Recall≥0.85 / FAR≤0.030）")
    add("")
    add("| 模型包 | Δ | 臂 | F1 | F1门 | Recall | Recall门 | FAR/帧 | FAR门 | 总判定 |")
    add("|---|--:|---|--:|:--:|--:|:--:|--:|--:|:--:|")
    for model in models:
        for delta in deltas:
            entry = products.get((model, delta))
            if not entry:
                continue
            for arm in ARMS:
                metrics = entry["arms"].get(arm)
                if not metrics:
                    continue
                g = gates(metrics)
                add(
                    f"| {model} | {delta} | {arm} | {_pct(metrics['f1'])} | {g['f1'][0]} | "
                    f"{_pct(metrics['recall'])} | {g['recall'][0]} | {_f(metrics['far_per_frame'])} | "
                    f"{g['far_per_frame'][0]} | {g['overall'][0]} |"
                )
    add("")

    # ---- 对齐锚点 ----
    add("## 对齐锚点（rc2 Δ=8 replay vs README §5.18.1「当前交付档」权威数）")
    add("")
    anchor = anchor_check.get("rc2_d8_replay")
    if anchor is None:
        add("> ⚠️ 缺 rc2 Δ=8 产物，锚点未校。")
    else:
        ref = HIST_24SEQ_DELIVERY_D8
        add("| 量 | 本次实测 | README §5.18.1 权威值 | 逐位一致 |")
        add("|---|--:|--:|:--:|")
        for key, label, digits in (("f1", "F1", 6), ("far", "FAR/帧", 6), ("hit5", "hit@5px", 6)):
            got = anchor.get(key)
            want = ref.get(key)
            same = got is not None and want is not None and abs(got - want) < 5e-7
            add(f"| {label} | {_f(got, digits)} | {_f(want, digits)} | {'✅ 是' if same else '❌ 否'} |")
    add("")
    return "\n".join(lines)


def anchor_check(products: dict) -> dict:
    """校准 rc2 Δ=8 replay 与 README 权威档是否逐位一致。"""
    entry = products.get(("rc2", 8))
    if not entry:
        return {}
    replay = entry["arms"].get("replay", {})
    return {
        "rc2_d8_replay": {
            "f1": replay.get("f1"),
            "far": replay.get("far_per_frame"),
            "hit5": replay.get("hit5"),
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="24 序 Fast Infer 严格对照汇总")
    parser.add_argument("--root-24", default="runs/lookahead/fi_nogmc24")
    parser.add_argument("--out-md", default="")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-memory", default="")
    args = parser.parse_args()

    root24 = Path(args.root_24)
    if not root24.is_dir():
        print(f"[ERROR] 找不到产物根 {root24}")
        raise SystemExit(2)

    products = load_product(root24)
    if not products:
        print(f"[ERROR] {root24} 下没有可用的 arms.json")
        raise SystemExit(3)

    anchors = anchor_check(products)
    md = render(products, anchors)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "machine": platform.node(),
        "root_24": str(root24),
        "anchor_check": anchors,
        "anchor_reference": HIST_24SEQ_DELIVERY_D8,
        "delivery_gates": DELIVERY_GATES,
        "products": {
            f"{model}/d{delta}": {
                "path": entry["path"],
                "main_arm": entry["main_arm"],
                "elapsed_seconds": entry["elapsed_seconds"],
                "arms": entry["arms"],
                "delivery_verdict": {a: gates(m)["overall"][0] for a, m in entry["arms"].items()},
            }
            for (model, delta), entry in sorted(products.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0))
        },
    }

    if args.out_json:
        path = Path(args.out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] summary json -> {path}")

    if args.out_md:
        path = Path(args.out_md)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(md + "\n", encoding="utf-8")
        print(f"[OK] summary md   -> {path}")

    if args.out_memory:
        path = Path(args.out_memory)
        path.parent.mkdir(parents=True, exist_ok=True)
        cmd = f"python manu/tracking/lookahead/server_overnight/summarize_nogmc24_results.py --root-24 {args.root_24}"
        path.write_text(
            "\n".join(
                [
                    "# Fast Infer × no-GMC（24 序官方口径）· 自动结果档",
                    "",
                    f"> 自动生成于 {datetime.now():%Y-%m-%d %H:%M:%S}，执行机 `{platform.node()}`。",
                    "> 数字**全部来自 arms.json 自动提取**，未经人工抄写。",
                    "",
                    "## 复现命令（铁律九）",
                    "",
                    "```bash",
                    f"cd {os.environ.get('REPO', '/tmp/pycharm_project_10ae9e2e')}",
                    cmd,
                    "```",
                    "",
                    "## 产物路径",
                    "",
                    f"- 产物根：`{root24}`",
                    f"- 汇总 JSON：`{args.out_json or '(未写)'}`",
                    f"- 汇总 MD：`{args.out_md or '(未写)'}`",
                    "",
                    "---",
                    "",
                    md,
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[OK] memory doc   -> {path}")

    if not (args.out_md or args.out_json or args.out_memory):
        print(md)


if __name__ == "__main__":
    main()
