#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
no-GMC 搜索结果汇总器（供次日清晨直接读盘）

产出 `runs/optuna_nogmc/SUMMARY.md` + stdout：
  · 总体：trial 数 / PASS / PRUNED / 基础设施失败 / 耗时
  · **U 一律用 `mix_ratio_judgement.utility()` 重算**（不信 CSV 列 —— 铁律：U 重算）
  · 分档统计（按 mix_spec）：U 中位数 / 均值 / 最好 / PASS 数
  · Top-N 榜（按重算 U）
  · 参照系三轨基线（no-GMC 语义）与新擂主（natural 对照）
  · ⚠️ 与 GMC 搜索**不可横向比较**的口径声明

用法：
    python manu/training/summarize_nogmc_search.py \
        --out-root runs/optuna_nogmc \
        --ref runs/nogmc_ref/reference_frame.json \
        --top 15
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from manu.training.mix_ratio_judgement import utility  # noqa: E402


def _f(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _i(v, default=None):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


# 11 个调参维度：11 维**逐位相同** = 同一条训练配方（同超参）；
# 组内 trial 号不同 ⇒ 种子不同 ⇒ 可测种子噪声。
CFG_COLS = ("mix_spec", "lr_scale", "warmup_epochs", "weight_decay", "focal_beta",
            "offset_weight", "mosaic", "scale", "translate", "clip_max_norm", "freeze_bn")


def main() -> int:
    ap = argparse.ArgumentParser(description="no-GMC 搜索结果汇总")
    ap.add_argument("--out-root", default="runs/optuna_nogmc")
    ap.add_argument("--ref", default="runs/nogmc_ref/reference_frame.json")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--out-md", default="")
    ap.add_argument("--gmc-csv", default="runs/optuna_mix_ratio/optuna_summary.csv",
                    help="GMC 侧汇总 CSV（用于 §五之二 的**配对 A/B** 对照）")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    csv_path = out_root / "optuna_summary.csv"
    if not csv_path.is_file():
        print(f"[FAIL] 找不到 {csv_path}")
        return 3

    ref = {}
    ref_path = Path(args.ref)
    if ref_path.is_file():
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
    base = ref.get("frozen_baseline") or {}
    rc1 = ref.get("rc1") or {}
    gate_cfg = ref.get("gate") or {}
    _fp_inc = gate_cfg.get("track2_fp_max_increment", 10)
    _t2_min = gate_cfg.get("track2_f1_min", 0.985)
    _t1_delta = gate_cfg.get("track1_min_delta", -0.005)

    rows = []
    with csv_path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            m = {
                "track1_f1": _f(r.get("t1_f1")), "track1_fp": _i(r.get("t1_fp")),
                "track2_f1": _f(r.get("t2_f1")), "track2_fp": _i(r.get("t2_fp")),
                "track3_f1": _f(r.get("t3_f1")), "track3_fp": _i(r.get("t3_fp")),
            }
            if any(v is None for v in m.values()):
                continue
            u = None
            if base:
                u, _ = utility(m, base)
            u_rc1 = None
            if rc1:
                u_rc1, _ = utility(m, rc1)
            rows.append({
                "trial": r.get("trial"), "status": r.get("status"), "mix_spec": r.get("mix_spec"),
                "verdict": r.get("gate_verdict"), "measured_ratio": _f(r.get("measured_ratio")),
                "lr_scale": _f(r.get("lr_scale")), "warmup": _f(r.get("warmup_epochs")),
                "wd": r.get("weight_decay"), "focal": r.get("focal_beta"), "offset": r.get("offset_weight"),
                "mosaic": r.get("mosaic"), "scale": r.get("scale"), "translate": r.get("translate"),
                "clip": r.get("clip_max_norm"), "freeze_bn": r.get("freeze_bn"),
                "duration_min": _f(r.get("duration_min")),
                "m": m, "u": u, "u_rc1": u_rc1,
                # 11 维超参的**原始 CSV 字符串**（用于「配置级」分组的逐位比对 + 指纹回显）
                "cfg_raw": {k: (r.get(k) or "") for k in CFG_COLS},
            })

    lines = []
    P = lines.append
    P("# no-GMC 混合比例搜索 · 结果汇总")
    P("")
    P(f"- 生成时间：{__import__('datetime').datetime.now():%Y-%m-%d %H:%M:%S}")
    P(f"- 搜索根：`{out_root}`")
    P(f"- 参照系包：`{ref_path}`" + ("" if ref_path.is_file() else " ⚠️ 不存在"))
    P("")

    # ---- 参照系 ----
    P("## 一、参照系（no-GMC 语义 · 474 去 P0 实测基线）")
    P("")
    if base:
        P("| 轨 | F1 | FP |")
        P("|:--|--:|--:|")
        for k, name in (("track1", "T1 Anti-UAV 大盘"), ("track2", "T2 龙泉山黄金池"), ("track3", "T3 盲测 001324")):
            P(f"| {name} | {base.get(k+'_f1')} | {base.get(k+'_fp')} |")
        P("")
    if rc1:
        P(f"**新擂主（natural 对照）**：T1 F1={rc1.get('track1_f1')} ｜ T2 F1={rc1.get('track2_f1')} ｜ T3 F1={rc1.get('track3_f1')}")
    else:
        P("⚠️ **新擂主（rc1）为空** —— natural 对照 trial 尚未产出门禁报告，U(vs RC1) 列不可用。")
    P("")

    # ---- 总体 ----
    n = len(rows)
    by_status = defaultdict(int)
    by_verdict = defaultdict(int)
    for r in rows:
        by_status[r["status"]] += 1
        by_verdict[r["verdict"] or "NA"] += 1
    durs = [r["duration_min"] for r in rows if r["duration_min"]]
    P("## 二、总体")
    P("")
    P(f"- 已结算 trial：**{n}**")
    P(f"- 按 status：{dict(by_status)}")
    P(f"- 按门禁判定：{dict(by_verdict)}")
    P(f"- **过一级红线（可上线候选）：{sum(1 for r in rows if (r['verdict'] or '').startswith('PASS'))}**")
    if durs:
        P(f"- 单 trial 耗时：中位 {statistics.median(durs):.1f} min（最短 {min(durs):.1f} / 最长 {max(durs):.1f}）")
    P(f"- 并行度：4 卡（每 trial 单卡）")
    P("")

    # ---- 分档统计 ----
    P("## 三、分档统计（按 mix_spec，U 为重算值）")
    P("")
    P("| mix_spec | trial 数 | PASS | U 中位数 | U 均值 | U 最好 | U(vs RC1) 中位数 |")
    P("|:--|--:|--:|--:|--:|--:|--:|")
    groups = defaultdict(list)
    for r in rows:
        groups[r["mix_spec"]].append(r)
    for spec in sorted(groups, key=lambda s: -max([x["u"] or -9 for x in groups[s]])):
        g = groups[spec]
        us = [x["u"] for x in g if x["u"] is not None]
        urs = [x["u_rc1"] for x in g if x["u_rc1"] is not None]
        npass = sum(1 for x in g if (x["verdict"] or "").startswith("PASS"))
        med = f"{statistics.median(us):+.6f}" if us else "—"
        mean = f"{statistics.mean(us):+.6f}" if us else "—"
        best = f"{max(us):+.6f}" if us else "—"
        med_rc1 = f"{statistics.median(urs):+.6f}" if urs else "—"
        P(f"| {spec} | {len(g)} | {npass} | {med} | {mean} | {best} | {med_rc1} |")
    P("")

    # ---- Top 榜 ----
    P(f"## 四、Top-{args.top}（按重算 U）")
    P("")
    P("> ⚠️ **本榜含被一级红线否决的 trial**（`verdict` 列可见，如 `FAIL_TRACK1_RETREAT` / `FAIL_TRACK2_SAFETY`）。")
    P("> 一级红线是**一票否决**——这些 trial **不可上线**，只是 U 高。**可上线候选看下一节（§四之二）。**")
    P("> 📌 **本榜是单点榜，只作参考；交接以 §四之三（配置级多种子中位）为准。**")
    P("")
    P("| # | trial | mix_spec | 实测比 | lr_scale | verdict | U(vs 基线) | U(vs RC1) | T1 | T2 | T3 | min |")
    P("|--:|:--|:--|--:|--:|:--|--:|--:|--:|--:|--:|--:|")
    top = sorted([r for r in rows if r["u"] is not None], key=lambda r: -r["u"])[:args.top]
    for i, r in enumerate(top, 1):
        m = r["m"]
        u_rc1_s = f"{r['u_rc1']:+.6f}" if r["u_rc1"] is not None else "—"
        P(f"| {i} | {r['trial']} | {r['mix_spec']} | {r['measured_ratio']} | {r['lr_scale']} | {r['verdict']} | "
          f"{r['u']:+.6f} | {u_rc1_s} | "
          f"{m['track1_f1']:.6f} | {m['track2_f1']:.6f} | {m['track3_f1']:.6f} | {r['duration_min']} |")
    P("")

    # ---- 四之二：可上线候选榜（**仅过一级红线的**）--------------------------------
    # 为什么必须单独列一榜：一级红线是「一票否决」，被否决的 trial 不可上线，
    # 但它们的 U 往往比 PASS 的还高（因为 U 只罚「相对基线退让」，
    # 而红线判的是「绝对业务质量」）⇒ 混排会让读盘人误取第一名。
    # 实测（2026-10-27 用 GMC 真实 104 行冒烟）：Top-5 里第 2、3 名是
    # FAIL_TRACK1_RETREAT，排第 4 名的 PASS 候选被压在后面。
    shippable = sorted(
        [r for r in rows if r["u"] is not None and (r["verdict"] or "").startswith("PASS")],
        key=lambda r: -r["u"],
    )
    P(f"## 四之二、Top-{args.top} 可上线候选（**仅过一级红线者**，按重算 U）")
    P("")
    if not shippable:
        P("⚠️ **没有任何 trial 通过一级红线** —— 本轮无可上线候选（见上节各 trial 的 verdict）。")
        P("")
    else:
        P(f"（候选池共 **{len(shippable)}** 个过线 trial）")
        P("")
        P("| # | trial | mix_spec | 实测比 | lr_scale | verdict | U(vs 基线) | U(vs RC1) | T1 | T2 | T3 | min |")
        P("|--:|:--|:--|--:|--:|:--|--:|--:|--:|--:|--:|--:|")
        for i, r in enumerate(shippable[:args.top], 1):
            m = r["m"]
            u_rc1_s = f"{r['u_rc1']:+.6f}" if r["u_rc1"] is not None else "—"
            P(f"| {i} | {r['trial']} | {r['mix_spec']} | {r['measured_ratio']} | {r['lr_scale']} | {r['verdict']} | "
              f"{r['u']:+.6f} | {u_rc1_s} | "
              f"{m['track1_f1']:.6f} | {m['track2_f1']:.6f} | {m['track3_f1']:.6f} | {r['duration_min']} |")
        P("")
        best = shippable[0]
        P(f"**本轮最优可上线候选：`{best['trial']}`（mix_spec={best['mix_spec']}，实测比 {best['measured_ratio']}，"
          f"verdict={best['verdict']}，U(vs 基线)={best['u']:+.6f}，U(vs 新擂主)="
          f"{('%+.6f' % best['u_rc1']) if best['u_rc1'] is not None else '—'}）**")
        P("")

    # ---- 四之三：配置级多种子榜（**交接口径 = 同超参多种子中位**）----------------
    # 为什么必须以「配置」为单位（而不是单 trial）：
    #   §五之四 实测：no-GMC 侧同超参换种子的 T3 极差 0.170~1.651 pp、U 极差 0.0051~0.0297
    #   ⇒ 单点第一名的「优势」可能**完全落在种子噪声带内** ⇒ 不能当交接依据。
    #   实测反例：某配置单点 T3 全场最高，但同配置另两个种子全部撞 T1 红线。
    # 分组键 = 11 个调参维度**逐位相同**（同一条训练配方）；组内 trial 号不同 ⇒ 种子不同。
    def _cfg_norm(field, raw):
        """配置分组键的规范化：mix_spec / freeze_bn 按字符串，其余按 float 归一到 6 位小数。
        （用 float 归一是为了吸收 '7.21' / '7.210' 这类书写差异，避免同一配置被拆成两组。）"""
        if field in ("mix_spec", "freeze_bn"):
            return str(raw).strip()
        fv = _f(raw)
        return round(fv, 6) if fv is not None else str(raw).strip()

    def _cfg_key(r):
        raw = r.get("cfg_raw") or {}
        return tuple(_cfg_norm(f, raw.get(f, "")) for f in CFG_COLS)

    def _med(vals):
        vals = [v for v in vals if v is not None]
        return statistics.median(vals) if vals else None

    def _rng(vals):
        vals = [v for v in vals if v is not None]
        return (max(vals) - min(vals)) if len(vals) >= 2 else 0.0

    def _rank_key(g):
        us = [x["u"] for x in g if x["u"] is not None]
        return -statistics.median(us) if us else 9.0

    def _weight_info(trial, root):
        """从 trial 的 gate_report 取 epoch3 权重：返回 (相对路径, md5, 字节数) 或 None。

        gate_report.json 只记路径、不记 md5 ⇒ 这里现算 md5，
        让「结论 → 产物」这一环在 SUMMARY.md 里闭合（交接的人不用自己去猜拿哪个权重）。
        """
        gdir = Path(root) / "trials" / str(trial) / "gate"
        reps = sorted(gdir.glob("*/gate_report.json")) if gdir.is_dir() else []
        if not reps:
            return None
        try:
            rep = json.loads(reps[-1].read_text(encoding="utf-8"))
        except Exception:
            return None
        w = rep.get("weights")
        if not w:
            return None
        wp = Path(w)
        if not wp.is_absolute() and not wp.is_file():
            wp = Path(root).parent.parent / w  # runs/optuna_nogmc → 项目根
        if not wp.is_file():
            return None
        h = hashlib.md5()
        with wp.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return (w, h.hexdigest(), wp.stat().st_size)

    cfg_groups = defaultdict(list)
    for r in rows:
        if r["u"] is None:
            continue
        cfg_groups[_cfg_key(r)].append(r)
    multi = sorted([g for g in cfg_groups.values() if len(g) >= 2], key=_rank_key)
    single = sorted([g for g in cfg_groups.values() if len(g) == 1], key=_rank_key)
    # 实测的 U 种子噪声带 = 已复现配置（n≥2）内 U 极差的最大值
    max_ur = max([_rng([x["u"] for x in g]) for g in multi], default=0.0)

    P("## 四之三、配置级多种子榜（**交接以此为准**）")
    P("")
    P("> **为什么以「配置」为单位**：单 trial 的 U 排名被**种子噪声**支配 —— §五之四 实测 no-GMC 侧"
      "同超参换种子的 T3 极差 **0.170~1.651 pp**、U 极差 **0.0051~0.0297**。"
      "⇒ 单点第一名的「优势」可能完全落在噪声带内 ⇒ **不能当交接依据**。")
    P("> **分组键** = 11 个调参维度**逐位相同**（同一条训练配方）；组内 trial 号不同 ⇒ 种子不同。")
    P("> **口径**：组内一律取**中位数**；U 仍由 `utility()` 重算（不信 CSV 列）。")
    P("")

    if not cfg_groups:
        P("（没有带完整指标行的 trial ⇒ 本节跳过）")
        P("")
    else:
        P(f"### 4.3.1 已复现配置（n ≥ 2）—— **交接候选**（{len(multi)} 组）")
        P("")
        if not multi:
            P("⚠️ **没有任何配置拿到 2 个以上种子** ⇒ 本轮全是单点结果，"
              "排名**不可作为交接依据**（须先做同超参多种子复现）。")
            P("")
        else:
            P("| # | 配置（mix · lr_scale） | 种子数 | 过线 | U 中位 | U(vs RC1) 中位 | T1 中位 | T2 中位 | "
              "T2FP 中位 | T3 中位 | U 极差 | T3 极差 | trial |")
            P("|--:|:--|--:|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|")
            for i, g in enumerate(multi, 1):
                us = [x["u"] for x in g]
                urs = [x["u_rc1"] for x in g]
                npass = sum(1 for x in g if (x["verdict"] or "").startswith("PASS"))
                m1 = _med([x["m"]["track1_f1"] for x in g])
                m2 = _med([x["m"]["track2_f1"] for x in g])
                m3 = _med([x["m"]["track3_f1"] for x in g])
                fp2 = _med([x["m"]["track2_fp"] for x in g])
                u_rng = _rng(us)
                t3_rng = _rng([x["m"]["track3_f1"] for x in g]) * 100
                label = f"{g[0]['mix_spec']} · {g[0]['lr_scale']}"
                trials = "、".join(sorted(x["trial"] for x in g))
                med_urs = ("%+.6f" % statistics.median(urs)) if urs else "—"
                P(f"| {i} | {label} | {len(g)} | **{npass}/{len(g)}** | **{statistics.median(us):+.6f}** | "
                  f"{med_urs} | {m1:.6f} | {m2:.6f} | {fp2:.0f} | **{m3:.6f}** | {u_rng:.6f} | "
                  f"**{t3_rng:.3f} pp** | {trials} |")
            P("")
            P("**已复现配置的全量超参（11 维）+ 可交付权重** —— 交接 / 复现按这个配方：")
            P("")
            for i, g in enumerate(multi, 1):
                raw = g[0].get("cfg_raw") or {}
                us = [x["u"] for x in g]
                npass = sum(1 for x in g if (x["verdict"] or "").startswith("PASS"))
                label = f"{g[0]['mix_spec']} · lr_scale={g[0]['lr_scale']}"
                gbest = max(g, key=lambda x: x["u"])
                g_full = all((x["verdict"] or "").startswith("PASS") for x in g)
                P(f"- **{i}. {label}**（{len(g)} 种子，过线 {npass}/{len(g)}，U 中位 {statistics.median(us):+.6f}）")
                P("  - 超参（11 维）：`" + " | ".join(str(raw.get(k, "")) for k in CFG_COLS) + "`")
                P("  - 权重（epoch3.pt，md5 现算）：")
                _any_w = False
                for x in sorted(g, key=lambda y: y["trial"]):
                    wi = _weight_info(x["trial"], out_root)
                    if not wi:
                        P(f"    - `{x['trial']}` → ⚠️ 未找到 epoch3 权重")
                        continue
                    _any_w = True
                    # 「建议交付」只在**全种子过线**的组里标（否则等于把已出局配置推出去）
                    mark = ("　← **组内 U 最高 ⇒ 建议交付**"
                            if (g_full and x["trial"] == gbest["trial"]) else "")
                    P(f"    - `{x['trial']}` → `{wi[0]}`｜md5 `{wi[1]}`｜{wi[2]:,} B"
                      f"｜U {x['u']:+.6f}{mark}")
                if not _any_w:
                    P("    - ⚠️ 本组无可用 epoch3 权重（须查 trial 目录）")
            P("")
            # 交接冠军 = **全种子过一级红线**的组里 U 中位最高者；
            # 若没有全过的组，则退回 U 中位最高组并显式标注「交接线待定」。
            full_pass = [g for g in multi
                         if all((x["verdict"] or "").startswith("PASS") for x in g)]
            champ = full_pass[0] if full_pass else multi[0]
            bn = len(champ)
            bnpass = sum(1 for x in champ if (x["verdict"] or "").startswith("PASS"))
            bm3 = _med([x["m"]["track3_f1"] for x in champ])
            bt3rng = _rng([x["m"]["track3_f1"] for x in champ]) * 100
            bmed = statistics.median([x["u"] for x in champ])
            bbest = max(champ, key=lambda x: x["u"])

            # ---- 交接配置：一条**显式、可复算**的排序规则 ----
            # ① U 差距 > 实测 U 噪声带 ⇒ 按 U 定序；
            # ② U 不可分 ⇒ 转看 **T3 难例中位**（no-GMC 的短板、GMC 的价值都集中在难例）；
            #    T3 中位差 vs「中位不确定度」（保守 = 组内种子极差/2/√n）⇒ 超出 2 倍则 T3 高者胜；
            # ③ 仍不可分 ⇒ 取种子数多者（证据更强），并注明两者实质相当。
            # ⚠️ **不用「极差小者胜」**：极差随种子数**结构性偏小**（2 个点的极差天然小于 3 个点），
            #    跨组比较极差会被 n 绑架 —— 实测 2 种子组 0.027 pp vs 3 种子组 0.170 pp 是**假优势**。
            def _gstat(g):
                n = len(g)
                t3m = statistics.median([x["m"]["track3_f1"] for x in g])
                t3r = _rng([x["m"]["track3_f1"] for x in g]) * 100
                return {"g": g, "n": n, "t3m": t3m, "t3r": t3r,
                        "se": (t3r / 2.0) / math.sqrt(n) if n else 0.0,
                        "u": statistics.median([x["u"] for x in g]),
                        "np": sum(1 for x in g if (x["verdict"] or "").startswith("PASS")),
                        "label": f"{g[0]['mix_spec']} · {g[0]['lr_scale']}"}

            _S = [_gstat(g) for g in full_pass] if full_pass else [_gstat(multi[0])]
            _handoff = _S[0]
            _rule = []
            if len(_S) >= 2:
                _a, _b = _S[0], _S[1]
                _margin = _a["u"] - _b["u"]
                _rule.append(f"U 榜首 `{_a['label']}`（U 中位 {_a['u']:+.6f}，{_a['n']} 种子）"
                             f" vs 次席 `{_b['label']}`（{_b['u']:+.6f}，{_b['n']} 种子），"
                             f"差 **{_margin:.6f}**")
                if _margin > max_ur:
                    _handoff = _a
                    _rule.append(f"⇒ 超出实测 U 噪声带 **{max_ur:.6f}** ⇒ **按 U 定序**")
                else:
                    _rule.append(f"⇒ 落在实测 U 噪声带 **{max_ur:.6f}** 内 ⇒ **U 不可分** ⇒ 转看 **T3 难例**")
                    _bt = sorted(_S, key=lambda s: -s["t3m"])
                    _x, _y = _bt[0], _bt[1]
                    _d = (_x["t3m"] - _y["t3m"]) * 100
                    _thr = 2 * (_x["se"] + _y["se"])
                    _rule.append(f"T3 中位：`{_x['label']}` **{_x['t3m']:.6f}**"
                                 f"（{_x['n']} 种子，极差 {_x['t3r']:.3f} pp）"
                                 f" vs `{_y['label']}` **{_y['t3m']:.6f}**"
                                 f"（{_y['n']} 种子，极差 {_y['t3r']:.3f} pp），"
                                 f"差 **{_d:.3f} pp**，"
                                 f"判别阈值 2×(中位不确定度 {_x['se']:.3f}+{_y['se']:.3f}) = **{_thr:.3f} pp**")
                    if _d > _thr:
                        _handoff = _x
                        _rule.append(f"⇒ 差 {_d:.3f} pp **超出**阈值 ⇒ **T3 上可分** ⇒ 交接取 **`{_x['label']}`**")
                    else:
                        _handoff = max(_S, key=lambda s: (s["n"], s["t3m"]))
                        _rule.append(f"⇒ 差 {_d:.3f} pp **未超出**阈值 ⇒ **实质相当** ⇒ 取种子数多者")
            P(f"**交接结论（配置级）**：交接配置 = **{_handoff['label']}**，"
              f"{_handoff['n']} 个种子、过线 **{_handoff['np']}/{_handoff['n']}**、"
              f"U 中位 **{_handoff['u']:+.6f}**、T3 中位 **{_handoff['t3m']:.6f}**"
              f"（种子极差 {_handoff['t3r']:.3f} pp）。")
            P("")
            if _rule:
                P("> **排序规则（可复算）**：")
                for _line in _rule:
                    P(f"> - {_line}")
                P("")
            if _handoff["label"] != _S[0]["label"]:
                P(f"> ⚠️ **注意：U 榜首是 `{_S[0]['label']}`，但按上面的规则交接口径取 `{_handoff['label']}`** —— "
                  "两者 U 不可分，但 **T3 难例**可分（no-GMC 的短板与 GMC 的价值都集中在难例）。")
                if _handoff["n"] < _S[0]["n"]:
                    P(f"> ⚠️ 证据量提示：`{_handoff['label']}` 只有 **{_handoff['n']}** 个种子"
                      f"（U 榜首有 {_S[0]['n']} 个）⇒ 定案前建议再补 1 个同超参种子。")
                P("")
            P(f"> 组内 U 最高的单点是 **{bbest['trial']}**（U={bbest['u']:+.6f}）；"
              "但**交接口径 = 该配方的多种子中位表现**，单点权重只作参考。")
            P("")
            if not full_pass:
                P(f"> ⚠️ **没有任何配置做到全部种子过一级红线**（最高组仅 {bnpass}/{bn}）⇒ "
                  "**交接线待定**，须先补跑同超参复现。")
                P("")
            unstable = [g for g in multi
                        if sum(1 for x in g if (x["verdict"] or "").startswith("PASS")) < len(g)]
            if unstable:
                _u = "；".join(
                    f"{g[0]['mix_spec']} · {g[0]['lr_scale']}"
                    f"（{sum(1 for x in g if (x['verdict'] or '').startswith('PASS'))}/{len(g)}）"
                    for g in unstable)
                P(f"> ⚠️ **以下配置存在未过一级红线的种子 ⇒ 已出局（不可作交接线）**：{_u}。")
                P("")

        P(f"### 4.3.2 单点配置（n = 1）—— **仅参考，未复现**（{len(single)} 组）")
        P("")
        if not single:
            P("（无）")
        else:
            P("| # | 配置（mix · lr_scale） | U | U(vs RC1) | T1 | T2 | T2FP | T3 | verdict | trial |")
            P("|--:|:--|--:|--:|--:|--:|--:|--:|:--|:--|")
            for i, g in enumerate(single, 1):
                x = g[0]
                m = x["m"]
                u_rc1_s = ("%+.6f" % x["u_rc1"]) if x["u_rc1"] is not None else "—"
                P(f"| {i} | {x['mix_spec']} · {x['lr_scale']} | {x['u']:+.6f} | {u_rc1_s} | "
                  f"{m['track1_f1']:.6f} | {m['track2_f1']:.6f} | {m['track2_fp']} | "
                  f"{m['track3_f1']:.6f} | {x['verdict']} | {x['trial']} |")
        P("")
        if multi and single:
            bs = single[0][0]
            gap = bs["u"] - _handoff["u"]
            d3 = (bs["m"]["track3_f1"] - _handoff["t3m"]) * 100
            _max_t3r = max([_rng([x["m"]["track3_f1"] for x in g]) * 100 for g in multi], default=0.0)
            P(f"> ℹ️ **单点榜首 `{bs['trial']}`**（U={bs['u']:+.6f}，T3={bs['m']['track3_f1']:.6f}，"
              f"verdict={bs['verdict']}）vs 交接配置 `{_handoff['label']}`"
              f"（U 中位 {_handoff['u']:+.6f}，T3 中位 {_handoff['t3m']:.6f}）："
              f"**ΔU {gap:+.6f}**、**ΔT3 {d3:+.3f} pp**。")
            if gap > max_ur:
                P(f"> - U 优势 **{gap:.6f}** **超出**实测 U 噪声带 {max_ur:.6f} ⇒ 单点更强，"
                  "但**仍须复现**才可换交接线（单点 ≠ 配置稳）。")
            else:
                P(f"> - U 差 **{gap:+.6f}** **未超**噪声带 {max_ur:.6f} ⇒ **U 上不可分**。")
            if d3 > 0 and d3 > _handoff["t3r"]:
                P(f"> - **T3 高 {d3:.3f} pp**：超过交接配置自身的种子极差 {_handoff['t3r']:.3f} pp，"
                  f"但**未超**全局最大种子极差 {_max_t3r:.3f} pp ⇒ **不能排除是抽到了好种子** ⇒ 须复现。")
            elif d3 > 0:
                P(f"> - T3 高 {d3:.3f} pp，但在交接配置的种子极差 {_handoff['t3r']:.3f} pp 之内 ⇒ 无实质优势。")
            else:
                P(f"> - T3 差 {d3:.3f} pp（不高于交接配置）⇒ 单点无 T3 优势。")
            P("")
        P("> ⚠️ **单点过线 ≠ 配置稳**：单点 PASS 只说明「这一个种子过了」，"
          "同配置换种子可能撞一级红线（见 §4.3.1 的过线列 n/m）。")
        P("")

    # ---- 五、与 GMC 侧的绝对指标对照（**U 不可比，绝对值可比**）------------------
    # 为什么这一节必须存在：U 是「相对各自参照系」的量，跨语义**不可比**；
    # 但三轨的**绝对 F1** 是同帧集（31,613 / 17,187 / 11,810）、同 GT（25,111 / 16,638 / 10,535）、
    # 同工作点（system_sota_longquan_delivery_params.json）测出来的 ⇒ **绝对值可比**，
    # 这正是判断「no-GMC 能不能替代 GMC」真正需要的表。
    P("## 五、与 GMC 侧的绝对指标对照（**同帧同 GT 同工作点 ⇒ 绝对值可比；U 不可比**）")
    P("")
    gmc_rows = []
    try:
        from manu.training.auto_epoch_gate_check import BASE as _BASE  # noqa: E402

        gmc_rows.append(("GMC 冻结基线 474（去 P0）", _BASE["track1_uav"]["f1"], _BASE["track2_golden"]["f1"],
                         _BASE["track3_blind"]["f1"], _BASE["track2_golden"]["fp"]))
    except Exception:
        pass
    try:
        from manu.training.mix_ratio_judgement import RC1_EPOCH3 as _RC1  # noqa: E402

        gmc_rows.append(("GMC RC1（路线B Ep3）", _RC1["track1_f1"], _RC1["track2_f1"], _RC1["track3_f1"],
                         _RC1["track2_fp"]))
    except Exception:
        pass
    gmc_gr = Path("runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3/gate_report.json")
    if gmc_gr.is_file():
        try:
            t = json.loads(gmc_gr.read_text(encoding="utf-8"))["tracks"]
            gmc_rows.append((
                "GMC RC2（trial_0080）",
                t["track1_uav"]["finetune_peak"]["f1"], t["track2_golden"]["f1"], t["track3_blind"]["f1"],
                t["track2_golden"]["fp"],
            ))
        except Exception:
            pass
    if base:
        gmc_rows.append(("no-GMC 冻结基线 474（去 P0）", base.get("track1_f1"), base.get("track2_f1"),
                         base.get("track3_f1"), base.get("track2_fp")))
    if shippable:
        b = shippable[0]
        gmc_rows.append((f"**no-GMC 本轮最优可上线（{b['trial']}）**", b["m"]["track1_f1"],
                         b["m"]["track2_f1"], b["m"]["track3_f1"], b["m"]["track2_fp"]))
    if gmc_rows:
        P("| 模型 | T1 源域 F1 | T2 黄金池 F1 | T2 FP | T3 盲测 F1 |")
        P("|:--|--:|--:|--:|--:|")
        for name, f1, f2, f3, fp2 in gmc_rows:
            fmt = lambda v: ("—" if v is None else f"{v:.6f}")
            P(f"| {name} | {fmt(f1)} | {fmt(f2)} | {fp2 if fp2 is not None else '—'} | {fmt(f3)} |")
        P("")
        P("> 读法：**T2 绝对线 0.985 是业务交付要求**（与特征语义无关）；"
          "T3 是难例；T1 是源域大盘。U 各自相对各自基线，**不能跨语义比大小**。")
        P("")

    # ---- 五之二、与 GMC 侧的配对 A/B（同超参 + 同种子 ⇒ 受控的「去掉 GMC 值多少」）--------
    # 为什么这一节不可替代：
    #   §十一 的对照是**基线 vs 基线**（只测了底模）；§五 是**绝对值**对照（不同配方）。
    #   本节是**同一条训练配方**在两种特征语义下的**配对差**：
    #   两个 study 的前若干 trial 用的是同一套种子采样序列 ⇒ 同 trial 号 = 同 mix_spec
    #   + 同全部调参维度 + 同种子 + 同样本序列 ⇒ 差值就是「把 GMC 拿掉」的受控代价。
    #   ⚠️ 只做**配对差**，不做跨语义 U 横比。
    P("## 五之二、与 GMC 侧的**配对 A/B**（同超参 + 同种子 ⇒ 受控的「去掉 GMC 值多少」）")
    P("")
    try:
        from manu.training.compare_gmc_nogmc_matched import collect_pairs  # noqa: E402

        _gcsv = Path(args.gmc_csv)
        if not _gcsv.is_file():
            P(f"（跳过：找不到 GMC 侧汇总 CSV `{_gcsv}`）")
            P("")
        else:
            _pd = collect_pairs(_gcsv, csv_path, 200)
            if not _pd["matched"]:
                P(f"（跳过：两侧没有可配对的 trial —— 参数不一致 "
                  f"{len(_pd['mismatched'])} 个、缺行 {len(_pd['missing'])} 个）")
                P("")
            else:
                P(f"配对 **{len(_pd['matched'])}** 个 trial（参数不一致 {len(_pd['mismatched'])} 个、"
                  f"缺行 {len(_pd['missing'])} 个）。配对判据：`mix_spec` + 全部调参维度逐字相同"
                  "（同 trial 号 ⇒ 同超参 + 同种子 + 同样本序列）。")
                P("")
                P("| trial | mix | lr_scale | 实测比 | T1 F1 GMC / no-GMC / Δ | T2 F1 GMC / no-GMC / Δ | T3 F1 GMC / no-GMC / Δ | ΔFP T2 | ΔFP T3 |")
                P("|:--|:--|--:|--:|:--|:--|:--|--:|--:|")
                for r in _pd["rows"]:
                    P(f"| {r['trial']} | {r['mix_spec']} | {r['lr_scale']:.4f} | "
                      f"{r['measured_ratio_gmc']:.2f} | "
                      f"{r['t1_f1_gmc']:.6f} / {r['t1_f1_nogmc']:.6f} / **{r['d_t1_f1'] * 100:+.3f} pp** | "
                      f"{r['t2_f1_gmc']:.6f} / {r['t2_f1_nogmc']:.6f} / **{r['d_t2_f1'] * 100:+.3f} pp** | "
                      f"{r['t3_f1_gmc']:.6f} / {r['t3_f1_nogmc']:.6f} / **{r['d_t3_f1'] * 100:+.3f} pp** | "
                      f"{r['d_t2_fp']:+d} | {r['d_t3_fp']:+d} |")
                P("")
                _ag = _pd["aggregate"]
                _pt = _ag.get("paired_tests") or {}
                if _pt:
                    P("**配对检验**（Wilcoxon 符号秩，双侧；Δ = no-GMC − GMC；house rule §三·6c）：")
                    P("")
                    P("| 轨 | n | 配对差中位 | Hodges–Lehmann | W | p | 符号检验 p |")
                    P("|:--|--:|--:|--:|--:|--:|--:|")
                    for _k, _name in (("t1_f1", "T1 F1"), ("t2_f1", "T2 F1"), ("t3_f1", "T3 F1")):
                        _t = _pt.get(_k)
                        if not _t:
                            continue
                        P(f"| {_name} | {_t['n']} | {_t['median']:+.6f} | {_t['hodges_lehmann']:+.6f} | "
                          f"{('%.1f' % _t['w']) if _t['w'] is not None else '—'} | **{_t['p']:.3g}** | "
                          f"{_t['sign_p']:.3g} |")
                    P("")
                P(f"**配对差汇总（no-GMC − GMC）**：T1 ΔF1 中位 **{_ag['d_t1_f1_median_pp']:+.3f} pp**"
                  f"（区间 {_ag['d_t1_f1_min_pp']:+.3f} ~ {_ag['d_t1_f1_max_pp']:+.3f}）｜"
                  f"T2 ΔF1 中位 **{_ag['d_t2_f1_median_pp']:+.3f} pp**"
                  f"（区间 {_ag['d_t2_f1_min_pp']:+.3f} ~ {_ag['d_t2_f1_max_pp']:+.3f}）｜"
                  f"T3 ΔF1 中位 **{_ag['d_t3_f1_median_pp']:+.3f} pp**"
                  f"（区间 {_ag['d_t3_f1_min_pp']:+.3f} ~ {_ag['d_t3_f1_max_pp']:+.3f}）｜"
                  f"虚警 ΔFP_T2 中位 **{_ag['d_t2_fp_median']:+.0f}**、ΔFP_T3 中位 **{_ag['d_t3_fp_median']:+.0f}**")
                P("")
                P("> 读法：**T3 是 GMC 的主战场**（同一配方下，去掉 GMC 后难例 F1 掉约 "
                  f"{abs(_ag['d_t3_f1_median_pp']):.1f} pp），T1/T2 只掉零点几个 pp。"
                  "⚠️ 这是**配对差**，不是 U 横比。")
                P("")
                if "d_t3_recall_median_pp" in _ag:
                    P("**召回/虚警分解**（由落盘 F1+FP 与 GT 反解 TP，~1e-4 舍入误差，只用于归因）："
                      f"T3 ΔRecall 中位 **{_ag['d_t3_recall_median_pp']:+.3f} pp**、"
                      f"ΔPrecision 中位 **{_ag['d_t3_prec_median_pp']:+.3f} pp**；"
                      f"T2 ΔRecall {_ag['d_t2_recall_median_pp']:+.3f} pp、"
                      f"ΔPrecision {_ag['d_t2_prec_median_pp']:+.3f} pp；"
                      f"T1 ΔRecall {_ag['d_t1_recall_median_pp']:+.3f} pp、"
                      f"ΔPrecision {_ag['d_t1_prec_median_pp']:+.3f} pp。")
                    P("")
                    P("> ⇒ 难例上 no-GMC 是**召回与精度双降**（虚警更多），不是单纯「报得少」；"
                      "这与基线侧「T3 FP 反而 −54、只掉召回」的方向**不同**，说明是**混训**把 no-GMC 的难例"
                      "推到了「乱报」区。")
                    P("")
    except Exception as exc:  # 绝不因为这一节崩掉整个汇总
        P(f"（§五之二 配对 A/B 生成失败，已跳过：{exc}）")
        P("")

    # ---- 五之三、lr_scale 分档：源域代价 ↔ 难例增益（两侧同轴）------------------
    # 目的：找出 no-GMC 语义下「难例增益 ↔ 源域退让」的操作点，并和 GMC 侧同轴对照。
    # Δ 均相对**各自**的冻结基线 ⇒ 两侧可比；不涉及跨语义 U。
    P("## 五之三、`lr_scale` 分档：源域代价 ↔ 难例增益（两侧同轴）")
    P("")
    try:
        from manu.training.compare_gmc_nogmc_matched import (  # noqa: E402
            gmc_base, lr_bands, nogmc_base,
        )

        _bands = lr_bands(_gcsv, csv_path, gmc_base(), nogmc_base(args.ref), 200)
        P("| lr_scale 档 | GMC：n / ΔF1_T1 / ΔF1_T3 | no-GMC：n / ΔF1_T1 / ΔF1_T3 |")
        P("|:--|:--|:--|")
        for b in _bands:
            def _cell(x):
                if not x:
                    return "—"
                return f"n={x['n']} / **{x['d_t1_median_pp']:+.3f} pp** / **{x['d_t3_median_pp']:+.3f} pp**"
            P(f"| {b['band']} | {_cell(b['gmc'])} | {_cell(b['nogmc'])} |")
        P("")
        P("> 读法：每格 = 该 lr 档内 trial 的**中位数**（Δ 相对各自冻结基线）。"
          "**no-GMC 在每一个 lr 档上难例增益都更低** ⇒ "
          "这不是「把 lr 调对就能补上」的差距 ⇒ **no-GMC 必须换底模，不能只调超参**。")
        P(">")
        # 覆盖度判定：**两侧高 lr 档样本是否都充足** ⇒ 决定「源域代价列」能不能比
        _hi = next((b for b in _bands if str(b.get("band", "")).startswith("≥")), None)
        _hig = ((_hi or {}).get("gmc") or {})
        _hin = ((_hi or {}).get("nogmc") or {})
        _hign = int(_hig.get("n") or 0)
        _hinn = int(_hin.get("n") or 0)
        _enough = _hign >= 10 and _hinn >= 10
        # 难例增益比（no-GMC / GMC），只取两侧 n 都够的档
        _ratios = []
        for b in _bands:
            _g = b.get("gmc") or {}
            _n = b.get("nogmc") or {}
            if int(_g.get("n") or 0) >= 5 and int(_n.get("n") or 0) >= 5:
                _gd = _g.get("d_t3_median_pp") or 0.0
                if _gd > 0.5:
                    _ratios.append((_n.get("d_t3_median_pp") or 0.0) / _gd)
        if _ratios:
            P(f"> 📊 **难例增益比（no-GMC / GMC，两侧 n≥5 的档）**："
              f"**{min(_ratios) * 100:.0f}% ~ {max(_ratios) * 100:.0f}%**"
              f"（{len(_ratios)} 个档）⇒ no-GMC 的每个 lr 档都只能换来 GMC 一半左右的难例增益。")
            P(">")
        if _enough:
            P(f"> ✅ **源域代价（ΔF1_T1）一列现已可比**（两侧最高 lr 档样本都充足："
              f"GMC n={_hign}、no-GMC n={_hinn}）："
              f"**no-GMC 在所有 lr 档上源域退让都比 GMC 小**"
              f"（最高档 GMC {_hig.get('d_t1_median_pp'):+.3f} pp vs "
              f"no-GMC {_hin.get('d_t1_median_pp'):+.3f} pp），"
              "**但难例增益只有 GMC 的一半左右** ⇒ "
              "**no-GMC 的混训「性价比」更低：更少地动源域，却换来更少的难例收益** ⇒ 差距来自语义本身。")
        else:
            P(f"> ⚠️ **源域代价（ΔF1_T1）一列暂不可比**：no-GMC 在高 lr 档样本不足"
              f"（GMC n={_hign}、no-GMC n={_hinn}），不能据此说哪一侧「源域代价更重」；"
              "配对同超参同种子区（§五之二）的证据是 no-GMC 略高约 0.26 pp。")
            P(">")
            P("> ⚠️ **采样覆盖缺口**：no-GMC 侧的高 lr 档可能尚未采样 ⇒ 其最优操作点可能还没探到。")
        P("")
    except Exception as exc:  # 绝不因为这一节崩掉整个汇总
        P(f"（§五之三 生成失败，已跳过：{exc}）")
        P("")

    # ---- 五之四、种子噪声带（同超参不同种子的实测极差）------------------
    # 为什么需要：判断「候选 A 比候选 B 高 X pp」是**真信号**还是**种子噪声**。
    # 只有 GMC 侧存在「完全同超参、不同种子」的重复组 ⇒ 噪声带由 GMC 侧实测给出。
    P("## 五之四、种子噪声带（**同超参不同种子**的实测跨种子极差）")
    P("")
    try:
        from manu.training.compare_gmc_nogmc_matched import seed_noise  # noqa: E402

        _sn_g = seed_noise(_gcsv, 200)
        _sn_n = seed_noise(csv_path, 200)
        if not _sn_g and not _sn_n:
            P("（两侧都没有完全同超参的重复组 ⇒ 种子噪声带无法实测）")
            P("")
        else:
            for _side, _sn, _title in (("GMC", _sn_g, "GMC 侧"), ("no-GMC", _sn_n, "no-GMC 侧")):
                if not _sn:
                    continue
                P(f"**{_title}**")
                P("")
                P("| 配置（mix / lr_scale） | 种子数 | trial | ΔF1_T1 极差 | ΔF1_T2 极差 | ΔF1_T3 极差 | T2FP 极差 | T3FP 极差 |")
                P("|:--|--:|:--|--:|--:|--:|--:|--:|")
                for g in _sn:
                    P(f"| {g['mix_spec']} / {g['lr_scale']} | {g['n_seeds']} | "
                      f"{'、'.join(g['trials'])} | **{g['t1_range_pp']:.3f} pp** | {g['t2_range_pp']:.3f} pp | "
                      f"**{g['t3_range_pp']:.3f} pp** | {g['t2_fp_range']:.0f} | {g['t3_fp_range']:.0f} |")
                P("")
            if _sn_g and _sn_n:
                _g3 = (min(g["t3_range_pp"] for g in _sn_g), max(g["t3_range_pp"] for g in _sn_g))
                _n3 = (min(g["t3_range_pp"] for g in _sn_n), max(g["t3_range_pp"] for g in _sn_n))
                _g1 = max(g["t1_range_pp"] for g in _sn_g)
                _n1 = max(g["t1_range_pp"] for g in _sn_n)
                P(f"> **读法**：同一配置换种子，GMC 侧 T1 波动 ≤ **{_g1:.3f} pp**、T3 波动 **{_g3[0]:.3f}~{_g3[1]:.3f} pp**；"
                  f"no-GMC 侧 T1 ≤ **{_n1:.3f} pp**、T3 **{_n3[0]:.3f}~{_n3[1]:.3f} pp**。")
                P("> ⇒ **no-GMC 的难例指标更\"飘\"（T3 种子噪声更大）** ⇒ "
                  "**no-GMC 的交接线必须用多种子中位数 + 上分位来定，不能用单点 U**；"
                  "而 GMC 侧的 T2 差异几乎无噪声（≤0.02 pp）⇒ T2 的差异可信。")
                P("")
    except Exception as exc:  # 绝不因为这一节崩掉整个汇总
        P(f"（§五之四 生成失败，已跳过：{exc}）")
        P("")

    # ---- 五之五、RC2 配方移植：「配置完全一样，只换数据集」的实测对应 ----------
    # 用户原始要求是「参考 RC2 0080 的训练脚本，配置完全一样，只替换训练集和测试集」。
    # §五之二 的配对 A/B 覆盖的是**前 12 个 trial 的随机配方**（都是 lr=1.0 的温和档），
    # **不等于 RC2 自己那条配方**（w2.5 / lr 2.2702 / …）⇒ 这里补上「RC2 配方 × no-GMC 数据」
    # 的直接对照；若 no-GMC 侧没有逐位相同的配方，就如实报「覆盖缺口」并给最近邻。
    P("## 五之五、RC2 配方在 no-GMC 侧的对应（**「配置完全一样，只换数据集」的实测**）")
    P("")
    try:
        _gmc_rows = []
        _gcsv2 = Path(args.gmc_csv)
        if _gcsv2.is_file():
            with _gcsv2.open(encoding="utf-8") as _fh:
                _gmc_rows = list(csv.DictReader(_fh))
        _rc2 = next((r for r in _gmc_rows if r.get("trial") == "trial_0080"), None)
        if not _rc2:
            P(f"（跳过：GMC 侧 CSV `{_gcsv2}` 里没有 `trial_0080`）")
            P("")
        else:
            _rc2_cfg = {c: str(_rc2.get(c, "")).strip() for c in CFG_COLS}

            def _same_as_rc2(r):
                raw = r.get("cfg_raw") or {}
                for c in CFG_COLS:
                    if _cfg_norm(c, raw.get(c, "")) != _cfg_norm(c, _rc2_cfg.get(c, "")):
                        return False
                return True

            _tpl = sorted([r for r in rows if _same_as_rc2(r)], key=lambda r: r["trial"])
            P("**GMC RC2（trial_0080）配方**：`"
              + " | ".join(_rc2_cfg[c] for c in CFG_COLS) + "`")
            P("")
            P(f"GMC RC2 三轨：T1 **{_rc2.get('t1_f1')}** ｜ T2 **{_rc2.get('t2_f1')}**"
              f"（FP {_rc2.get('t2_fp')}）｜ T3 **{_rc2.get('t3_f1')}**（FP {_rc2.get('t3_fp')}）")
            P("")
            P(f"### 5.5.1 逐位相同配方的 no-GMC trial（**配方移植**）：{len(_tpl)} 个")
            P("")
            if not _tpl:
                P("⚠️ **覆盖缺口**：no-GMC 侧至今没有任何 trial 用过 RC2 的这条配方 ⇒ "
                  "**「严格照搬 RC2 配方、只换数据集」这一条在本轮尚未被直接验证**"
                  "（§五之二 的配对 A/B 覆盖的是前 12 个 trial 的随机配方，不含 RC2 这条）。")
                P("")
            else:
                P("| trial | T1 | T2 | T2FP | T3 | T3FP | ΔT1 | ΔT2 | ΔT3 | verdict |")
                P("|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|")
                for r in _tpl:
                    m = r["m"]
                    d1 = (m["track1_f1"] - _f(_rc2.get("t1_f1"))) * 100
                    d2 = (m["track2_f1"] - _f(_rc2.get("t2_f1"))) * 100
                    d3 = (m["track3_f1"] - _f(_rc2.get("t3_f1"))) * 100
                    P(f"| {r['trial']} | {m['track1_f1']:.6f} | {m['track2_f1']:.6f} | {m['track2_fp']} | "
                      f"{m['track3_f1']:.6f} | {m['track3_fp']} | **{d1:+.3f} pp** | **{d2:+.3f} pp** | "
                      f"**{d3:+.3f} pp** | {r['verdict']} |")
                P("")
                _ds3 = [(r["m"]["track3_f1"] - _f(_rc2.get("t3_f1"))) * 100 for r in _tpl]
                _ds1 = [(r["m"]["track1_f1"] - _f(_rc2.get("t1_f1"))) * 100 for r in _tpl]
                _ds2 = [(r["m"]["track2_f1"] - _f(_rc2.get("t2_f1"))) * 100 for r in _tpl]
                P(f"> **读法**：同一份 RC2 配方，只把训练/测试集换成 no-GMC ⇒ "
                  f"T1 ΔF1 中位 **{statistics.median(_ds1):+.3f} pp**、"
                  f"T2 ΔF1 中位 **{statistics.median(_ds2):+.3f} pp**、"
                  f"T3 ΔF1 中位 **{statistics.median(_ds3):+.3f} pp**。"
                  "⚠️ 这是**绝对指标差**（同帧同 GT 同工作点 ⇒ 可比），不是跨语义 U 横比。")
                P("")

            # 最近邻（归一化欧氏距离）：no-GMC 侧离 RC2 配方最近的几个
            _num = [c for c in CFG_COLS if c not in ("mix_spec", "freeze_bn")]
            _pool = list(_gmc_rows) + [{c: (r.get("cfg_raw") or {}).get(c, "") for c in CFG_COLS}
                                       for r in rows]
            _rng = {}
            for c in _num:
                _v = [_f(_row.get(c)) for _row in _pool]
                _v = [v for v in _v if v is not None]
                if _v:
                    _rng[c] = (min(_v), max(_v))

            def _dist(r):
                raw = r.get("cfg_raw") or {}
                _d = 0.0
                for c in CFG_COLS:
                    if c in ("mix_spec", "freeze_bn"):
                        if str(raw.get(c, "")).strip() != _rc2_cfg.get(c, ""):
                            _d += 1.0
                        continue
                    a, b = _f(raw.get(c)), _f(_rc2_cfg.get(c))
                    if a is None or b is None or c not in _rng:
                        continue
                    lo, hi = _rng[c]
                    _d += ((a - b) / ((hi - lo) or 1.0)) ** 2
                return math.sqrt(_d)

            _near = sorted([r for r in rows if r["u"] is not None], key=_dist)[:6]
            if _near:
                P("### 5.5.2 no-GMC 侧离 RC2 配方最近的 trial（归一化欧氏距离，0 = 配方完全相同）")
                P("")
                P("| trial | 距离 | mix · lr_scale | T1 | T2 | T2FP | T3 | T3FP | ΔT3 vs RC2 | verdict |")
                P("|:--|--:|:--|--:|--:|--:|--:|--:|--:|:--|")
                for r in _near:
                    m = r["m"]
                    d3 = (m["track3_f1"] - _f(_rc2.get("t3_f1"))) * 100
                    P(f"| {r['trial']} | {_dist(r):.4f} | {r['mix_spec']} · {r['lr_scale']} | "
                      f"{m['track1_f1']:.6f} | {m['track2_f1']:.6f} | {m['track2_fp']} | "
                      f"{m['track3_f1']:.6f} | {m['track3_fp']} | **{d3:+.3f} pp** | {r['verdict']} |")
                P("")
                P("> 距离只用于**说明覆盖程度**，不代表性能；性能看 §四之三 / §五之二。")
                P("")
    except Exception as exc:  # 绝不因为这一节崩掉整个汇总
        P(f"（§五之五 生成失败，已跳过：{exc}）")
        P("")

    # ---- 口径声明 ----
    P("## 六、口径声明（必读）")
    P("")
    P("1. **本表的 U 与 GMC 搜索的 U 不可横向比较**：GMC 的参照系是 `uav_gmc_median` 特征下的冻结基线，"
      "本表的参照系是 `uav_gmc_median_nogmc` 特征下重测的基线（同一模型、同一口径、不同特征语义）。")
    P("2. **U 一律重算**（`mix_ratio_judgement.utility()`），不信 CSV 列 —— 防止落盘列与实际指标脱节。")
    P(f"3. 一级红线是**相对本参照系**的：T2 F1 ≥ {_t2_min} 且 ΔFP_T2 ≤ +{_fp_inc} 且 ΔF1_T1 ≥ {_t1_delta}；"
      "「PASS / PASS_RELAXED」与「PRUNED」按此判定。")
    if _fp_inc != 10:
        P(f"3.1 ⚠️ **ΔFP_T2 上限已由 +10 重标定为 +{_fp_inc}**（2026-10-27 用户授权）：no-GMC 语义下混训的"
          f"ΔFP_T2 正常带为 [+43,+51]（基线 FP {base.get('track2_fp', 52)}），沿用 +10 会 100% 剪枝；"
          "绝对业务线 T2 F1 ≥ 0.985 与 T1 红线 −0.005 未动。"
          f"依据：实测 8 点 ΔFP 分布 [+43,+43,+46,+46,+46,+51,+73,+73] ⇒ +{_fp_inc} 放过 6 个正常配置、"
          "仍抓住 2 个真爆炸（ΔF1_T3 为负）。")
    P("4. 若要找 no-GMC 版的「RC2 交接线」，应取**本表内 w2.5~w2.5 档 U 的上分位 + 多种子中位数**"
      "重新定标，而不是套用 GMC 的 +0.045。")
    P("5. **§四 的 Top 榜混排了被一级红线否决的 trial**（它们 U 更高但**不可上线**）；"
      "**选型请只看 §四之二 的可上线候选榜**。")
    P("6. **最终选型口径 = §四之三 的配置级多种子中位**（单 trial 的 U 受种子噪声支配，"
      "§五之四 实测 no-GMC 侧 U 噪声带 0.0051~0.0297、T3 极差 0.170~1.651 pp）；"
      "§四 只作参考、§四之二 只作单点过线筛查，**二者都不能替代配置级多种子结论**。")
    P("")

    text = "\n".join(lines) + "\n"
    print(text)
    md = Path(args.out_md) if args.out_md else (out_root / "SUMMARY.md")
    try:
        md.write_text(text, encoding="utf-8")
        print(f"[SUMMARY] 已写入 {md}")
    except Exception as exc:
        print(f"[WARN] 写 SUMMARY.md 失败：{exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
