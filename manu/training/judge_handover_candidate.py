#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**交接口径判定器**：从 no-GMC 搜索汇总 CSV 判定"哪个配置可以交接"。

为什么需要它
------------
交接口径（memory 铁律）：

1. **严格通道为上位门槛；放宽通道仅作诊断标记**
   ⇒ 只有 ``gate_verdict == "PASS"`` 的配置有资格被交接；
   ``PASS_RELAXED`` / ``FAIL_TRACK1_RETREAT`` 一律**排除在交接名单外**
   （可以列在附录里当诊断信息，但绝不能进交接）。
2. **交接口径必须是"同超参多种子中位"，不是单 trial 最高 U**
   ⇒ 先按 15 维配方逐位相同分组，组内取中位。
3. **噪声带用「极差/2/√n」修正**（不是裸极差——裸极差随种子数结构性偏小）。
4. **U 不可分 ⇒ 转看 T3 中位；T3 仍不可分 ⇒ 取种子数多者**。
5. 交付前必须说明**阻塞项**：例如最佳组只有 1 个种子 ⇒ 不满足"多种子"，
   只能作为"候选"而不是"交接线"。

用法（服务器 ``conda activate uav``，根目录 ``/tmp/pycharm_project_10ae9e2e``）：

```bash
python manu/training/judge_handover_candidate.py \
  --csv   /mnt/data/siping/runs/nogmc_focus/optuna_summary.csv \
  --ref   /mnt/data/siping/runs/nogmc_ref_weekend/reference_frame.json \
  --out-md   /mnt/data/siping/runs/nogmc_focus/HANDOVER.md \
  --out-json /mnt/data/siping/runs/nogmc_focus/HANDOVER.json \
  --min-seeds 2
```

退出码：0 = 有可交接候选；3 = 有候选但**种子数不足**（需补复现）；4 = 没有任何严格 PASS。

变体（零训练成本产物）—— ``--variants-csv``
-------------------------------------------
``cs_*``（跨种子 soup）与 ``swa_*``（同 run 相邻 epoch 平均）是**权重平均产物**，
**不是**该配置的又一个种子 ⇒ 它们的指标**绝不能**混进「同配方种子中位」。

用 ``--variants-csv TAG=路径``（可多个）把它们送进候选池：

* 变体行**单独建组**（键 = 变体行自己的 ``variant_key`` 列），不与普通 trial 行混组
  ⇒ **同配方中位数逐位不变**；
* 硬门禁用**变体自己的三轨指标**判定（它就是单个产物，没有中位可言）；
* 「多种子证据」取变体行自己的 ``seed_evidence_n`` 列（**源家族**的种子数），
  而不是变体自身的产物数 —— 阻塞项「种子数不足 min-seeds」据此判定；
* 变体**参与**与配置组的判定链比较，但报告里**单独成表**并标注
  「变体候选（权重平均，非种子）」，不与配置榜混排。

**不传 ``--variants-csv`` 时，输出与本参数引入前逐字节一致**（默认行为不变）。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path

CFG_COLS = ("mix_spec", "lr_scale", "warmup_epochs", "weight_decay", "focal_beta",
            "offset_weight", "mosaic", "scale", "translate", "clip_max_norm", "freeze_bn",
            "init_from", "epochs", "backbone_lr_scale", "freeze_backbone_epochs")
FLOAT_COLS = {"lr_scale", "warmup_epochs", "weight_decay", "focal_beta", "offset_weight",
              "mosaic", "scale", "translate", "clip_max_norm", "backbone_lr_scale"}
INT_COLS = {"epochs", "freeze_backbone_epochs"}

STRICT = "PASS"
DIAG = ("PASS_RELAXED", "FAIL_TRACK1_RETREAT")


# ----------------------------------------------------------------- 小工具
def _f(v, field=None):
    """把 CSV 字符串转成数值；空串/坏值返回 None。"""
    s = ("" if v is None else str(v)).strip()
    if s == "":
        return None
    if field in INT_COLS:
        try:
            return int(float(s))
        except Exception:
            return None
    try:
        return float(s)
    except Exception:
        return None


def _cfg_key(row: dict) -> tuple:
    out = []
    for c in CFG_COLS:
        s = str(row.get(c, "") or "").strip()
        if c in ("mix_spec", "init_from"):
            out.append(s)
        elif c in INT_COLS:
            try:
                out.append(int(float(s)))
            except Exception:
                out.append(s)
        elif c == "freeze_bn":
            out.append(s.lower() in ("true", "1", "yes"))
        else:
            try:
                out.append(round(float(s), 6))
            except Exception:
                out.append(s)
    return tuple(out)


def _median(vals) -> float | None:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return statistics.median(vals)


def _band(vals) -> tuple[float | None, int]:
    """噪声带 = **极差/2/√n**（修正口径，不用裸极差）。返回 (band, n)。

    n<2 ⇒ 无法估计 ⇒ band=0 且由调用方标记"噪声未估计"。
    """
    vals = [v for v in vals if v is not None]
    n = len(vals)
    if n < 2:
        return 0.0, n
    rng = max(vals) - min(vals)
    return rng / 2.0 / (n ** 0.5), n


def _fmt(v, spec: str = "{:.6f}") -> str:
    if v is None:
        return "—"
    try:
        return spec.format(v)
    except Exception:
        return str(v)


def _disjoint(a, b) -> bool:
    """两个 (med, band) 区间是否**不重叠** ⇒ 可分。"""
    if a[0] is None or b[0] is None:
        return False
    a_lo, a_hi = a[0] - a[1], a[0] + a[1]
    b_lo, b_hi = b[0] - b[1], b[0] + b[1]
    return a_hi < b_lo or b_hi < a_lo


# ----------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser(description="no-GMC 交接口径判定")
    ap.add_argument("--csv", required=True, nargs="+",
                    help="summarize_nogmc_search.py 产出的 optuna_summary.csv；"
                         "可给多个，格式 'TAG=路径'（TAG 建议用 study 名，如 weekend / focus）。"
                         "⚠️ **多 study 合并时必须带 TAG**：两个 study 都有 trial_0009，"
                         "但它们是**不同的 trial** ⇒ trial 名会加 'TAG:' 前缀以免混淆。")
    ap.add_argument("--variants-csv", default=[], nargs="+",
                    help="**零训练成本变体**（cs_* 跨种子 soup / swa_* 同 run 相邻 epoch 平均）"
                         "的门禁结果 CSV；格式同 --csv（'TAG=路径'）。"
                         "变体**单独建组**（键 = 变体自己的 variant_key 列），"
                         "**不进**同配方种子中位 ⇒ 绝不污染配置中位数；"
                         "硬门禁用**它自己的三轨指标**判；"
                         "多种子证据取变体行的 seed_evidence_n（**源家族**种子数）。"
                         "**不给 ⇒ 行为与本参数引入前逐字节一致**。")
    ap.add_argument("--ref", required=True, help="reference_frame.json（参照系 + 门禁阈值）")
    ap.add_argument("--out-md", default="")
    ap.add_argument("--out-json", default="")
    ap.add_argument("--min-seeds", type=int, default=2, help="交接所需的最小种子数")
    ap.add_argument("--top", type=int, default=12, help="报告里列出的配置数")
    ap.add_argument("--trials-root", default="", help="用于解析候选权重路径（可选）")
    args = ap.parse_args()

    ref = json.loads(Path(args.ref).read_text(encoding="utf-8"))
    fb = ref["frozen_baseline"]
    gate = ref.get("gate", {})
    t1_redline = fb["track1_f1"] + float(gate.get("track1_min_delta", -0.005))
    t2_fp_max = fb["track2_fp"] + int(gate.get("track2_fp_max_increment", 60))
    t2_f1_min = float(gate.get("track2_f1_min", 0.985))

    # ---- 读入（可能多份 CSV）------------------------------------------
    # ⚠️ **为什么要支持多份**：交接口径要求「同超参多种子中位」。
    #    周末那 114 条里有 33 条严格 PASS（含 T3 最高的 trial_0009），
    #    本轮 focus 的臂里又会给同配方补种子 ⇒ **同配方的种子横跨两个 study**。
    #    只读 focus 的 CSV ⇒ 候选池里少了周末的强候选，而且同配方的种子被拆成两组
    #    ⇒ n 永远算不到 2 ⇒ 交接口径永远不成立。
    rows: list[dict] = []
    for spec in args.csv:
        if "=" in spec and not os.path.exists(spec):
            tag, path = spec.split("=", 1)
        else:
            path = spec
            tag = Path(path).parent.name
        if not os.path.exists(path):
            print(f"[FATAL] 找不到 CSV：{path}")
            return 2
        for r in csv.DictReader(Path(path).open("r", encoding="utf-8")):
            # ⚠️ **trial 名加来源前缀**：两个 study 都有 trial_0009，但不是同一条 trial
            #    （周末 trial_0009 = w3.0/lr1.0/15ep；focus trial_0009 = w6.0/lr2.0/10ep）
            #    ⇒ 不加前缀会「同名不同物」，固化工��会取错权重。
            r["trial"] = f"{tag}:{r.get('trial')}"
            r["_src"] = tag
            rows.append(r)
    print(f"[INPUT] 读入 {len(rows)} 行，来自 {len(args.csv)} 份 CSV：{args.csv}")

    # ---- 读入**变体**（可选；零训练成本产物）---------------------------
    #
    # ⚠️ **为什么变体必须走独立通道**：
    #    `cs_*`（跨种子 soup）/ `swa_*`（同 run 相邻 epoch 平均）是**权重平均产物**，
    #    **不是**该配置的第 N 个种子 ⇒ 把它的指标混进「同配方种子中位」
    #    会**污染中位**（口径铁律：决策单元 = 配置的中位）。
    #    ⇒ 变体**另起一组**：硬门禁用**它自己的三轨指标**判，
    #      多种子证据取**源家族**的种子数（`seed_evidence_n` 列）。
    variant_rows: list[dict] = []
    variant_skipped = 0
    for spec in args.variants_csv:
        if "=" in spec and not os.path.exists(spec):
            tag, path = spec.split("=", 1)
        else:
            path = spec
            tag = Path(path).parent.name
        if not os.path.exists(path):
            print(f"[FATAL] 找不到变体 CSV：{path}")
            return 2
        for r in csv.DictReader(Path(path).open("r", encoding="utf-8")):
            r["_src"] = tag
            variant_rows.append(r)
    if variant_rows:
        print(f"[INPUT] 读入 {len(variant_rows)} 行**变体**，来自 {len(args.variants_csv)} 份变体 CSV："
              f"{args.variants_csv}")

    # ⚠️ **安全网**：防止有人把变体 CSV 误当普通 CSV 喂进 --csv
    #    ⇒ 变体行会混进同配方种子中位 ⇒ **污染中位数**。
    #    （正常 optuna_summary.csv 没有 variant_key 列 ⇒ 本分支不触发 ⇒ 行为不变。）
    _n_before = len(rows)
    rows = [r for r in rows if not (r.get("variant_key") or "").strip()]
    variant_skipped += _n_before - len(rows)
    if variant_skipped:
        print(f"[WARN] {variant_skipped} 行带 variant_key（**变体行**）已从普通 trial 行中剔除 "
              f"⇒ 不会污染同配方种子中位；变体请走 --variants-csv")

    # ---- 分组：**按配方把全部种子归组**（不看单条 trial 的 verdict）----
    #
    # ⚠️ **为什么不能只收 verdict==PASS 的 trial**（2026-11-02 15:40 实测发现）：
    #    交接口径的**决策单元是「配置」**，不是单条 trial —— 口径明写
    #    「同超参多种子中位」。单条 trial 的 `PASS_RELAXED` 只说明"**这条种子**的 T1
    #    没过红线"，**不代表这个配置的中位没过**。
    #    若只用 PASS 的 trial 建组 ⇒ 一个配置 3 条种子只有 1 条 PASS 时 n=1
    #    ⇒ 永远凑不够 min-seeds ⇒ **交接口径形同虚设**，
    #    而且恰恰会把"**中位过线**"的配置误判成不合格。
    #    （trial_0025 就是这种：三种子 T1 = 0.897516/0.895692/0.894412，
    #      中位 0.895692 差 0.0002pp ⇒ 确实不过线，
    #      但**理由必须是「中位不过线」而不是「只有 1 个种子」**。）
    # ⇒ 正确做法：**全部种子建组**（要求三轨指标齐全），
    #    再用**组的中位**过 T1/T2 硬门禁；单条 verdict 只作诊断信息展示。
    all_groups: dict[tuple, list[dict]] = {}
    diag_rows: list[dict] = []
    incomplete_rows: list[dict] = []
    for r in rows:
        status = (r.get("status") or "").strip().upper()
        verdict = (r.get("gate_verdict") or "").strip()
        if status != "SUCCESS":
            continue
        # ⚠️ **三轨指标必须齐全才能进交接名单**。
        #    实战踩坑（2026-11-02 冒烟）：周末那 3 条 CUDA OOM 事故臂
        #    （trial_0006/0011/0068）的门禁**事后已重跑修好**（rejudge 写了新 verdict），
        #    但 `optuna_summary.csv` 是**重判之前**生成的 ⇒ 这些行的三轨指标**是空的**，
        #    却带着 verdict=PASS + 有 utility ⇒ 会被误当成"严格 PASS 的强候选"排进交接榜。
        #    ⇒ 指标不全 = **无法验证**，一律不进交接名单，但要显式计数上报。
        if all(_f(r.get(k)) is not None for k in ("t1_f1", "t2_f1", "t3_f1")):
            all_groups.setdefault(_cfg_key(r), []).append(r)
            if verdict != STRICT:
                diag_rows.append(r)
        else:
            incomplete_rows.append(r)

    # ---- 每个配置组的中位与噪声带 ------------------------------------
    groups = []
    ineligible = []
    for key, rs in all_groups.items():
        u = [_f(r.get("utility")) for r in rs]
        t1 = [_f(r.get("t1_f1")) for r in rs]
        t2 = [_f(r.get("t2_f1")) for r in rs]
        t2fp = [_f(r.get("t2_fp")) for r in rs]
        t3 = [_f(r.get("t3_f1")) for r in rs]
        t3fp = [_f(r.get("t3_fp")) for r in rs]
        t1fp = [_f(r.get("t1_fp")) for r in rs]
        u_med = _median(u)
        u_band, n = _band(u)
        t3_med = _median(t3)
        t3_band, _ = _band(t3)
        g = {
            "key": key,
            "n": len(rs),
            "seeds": sorted({(r.get("seed") or "") for r in rs}),
            "trials": sorted(r.get("trial", "") for r in rs),
            "u_median": u_med, "u_band": u_band,
            "t1_median": _median(t1), "t1_band": _band(t1)[0], "t1_fp_median": _median(t1fp),
            "t2_median": _median(t2), "t2_fp_median": _median(t2fp),
            "t3_median": t3_med, "t3_band": t3_band, "t3_fp_median": _median(t3fp),
            "verdicts": sorted({(r.get("gate_verdict") or "").strip() for r in rs}),
            "cfg": {c: (rs[0].get(c, "") or "") for c in CFG_COLS},
            "u_vals": [v for v in u if v is not None],
            "t3_vals": [v for v in t3 if v is not None],
            "t1_vals": [v for v in t1 if v is not None],
        }
        g["u_mar_u"] = (u_med, u_band)
        g["t3_mar_t3"] = (t3_med, t3_band)
        # ---- 组级硬门禁（用中位）------------------------------------
        bad = []
        if g["t1_median"] is None or g["t1_median"] < t1_redline:
            bad.append(f"T1 中位 {g['t1_median']} < 红线 {t1_redline:.6f}")
        if g["t2_median"] is None or g["t2_median"] < t2_f1_min:
            bad.append(f"T2 F1 中位 {g['t2_median']} < {t2_f1_min}")
        if g["t2_fp_median"] is None or g["t2_fp_median"] > t2_fp_max:
            bad.append(f"T2 虚警中位 {g['t2_fp_median']} > {t2_fp_max}")
        g["eligible"] = not bad
        g["ineligible_reason"] = "；".join(bad)
        (groups if g["eligible"] else ineligible).append(g)

    groups.sort(key=lambda g: (-(g["u_median"] if g["u_median"] is not None else -9e9)))
    ineligible.sort(key=lambda g: (-(g["u_median"] if g["u_median"] is not None else -9e9)))

    # ---- 变体组（跨种子 soup / 同 run 相邻 epoch 平均）-----------------
    # ⚠️ **变体不是 seed**：它是**权重平均产物** ⇒ 混进「同配方种子中位」会污染中位
    #    ⇒ 变体**单独建组**，硬门禁用**它自己的三轨指标**判，
    #      多种子证据取**源家族**的种子数（`seed_evidence_n`）。
    #    它的 U 是**一次实测**（权重平均是确定性的）⇒ 噪声带取 0（点区间）。
    variant_groups = []
    for r in variant_rows:
        vk = (r.get("variant_key") or "").strip()
        if not vk:
            continue
        u = _f(r.get("utility"))
        t1 = _f(r.get("t1_f1")); t1fp = _f(r.get("t1_fp"))
        t2 = _f(r.get("t2_f1")); t2fp = _f(r.get("t2_fp"))
        t3 = _f(r.get("t3_f1")); t3fp = _f(r.get("t3_fp"))
        if t1 is None or t2 is None or t3 is None:
            continue
        seed_ev = int(_f(r.get("seed_evidence_n")) or 1)
        g = {
            "key": ("__variant__", vk),
            "n": 1, "seed_evidence_n": seed_ev,
            "is_variant": True,
            "variant_key": vk,
            "variant_kind": (r.get("variant_kind") or "").strip(),
            "variant_source": (r.get("variant_source") or "").strip(),
            "seed_evidence_source": (r.get("seed_evidence_source") or "").strip(),
            "weights": (r.get("weights") or "").strip(),
            "seeds": [], "trials": [vk],
            "u_median": u, "u_band": 0.0,
            "t1_median": t1, "t1_band": 0.0, "t1_fp_median": t1fp,
            "t2_median": t2, "t2_fp_median": t2fp,
            "t3_median": t3, "t3_band": 0.0, "t3_fp_median": t3fp,
            "verdicts": [(r.get("gate_verdict") or "").strip()],
            "cfg": {c: (r.get(c, "") or "") for c in CFG_COLS},
            "u_vals": [u] if u is not None else [],
            "t3_vals": [t3], "t1_vals": [t1],
        }
        g["u_mar_u"] = (g["u_median"], 0.0)
        g["t3_mar_t3"] = (g["t3_median"], 0.0)
        bad = []
        if t1 < t1_redline:
            bad.append(f"T1 {t1} < 红线 {t1_redline:.6f}")
        if t2 < t2_f1_min:
            bad.append(f"T2 F1 {t2} < {t2_f1_min}")
        if t2fp is not None and t2fp > t2_fp_max:
            bad.append(f"T2 虚警 {t2fp} > {t2_fp_max}")
        g["eligible"] = not bad
        g["ineligible_reason"] = "；".join(bad)
        variant_groups.append(g)
    variant_groups.sort(key=lambda g: (-(g["u_median"] if g["u_median"] is not None else -9e9)))

    # ---- 候选池 = 严格通道的配置组 + **过硬门禁的变体组** ------------------
    # ⚠️ 变体**单独成列**展示（不与配置榜混排）⇒ 但**参与判定链**
    #    （它本身就是可交付的 .pt 产物，U 是一次实测值）。
    pool = list(groups) + [g for g in variant_groups if g["eligible"]]
    pool.sort(key=lambda g: (-(g["u_median"] if g["u_median"] is not None else -9e9)))

    # ---- 判定链：U → T3 → 种子数 -------------------------------------
    chain = []
    winner = None
    reason = ""
    if not pool:
        reason = "没有任何**组中位过 T1/T2 硬门禁**的配置"
    else:
        # 1) U 可分？—— 相邻两组区间不重叠，且至少两组都有 n>=2
        u_sep = None
        for i in range(len(pool) - 1):
            a, b = pool[i], pool[i + 1]
            if _ev(a) >= 2 and _ev(b) >= 2 and _disjoint(a["u_mar_u"], b["u_mar_u"]):
                u_sep = (a, b)
                break
        if u_sep is not None:
            winner, runner = u_sep
            chain.append(f"U 可分：候选#{_rank(pool, winner)} 的 U 区间与下一名不重叠")
            reason = "U 可分 ⇒ 取 U 中位最高组"
        else:
            chain.append("U **不可分**（区间重叠，或头部组种子数不足无法估计噪声带）")
            # 2) 转看 T3
            t3_sep = None
            for i in range(len(pool) - 1):
                a, b = pool[i], pool[i + 1]
                if _ev(a) >= 2 and _ev(b) >= 2 and _disjoint(a["t3_mar_t3"], b["t3_mar_t3"]):
                    t3_sep = (a, b)
                    break
            if t3_sep is not None:
                winner, runner = t3_sep
                chain.append(f"T3 可分：候选#{_rank(pool, winner)} 的 T3 区间与下一名不重叠")
                reason = "U 不可分 ⇒ 转 T3 中位，T3 可分 ⇒ 取 T3 最高组"
            else:
                chain.append("T3 **仍不可分**")
                # 3) 取种子数多者（变体按**源家族**种子数计 ⇒ 证据量可比）
                mx = max(_ev(g) for g in pool)
                cands = [g for g in pool if _ev(g) == mx]
                winner = max(cands, key=lambda g: (g["u_median"] if g["u_median"] is not None else -9e9))
                chain.append(f"T3 仍不可分 ⇒ 取种子数多者（n={mx}）")
                if len(cands) > 1:
                    reason = "U/T3 均不可分 + 种子数并列 ⇒ 取 U 更高者（**需人工复核**）"
                else:
                    reason = "U/T3 均不可分 ⇒ 取种子数最多者"

    # ---- 阻塞项 --------------------------------------------------------
    blockers = []
    # ⚠️ **观察轨提醒**（2026-11-04 09:21 修）：T3 ≥ 0.87 在既定口径里是
    #    **观察轨、veto=false，不是硬门** ⇒ **未达 0.87 不构成交接阻塞**
    #    ⇒ 原来它被 `blockers.append(...)` 当成硬门失败 ⇒ 既污染 `handover_ready`
    #    （line 643：`bool(winner is not None and not blockers)`），
    #    又让读报告的人误以为「T3 没过线 ⇒ 硬门失败」。
    #    ⇒ 改：进 `observations`（**仍然显式交代**，只是不进硬门阻塞项）。
    observations = []
    if winner is None:
        blockers.append("没有任何组中位过 T1/T2 硬门禁的配置 ⇒ 无法交接")
    else:
        # ⚠️ **守住「放宽通道不得进正式交接」这条口径铁律**：
        #    组的**中位**过线 ⇒ 这个**配置**有资格交接；
        #    但如果组内**一条严格 PASS 的种子都没有**（全是 PASS_RELAXED），
        #    那么交出去的实际产物必然是放宽通道 ⇒ 必须显式列为阻塞项，交人工裁定。
        if not any(v == STRICT for v in winner.get("verdicts", [])):
            blockers.append(
                f"候选组内**没有任何一条严格 PASS 的种子**（verdict 集合 = "
                f"{winner.get('verdicts')}）⇒ 封存的实际产物必然来自放宽通道 ⇒ "
                f"按「放宽通道不得进正式交接」的口径，**只能作『候选』不能作『交接线』**，"
                f"需人工裁定")
        if _ev(winner) < args.min_seeds:
            _ev_desc = (f"**{_ev(winner)}** 个种子（变体：源家族种子数）"
                        if winner.get("is_variant") else f"**{winner['n']}** 个种子")
            blockers.append(
                f"交接候选只有 {_ev_desc}（要求 ≥{args.min_seeds}）"
                f" ⇒ 交接口径（同超参多种子中位）**尚不成立**，只能作『候选』不能作『交接线』")
        if winner["t1_median"] is not None:
            marg = winner["t1_median"] - t1_redline
            if marg < 0:
                blockers.append(f"候选 T1 中位 {winner['t1_median']:.6f} **低于红线** {t1_redline:.6f}")
            elif marg < 0.0005:
                blockers.append(
                    f"候选 T1 中位距红线仅 **{marg * 100:.4f} pp** ⇒ 余量极薄，建议再补 1 个种子确认")
        if winner["t3_median"] is not None and winner["t3_median"] < 0.87:
            observations.append(
                f"候选 T3 中位 {winner['t3_median']:.6f} **未达 0.87 观察目标**"
                f"（差 {(0.87 - winner['t3_median']) * 100:.3f} pp）"
                f" ⇒ ⚠️ **观察轨、veto=false ⇒ 不构成交接阻塞**，仅作显式交代")
        if winner["t2_fp_median"] is not None and winner["t2_fp_median"] > t2_fp_max:
            blockers.append(f"候选 T2 虚警中位 {winner['t2_fp_median']:.0f} 超上限 {t2_fp_max}")

    # ---- 输出 ----------------------------------------------------------
    lines = []
    lines.append("# no-GMC 交接口径判定报告\n")
    lines.append("## 0. 口径（硬约束）\n")
    lines.append(f"- **决策单元 = 配置（同超参）**，不是单条 trial："
                 f"先把**全部种子**按 15 维配方逐位归组（不限单条 verdict），"
                 f"再用**组的中位**过 T1/T2 硬门禁。")
    lines.append(f"- **严格通道为上位门槛**：组中位 T1 ≥ 红线、组中位 T2 F1 ≥ {t2_f1_min}、"
                 f"组中位 T2 虚警 ≤ {t2_fp_max} ⇒ 该组才有资格交接；"
                 f"单条 trial 的 `PASS_RELAXED` / `FAIL_TRACK1_RETREAT` **仅作诊断**（见附录）。")
    lines.append(f"- **同超参多种子中位**，不是单 trial 最高 U。")
    lines.append(f"- **噪声带 = 极差/2/√n**（修正口径）。")
    lines.append(f"- 判定链：**U 可分 ⇒ 取 U 最高；U 不可分 ⇒ 看 T3 中位；T3 仍不可分 ⇒ 取种子数多者**。\n")
    lines.append("## 1. 参照系与红线\n")
    lines.append("| 项 | 值 |")
    lines.append("|:--|--:|")
    lines.append(f"| T1 红线（frozen_baseline + track1_min_delta） | **{t1_redline:.16f}** |")
    lines.append(f"| T2 F1 下限 | {t2_f1_min} |")
    lines.append(f"| T2 虚警上限（基线 {fb['track2_fp']} + 增量 {gate.get('track2_fp_max_increment')}） | {t2_fp_max} |")
    lines.append(f"| T3 观察目标 | ≥ 0.87 |")
    lines.append(f"| 参照系 frozen_baseline | T1 {fb['track1_f1']:.6f} / T2 {fb['track2_f1']:.6f}(FP {fb['track2_fp']}) / T3 {fb['track3_f1']:.6f} |")
    lines.append(f"| 参照系 rc1 | T1 {ref['rc1']['track1_f1']:.6f} / T2 {ref['rc1']['track2_f1']:.6f}(FP {ref['rc1']['track2_fp']}) / T3 {ref['rc1']['track3_f1']:.6f} |")
    lines.append("")

    lines.append(f"## 2. 严格通道配置榜（组中位过 T1/T2 硬门禁；按 U 中位降序，"
                 f"共 {len(groups)} 组 / {sum(g['n'] for g in groups)} 个 trial）\n")
    lines.append("| # | n | U 中位 | U 噪声带 | T1 中位 | T1 余量 | T2 中位(FP) | T3 中位 | T3 噪声带 | 配方 |")
    lines.append("|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|")
    for i, g in enumerate(groups[: args.top], start=1):
        t1m = g["t1_median"]
        marg = (t1m - t1_redline) if t1m is not None else None
        marg_s = f"{(marg * 100):+.4f} pp" if marg is not None else "—"
        c = g["cfg"]
        recipe = (f"{c['mix_spec']}·lr{c['lr_scale']}·{c['init_from']}·{c['epochs']}ep"
                  f"·bb{c['backbone_lr_scale']}·fbb{c['freeze_backbone_epochs']}")
        lines.append(
            f"| {i} | {g['n']} | {_fmt(g['u_median'], '{:+.6f}')} | ±{g['u_band']:.6f} | "
            f"{_fmt(t1m)} | {marg_s} | {_fmt(g['t2_median'])}({_fmt(g['t2_fp_median'], '{:.0f}')}) | "
            f"{_fmt(g['t3_median'])} | ±{g['t3_band']:.6f} | `{recipe}` |")
    lines.append("")

    if incomplete_rows:
        lines.append(f"> ⚠️ **另有 {len(incomplete_rows)} 条 `verdict=PASS` 但三轨指标为空的 trial "
                     f"已被排除**（无法验证 ⇒ 不得进交接）："
                     f"{', '.join(sorted(r.get('trial', '') for r in incomplete_rows))}。"
                     f"成因通常是门禁事故后**重判了但没重跑汇总** ⇒ "
                     f"重跑 `summarize_nogmc_search.py` 即可补齐。")
        lines.append("")

    lines.append("## 3. 判定链\n")
    for c in chain:
        lines.append(f"- {c}")
    lines.append(f"- **结论：{reason}**\n")

    lines.append("## 4. 交接候选\n")
    if winner is None:
        lines.append("**无**（没有任何严格 PASS 配置）\n")
    else:
        c = winner["cfg"]
        if winner.get("is_variant"):
            lines.append(f"### {winner['variant_key']} —— **变体候选（权重平均产物，非种子）**\n")
            lines.append(f"- 变体类型：`{winner.get('variant_kind') or '?'}`"
                         f"｜源：`{winner.get('variant_source') or '?'}`")
            lines.append(f"- 权重：`{winner.get('weights') or '?'}`")
            lines.append(f"- 多种子证据：**{_ev(winner)}**"
                         f"（{winner.get('seed_evidence_source') or '源家族种子数'}）")
            lines.append("")
        else:
            lines.append(f"### {winner['trials']} — `{c['mix_spec']}·lr{c['lr_scale']}·{c['init_from']}·{c['epochs']}ep·bb{c['backbone_lr_scale']}·fbb{c['freeze_backbone_epochs']}`\n")
        lines.append("| 项 | 值 |")
        lines.append("|:--|--:|")
        lines.append(f"| 种子数 | **{winner['n']}**"
                     + (f"（多种子证据 {_ev(winner)}）" if winner.get("is_variant") else "") + " |")
        lines.append(f"| U 中位 | {winner['u_median']:+.6f} ± {winner['u_band']:.6f} |")
        lines.append(f"| T1 中位 | {winner['t1_median']:.6f}（红线 {t1_redline:.6f}，余量 {(winner['t1_median'] - t1_redline) * 100:+.4f} pp） |")
        lines.append(f"| T1 虚警中位 | {winner['t1_fp_median']:.0f}（基线 {fb['track1_fp']}） |")
        lines.append(f"| T2 中位 | {winner['t2_median']:.6f}，虚警中位 {winner['t2_fp_median']:.0f}（上限 {t2_fp_max}） |")
        lines.append(f"| T3 中位 | {winner['t3_median']:.6f} ± {winner['t3_band']:.6f}（目标 ≥ 0.87） |")
        lines.append(f"| T3 虚警中位 | {winner['t3_fp_median']:.0f} |")
        lines.append(f"| 逐 trial | {', '.join(winner['trials'])} |")
        lines.append(f"| U 逐值 | {', '.join(f'{v:+.6f}' for v in winner['u_vals'])} |")
        lines.append(f"| T1 逐值 | {', '.join(f'{v:.6f}' for v in winner['t1_vals'])} |")
        lines.append(f"| T3 逐值 | {', '.join(f'{v:.6f}' for v in winner['t3_vals'])} |")
        lines.append("")

    if variant_groups:
        lines.append(f"## 4b. 变体候选（**权重平均产物，非种子**；{len(variant_groups)} 个，"
                     f"按 U 降序）\n")
        lines.append("- 变体**不在 study 里**（不是 trial）⇒ 单独建组、单独成列；"
                     "**绝不混进同配方种子中位**（否则会污染配置中位数）。")
        lines.append("- 硬门禁用**它自己的三轨指标**判；多种子证据取**源家族**的种子数。")
        lines.append("")
        lines.append("| 变体 | 类型 | 源 | 多种子证据 | U | T1 | T1 余量 | T2(FP) | T3 | 门禁 |")
        lines.append("|:--|:--|:--|--:|--:|--:|--:|--:|--:|:--|")
        for g in variant_groups:
            t1m = g["t1_median"]
            marg = (t1m - t1_redline) if t1m is not None else None
            marg_s = f"{(marg * 100):+.4f} pp" if marg is not None else "—"
            lines.append(
                f"| `{g['variant_key']}` | {g.get('variant_kind') or '?'} | "
                f"{(g.get('variant_source') or '?')[:38]} | {_ev(g)} | "
                f"{_fmt(g['u_median'], '{:+.6f}')} | {_fmt(t1m)} | {marg_s} | "
                f"{_fmt(g['t2_median'])}({_fmt(g['t2_fp_median'], '{:.0f}')}) | "
                f"{_fmt(g['t3_median'])} | "
                + ("✅ 过硬门禁" if g["eligible"] else f"❌ {g['ineligible_reason']}") + " |")
        lines.append("")

    lines.append("## 5. 阻塞项（必须显式交代）\n")
    if blockers:
        for b in blockers:
            lines.append(f"- ⚠️ {b}")
    else:
        lines.append("- ✅ 无阻塞项")
    lines.append("")
    lines.append("## 5b. 观察轨提醒（**非硬门、veto=false**，但必须显式交代）\n")
    if observations:
        for o in observations:
            lines.append(f"- ℹ️ {o}")
    else:
        lines.append("- ✅ 无观察轨提醒")
    lines.append("")

    if ineligible:
        lines.append(f"## 附录 B. 组中位不过硬门禁的配置（**不可交接**，{len(ineligible)} 组，"
                     f"按 U 中位降序）\n")
        lines.append("| # | n | U 中位 | T1 中位 | T1 余量 | T2 中位(FP) | T3 中位 | 不过门禁的原因 | 配方 |")
        lines.append("|--:|--:|--:|--:|--:|--:|--:|:--|:--|")
        for i, g in enumerate(ineligible[: args.top], start=1):
            t1m = g["t1_median"]
            marg = (t1m - t1_redline) if t1m is not None else None
            marg_s = f"{(marg * 100):+.4f} pp" if marg is not None else "—"
            c = g["cfg"]
            recipe = (f"{c['mix_spec']}·lr{c['lr_scale']}·{c['init_from']}·{c['epochs']}ep"
                      f"·bb{c['backbone_lr_scale']}·fbb{c['freeze_backbone_epochs']}")
            lines.append(
                f"| {i} | {g['n']} | {_fmt(g['u_median'], '{:+.6f}')} | {_fmt(t1m)} | {marg_s} | "
                f"{_fmt(g['t2_median'])}({_fmt(g['t2_fp_median'], '{:.0f}')}) | {_fmt(g['t3_median'])} | "
                f"{g['ineligible_reason']} | `{recipe}` |")
        lines.append("")

    if diag_rows:
        lines.append(f"## 附录 A. 放宽通道诊断清单（**不可交接**，{len(diag_rows)} 条，仅供理解 T3/T1 权衡）\n")
        diag_rows.sort(key=lambda r: -(_f(r.get("t3_f1")) or -9e9))
        lines.append("| trial | verdict | T1 | T1 缺口 | T2 FP | T3 | 配方 |")
        lines.append("|:--|:--|--:|--:|--:|--:|:--|")
        for r in diag_rows[:20]:
            t1 = _f(r.get("t1_f1"))
            t3 = _f(r.get("t3_f1"))
            gap = (t1 - t1_redline) if t1 is not None else None
            c = {k: (r.get(k, "") or "") for k in CFG_COLS}
            recipe = (f"{c['mix_spec']}·lr{c['lr_scale']}·{c['init_from']}·{c['epochs']}ep"
                      f"·bb{c['backbone_lr_scale']}")
            lines.append(f"| {r.get('trial')} | {r.get('gate_verdict')} | {t1:.6f} | "
                         f"{gap * 100:+.4f} pp | {r.get('t2_fp')} | {t3:.6f} | `{recipe}` |"
                         if t1 is not None and t3 is not None else
                         f"| {r.get('trial')} | {r.get('gate_verdict')} | {t1} | {gap} | "
                         f"{r.get('t2_fp')} | {t3} | `{recipe}` |")
        lines.append("")

    md = "\n".join(lines)
    print(md)
    if args.out_md:
        Path(args.out_md).write_text(md + "\n", encoding="utf-8")
        print(f"[OK] 写出 {args.out_md}")
    if args.out_json:
        doc = {
            "strict_verdict": STRICT,
            "redlines": {"track1": t1_redline, "track2_f1_min": t2_f1_min,
                         "track2_fp_max": t2_fp_max, "track3_target": 0.87},
            "n_groups": len(groups),
            "n_trials_strict": sum(g["n"] for g in groups),
            "n_groups_ineligible": len(ineligible),
            "n_incomplete_excluded": len(incomplete_rows),
            "incomplete_trials": sorted(r.get("trial", "") for r in incomplete_rows),
            "chain": chain,
            "reason": reason,
            "winner": None if winner is None else {
                "trials": winner["trials"], "n": winner["n"], "cfg": winner["cfg"],
                "u_median": winner["u_median"], "u_band": winner["u_band"],
                "t1_median": winner["t1_median"], "t1_fp_median": winner["t1_fp_median"],
                "t2_median": winner["t2_median"], "t2_fp_median": winner["t2_fp_median"],
                "t3_median": winner["t3_median"], "t3_band": winner["t3_band"],
                "t3_fp_median": winner["t3_fp_median"],
                # ⚠️ **变体候选**（soup / SWA）才有这几项：它**不是 trial** ⇒
                #    固化脚本不能去 trials_root 里找 epoch*.pt ⇒ 直接用它自己的权重路径。
                **({"is_variant": True, "variant_key": winner["variant_key"],
                    "variant_kind": winner["variant_kind"],
                    "variant_source": winner["variant_source"],
                    "weights": winner["weights"],
                    "verdicts": winner["verdicts"],
                    "seed_evidence_n": winner["seed_evidence_n"]}
                   if winner.get("is_variant") else {}),
            },
            "blockers": blockers,
            "observations": observations,
            "handover_ready": bool(winner is not None and not blockers),
            "groups": [{
                "cfg": g["cfg"], "n": g["n"], "trials": g["trials"],
                "u_median": g["u_median"], "u_band": g["u_band"],
                "t1_median": g["t1_median"], "t2_median": g["t2_median"],
                "t2_fp_median": g["t2_fp_median"], "t3_median": g["t3_median"],
                "t3_band": g["t3_band"],
            } for g in groups],
            "ineligible_groups": [{
                "cfg": g["cfg"], "n": g["n"], "trials": g["trials"],
                "u_median": g["u_median"], "t1_median": g["t1_median"],
                "t2_median": g["t2_median"], "t2_fp_median": g["t2_fp_median"],
                "t3_median": g["t3_median"], "reason": g["ineligible_reason"],
            } for g in ineligible],
            # ⚠️ **变体组单列**（不与配置组混排）⇒ 审计时能看清「谁是变体、证据多少」
            #    （**仅在有变体时**才加这个键 ⇒ 不传 --variants-csv 时 JSON 逐字节不变）
            **({"variant_groups": [{
                "variant_key": g["variant_key"], "variant_kind": g["variant_kind"],
                "variant_source": g["variant_source"], "weights": g["weights"],
                "seed_evidence_n": g["seed_evidence_n"], "n": g["n"],
                "u_median": g["u_median"], "t1_median": g["t1_median"],
                "t2_median": g["t2_median"], "t2_fp_median": g["t2_fp_median"],
                "t3_median": g["t3_median"], "eligible": g["eligible"],
                "reason": g["ineligible_reason"], "verdicts": g["verdicts"],
                "cfg": g["cfg"],
            } for g in variant_groups]} if variant_groups else {}),
        }
        Path(args.out_json).write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")
        print(f"[OK] 写出 {args.out_json}")

    if winner is None:
        return 4
    if blockers:
        return 3
    return 0


def _ev(g) -> int:
    """**多种子证据条数**：变体取 `seed_evidence_n`（源家族种子数），普通配置组取组内 n。

    ⚠️ 普通配置组没有 `seed_evidence_n` ⇒ `_ev(g) == g["n"]` ⇒ **行为与引入本参数前逐位一致**。
    """
    try:
        return int(g.get("seed_evidence_n") or g.get("n") or 1)
    except Exception:
        return 1


def _rank(groups, g) -> int:
    for i, x in enumerate(groups, start=1):
        if x is g:
            return i
    return -1


if __name__ == "__main__":
    raise SystemExit(main())
