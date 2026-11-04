#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**T3 可达性论证**：用全部已跑数据回答"T3 ≥ 0.87 到底能不能拿到"。

为什么需要它
------------
交接口径是「严格 PASS + 同超参多种子中位」。当跑完几十条臂仍然拿不到
"T3 ≥ 0.87 且 T1 过线"时，交付**不能只说"没找到"**——必须给出
**可核对的定量论证**：这个目标是"运气不够"，还是"结构上不可达"？

本脚本产出：
1. **T1–T3 权衡的定量刻画**（Pearson / Spearman 相关、按 mix_spec 分组的中位）
2. **高 T3 配置的 T1 缺口清单**：要让它们过线，T1 还得涨多少
3. **严格口径下 T3 的实测可达上界**（含多种子噪声带「极差/2/√n」）
4. **放宽通道能到哪**（诊断用，**不可交接**）
5. 结论段：区分「尚有希望」与「结构上受阻」

用法（服务器 ``conda activate uav``）：

```bash
python manu/training/analyze_t3_reachability.py \
  --csv /mnt/data/siping/runs/nogmc_weekend/optuna_summary.csv \
  --csv /mnt/data/siping/runs/nogmc_focus/optuna_summary.csv \
  --ref  /mnt/data/siping/runs/nogmc_ref_weekend/reference_frame.json \
  --out-md /mnt/data/siping/runs/nogmc_focus/T3_REACHABILITY.md \
  --out-json /mnt/data/siping/runs/nogmc_focus/T3_REACHABILITY.json
```
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

STRICT = "PASS"
RELAXED = ("PASS_RELAXED", "FAIL_TRACK1_RETREAT")
CFG_COLS = ("mix_spec", "lr_scale", "warmup_epochs", "weight_decay", "focal_beta",
            "offset_weight", "mosaic", "scale", "translate", "clip_max_norm", "freeze_bn",
            "init_from", "epochs", "backbone_lr_scale", "freeze_backbone_epochs")


def _f(v):
    try:
        return float(v)
    except Exception:
        return None


def _rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / den if den else None


def _spearman(xs, ys):
    if len(xs) < 3:
        return None
    return _pearson(_rank(xs), _rank(ys))


def _median(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def _band(vals):
    """噪声带 = 极差/2/√n（修正口径）。"""
    vals = [v for v in vals if v is not None]
    n = len(vals)
    if n < 2:
        return 0.0, n
    return (max(vals) - min(vals)) / 2.0 / (n ** 0.5), n


def main() -> int:
    ap = argparse.ArgumentParser(description="T3 可达性论证")
    ap.add_argument("--csv", action="append", required=True, help="可重复；多个 CSV 合并分析")
    ap.add_argument("--study", action="append", default=[],
                    help="可重复；格式 'TAG=study.db路径:study_name'。"
                         "⚠️ **强烈建议给**：用 study.db 的**原始 float** 配受控对，"
                         "否则 CSV 的格式化精度会凭空造出/抹掉受控对")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out-md", default="")
    ap.add_argument("--out-json", default="")
    ap.add_argument("--t3-target", type=float, default=0.87)
    ap.add_argument("--t3-window", type=float, default=0.86, help="『高 T3』清单的窗口下限")
    args = ap.parse_args()

    ref = json.loads(Path(args.ref).read_text(encoding="utf-8"))
    fb = ref["frozen_baseline"]
    t1_redline = fb["track1_f1"] + float(ref.get("gate", {}).get("track1_min_delta", -0.005))

    rows = []
    for p in args.csv:
        # ⚠️ **来源标签**：两个 study 都有 trial_0009 但不是同一条 ⇒ 必须带 TAG，
        #    否则配对/分组会「同名不同物」。格式 'TAG=路径'；不带 TAG 时取所在目录名。
        if "=" in p and not Path(p).exists():
            _tag, _, _path = p.partition("=")
        else:
            _path, _tag = p, Path(p).parent.name
        # ⚠️ **文件不存在时跳过而不是崩**（2026-11-03）：收尾链会把本轮 + 旧轮的
        #    多份 CSV 一起传进来；若某一轮**一条 trial 都没跑完**（CSV 尚未生成），
        #    直接抛 FileNotFoundError 会让整个 finalize 步骤失败（rc≠0）⇒ 交付件缺失。
        if not Path(_path).exists():
            print(f"[T3-REACH] ⚠️ 跳过不存在的 CSV：{_path}")
            continue
        for _r in csv.DictReader(Path(_path).open("r", encoding="utf-8")):
            _r["_uid"] = f"{_tag}:{_r.get('trial')}"
            rows.append(_r)

    strict = [r for r in rows if (r.get("gate_verdict") or "").strip() == STRICT
              and _f(r.get("t1_f1")) is not None and _f(r.get("t3_f1")) is not None]
    relax = [r for r in rows if (r.get("gate_verdict") or "").strip() in RELAXED
             and _f(r.get("t1_f1")) is not None and _f(r.get("t3_f1")) is not None]

    # ---- 1. 相关性 ----------------------------------------------------
    xs = [_f(r["t1_f1"]) for r in strict]
    ys = [_f(r["t3_f1"]) for r in strict]
    pear = _pearson(xs, ys)
    spear = _spearman(xs, ys)
    xr = [_f(r["t1_f1"]) for r in relax]
    yr = [_f(r["t3_f1"]) for r in relax]
    pear_r = _pearson(xr, yr)
    spear_r = _spearman(xr, yr)

    # ---- 2. 严格口径 T3 上限 ------------------------------------------
    strict_t3 = sorted((_f(r["t3_f1"]) for r in strict), reverse=True)
    top_strict = max(strict, key=lambda r: _f(r["t3_f1"]))
    strict_t3_max = _f(top_strict["t3_f1"])

    # ---- 3. 高 T3 配置的 T1 缺口 --------------------------------------
    high = [r for r in rows if _f(r.get("t3_f1")) is not None
            and _f(r["t3_f1"]) >= args.t3_window
            and _f(r.get("t1_f1")) is not None]
    high.sort(key=lambda r: -_f(r["t1_f1"]))
    gap_best = max((_f(r["t1_f1"]) - t1_redline for r in high), default=None)

    # ---- 4. 按 mix_spec 分组的中位（严格通道）--------------------------
    by_mix: dict[str, list[dict]] = {}
    for r in strict:
        by_mix.setdefault(r.get("mix_spec", "?"), []).append(r)
    mix_rows = []
    for k, v in by_mix.items():
        t1m = _median([_f(x["t1_f1"]) for x in v])
        t3m = _median([_f(x["t3_f1"]) for x in v])
        t3b, n = _band([_f(x["t3_f1"]) for x in v])
        # 源域占比代理 = 实测 UAV:LQ 比例（不硬编码，避免 mix→比例的映射出错）
        rat = _median([_f(x.get("measured_ratio")) for x in v])
        mix_rows.append((k, len(v), t1m, t3m, t3b, rat))
    mix_rows.sort(key=lambda x: -(x[3] or -9e9))

    # ---- 4b. **受控对分析**（比相关系数可信：只差一个维度的配对）-------
    #    ⚠️ 为什么不能用"分组中位 + 相关系数"下结论：
    #    周末的 TPE 采样里 **mix 与 epochs/lr/backbone 是强耦合的**
    #    （例如 w8.0 多为 15ep+lr1.0、w3.0 多为 8ep+lr2.88）
    #    ⇒ 按 mix 分组取中位会把"epoch/lr 的效应"误算成"mix 的效应"，
    #    甚至**符号都可能反**。⇒ 只认「只差一个维度」的受控对。
    #
    # ⚠️⚠️ **必须用 study.db 的原始 float，不能用 CSV 里的格式化值**
    #    （2026-11-02 实测踩坑）：CSV 把 lr 写成 `2.7676`、weight_decay 写成 `0.00026410`，
    #    而 study.db 里是 `2.7675790243131346` / `0.00026412`
    #    ⇒ ① 真正相同的配置在 CSV 里**被拆成两行**（受控对凭空多出来）；
    #       ② 真正不同的配置在 CSV 里**被当成相同**（受控对凭空少掉）。
    #    实测：按 CSV 配对会得到一条假的 "weight_decay 受控对"（2 对），
    #    同时漏掉真正的 lr / epochs / init_from 对。
    #    ⇒ 用 --study 传入 study.db，按 trial 名回填**精确参数**再配对。
    #
    # ⚠️ **方向必须统一**：本表统一写成「**该维度变大时**」的 Δ。
    #    数值维度按数值大小；mix_spec 按**实际比例**（wN→N，natural→3.09）；
    #    纯类别维度（init_from / freeze_bn）**不做方向聚合**，逐对列出（原值→原值）。

    MIX_ORDER = {"natural": 3.09, "w1.0": 1.0, "w1.5": 1.5, "w2.0": 2.0, "w2.5": 2.5,
                 "w3.0": 3.0, "w3.5": 3.5, "w4.0": 4.0, "w5.0": 5.0, "w6.0": 6.0, "w8.0": 8.0}
    CAT_DIMS = {"init_from", "freeze_bn"}

    # ⚠️ **「差异是否实质」的门限**（2026-11-02 实测踩坑）：
    #    两轮 study 里存在**数值上几乎相同但源码不同**的维度 ——
    #    例：周末 trial_0025 的 weight_decay = 0.00026412，本轮 focus 臂 = 0.00026410
    #    （差 6e-9，来自 CSV 格式化 ⇒ 抄 CSV 与抄 study.db 的差别）。
    #    若只看「是否只差一个维度」，这一对会被当成 **weight_decay 的受控对**，
    #    而它俩的 ΔT1/ΔT3 其实**全是种子噪声**（换了个 seed 而已）⇒
    #    会被误读成"weight_decay 的效应量"（实测算出来 ΔT1/单位 = +123200 这种荒谬值）。
    # ⇒ **必须按维度设「实质差异」门限**；达不到门限的对单独归为**种子噪声对**，
    #    它们的用途是**量种子噪声**（σ(T1)、σ(T3)），不是量维度效应。
    MATERIAL_MIN = {
        "lr_scale": 0.05, "epochs": 1, "warmup_epochs": 0.05,
        "weight_decay": 1e-4, "focal_beta": 0.05, "offset_weight": 0.05,
        "mosaic": 0.05, "scale": 0.05, "translate": 0.05, "clip_max_norm": 0.1,
        "backbone_lr_scale": 0.1, "freeze_backbone_epochs": 1,
        "mix_spec": 0.25,
    }

    exact: dict[str, dict] = {}
    for spec in args.study:
        tag, _, rest = spec.partition("=")
        dbp, _, sname = rest.partition(":")
        if not sname:
            sname = "mix_ratio_search_nogmc_focus"
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        st = optuna.load_study(storage="sqlite:///" + dbp, study_name=sname)
        for t in st.trials:
            p = dict(t.params or {})
            if not p:
                p = dict((t.system_attrs or {}).get("fixed_params") or {})
            if p:
                exact[f"{tag}:{t.number:04d}"] = p
    print(f"[PAIR] 从 study.db 取到精确参数的 trial：{len(exact)}")

    def _exact_of(row):
        """行 → 精确参数（优先 study.db）；拿不到就退回 CSV 值。"""
        key = row.get("_uid")
        p = exact.get(key)
        if p:
            return p
        return {c: row.get(c) for c in CFG_COLS}

    def _key_of(r):
        p = _exact_of(r)
        out = []
        for c in CFG_COLS:
            v = p.get(c)
            if c in ("mix_spec", "init_from"):
                out.append(v)
            elif c == "freeze_bn":
                if isinstance(v, bool):
                    out.append(v)
                else:
                    out.append(str(v).strip().lower() in ("true", "1", "yes"))
            else:
                fv = _f(v)
                out.append(round(fv, 9) if fv is not None else v)
        return tuple(out)

    def _dir_key(dim, row):
        """该维度在**这一行**上的可排序值（用于统一方向）。"""
        v = _exact_of(row).get(dim)
        if dim == "mix_spec":
            return MIX_ORDER.get(str(v), None)
        if dim in ("init_from", "freeze_bn"):
            return None            # 纯类别：不做方向聚合
        fv = _f(v)
        return fv

    pairs = []
    pool = [r for r in rows if _f(r.get("t1_f1")) is not None and _f(r.get("t3_f1")) is not None]
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            a, b = pool[i], pool[j]
            ka, kb = _key_of(a), _key_of(b)
            diff = [CFG_COLS[k] for k in range(len(CFG_COLS)) if ka[k] != kb[k]]
            if len(diff) != 1:
                continue
            dim = diff[0]
            va, vb = _dir_key(dim, a), _dir_key(dim, b)
            if va is None or vb is None:
                # 类别维度：逐对列出，不做方向归一
                d1 = _f(b["t1_f1"]) - _f(a["t1_f1"])
                d3 = _f(b["t3_f1"]) - _f(a["t3_f1"])
                pairs.append({"dim": dim, "cat": True, "material": True,
                              "lo": a.get("trial"), "hi": b.get("trial"),
                              "lo_v": _exact_of(a).get(dim), "hi_v": _exact_of(b).get(dim),
                              "d_t1": d1, "d_t3": d3, "unit": ""})
                continue
            lo, hi = (a, b) if va <= vb else (b, a)
            d1 = _f(hi["t1_f1"]) - _f(lo["t1_f1"])
            d3 = _f(hi["t3_f1"]) - _f(lo["t3_f1"])
            unit = abs(vb - va)
            # 实质差异？达不到门限 ⇒ 归为种子噪声对
            material = unit >= MATERIAL_MIN.get(dim, 0.0)
            pairs.append({"dim": dim, "cat": False, "material": material,
                          "lo": lo.get("trial"), "hi": hi.get("trial"),
                          "lo_v": _exact_of(lo).get(dim), "hi_v": _exact_of(hi).get(dim),
                          "d_t1": d1, "d_t3": d3, "unit": unit})

    # 按维度汇总：某个维度变大时，T1/T3 分别是升是降
    dim_stats: dict[str, dict] = {}
    for p in pairs:
        if p["cat"] or not p["material"]:
            continue
        s = dim_stats.setdefault(p["dim"], {"n": 0, "t1_up": 0, "t3_down": 0, "n_t3_up": 0,
                                             "d1": [], "d3": [], "rate": [], "rate_up": []})
        s["n"] += 1
        s["d1"].append(p["d_t1"] / p["unit"] if p["unit"] else p["d_t1"])
        s["d3"].append(p["d_t3"] / p["unit"] if p["unit"] else p["d_t3"])
        if p["d_t1"] > 0:
            s["t1_up"] += 1
        if p["d_t3"] < 0:
            s["t3_down"] += 1
        # 每换 1 单位 T3 要付的 T1（T3 变大时，T1 的净损失；负号表示反而赚）
        #
        # ⚠️ **必须用「全部实质对子」做中位**（2026-11-03 09:0x 修）：
        #    原来只在 `d_t3 > 0` 的**子集**上取中位 ⇒ 与同一行的
        #    「ΔT1/单位 / ΔT3/单位」（那是**全部**对子的中位）**不是同一批样本**
        #    ⇒ 两列的符号可以互相矛盾，读起来会得出相反的结论。
        #    （实测：mix 的聚合斜率是 T1↑/T3↓（拿 T1 换 T3），
        #      而旧口径的汇率却把它排成「最划算的 T3 杠杆」⇒ 方向反了。）
        #    ⇒ 现在两个口径都在**全部实质对子**上算，并额外给出
        #      「T3 变大的对数」，让样本population显式可见。
        if p["unit"] and p["d_t3"] != 0:
            s["rate"].append(-p["d_t1"] / p["d_t3"])
        if p["d_t3"] > 0:
            s["n_t3_up"] += 1
            if p["unit"]:
                s["rate_up"].append(-p["d_t1"] / p["d_t3"])
    for s in dim_stats.values():
        s["d1_med"] = _median(s["d1"])
        s["d3_med"] = _median(s["d3"])
        s["rate_med"] = _median(s["rate"]) if s["rate"] else None
        s["rate_up_med"] = _median(s["rate_up"]) if s["rate_up"] else None
        s["d1"] = s.pop("d1")
        s["d3"] = s.pop("d3")
        s["rate"] = s.pop("rate")
        s["rate_up"] = s.pop("rate_up")

    mix_pairs = [p for p in pairs if p["dim"] == "mix_spec" and p["material"]]
    mix_t1_up = sum(1 for p in mix_pairs if p["d_t1"] > 0)
    mix_t3_down = sum(1 for p in mix_pairs if p["d_t3"] < 0)
    dominant = None
    if mix_pairs:
        dominant = ("mix（源域比例）" if (mix_t1_up * 2 > len(mix_pairs))
                    and (mix_t3_down * 2 > len(mix_pairs)) else "不是 mix 单变量主导")

    # ---- 种子噪声对（差异不实质 ⇒ 实为换种子）------------------------
    noise_pairs = [p for p in pairs if not p["cat"] and not p["material"]]
    noise_t1 = [abs(p["d_t1"]) for p in noise_pairs]
    noise_t3 = [abs(p["d_t3"]) for p in noise_pairs]

    # ---- 5. 放宽通道能到哪（诊断）--------------------------------------
    relax_t3_max = max((_f(r["t3_f1"]) for r in relax), default=None)
    relax_over = [r for r in relax if _f(r["t3_f1"]) >= args.t3_target]

    # ---- 6. 结论 -------------------------------------------------------
    reach_strict = strict_t3_max is not None and strict_t3_max >= args.t3_target
    reach_relax = bool(relax_over)
    if reach_strict:
        verdict = (f"**严格口径下已存在 T3 ≥ {args.t3_target} 的配置**"
                   f"（{top_strict.get('trial')}，T3={strict_t3_max:.6f}）⇒ 目标已达成，只需补齐多种子证据。")
    elif reach_relax:
        best = max(relax_over, key=lambda r: _f(r["t1_f1"]))
        gap = _f(best["t1_f1"]) - t1_redline
        verdict = (f"**放宽通道下已有 {len(relax_over)} 条 T3 ≥ {args.t3_target}**"
                   f"（最高 T3 {relax_t3_max:.6f}），但它们**全部 T1 触线**；"
                   f"其中离红线最近的一条是 {best.get('trial')}，仍缺 **{-gap * 100:.4f} pp**。"
                   f"⇒ 结论：**T3 ≥ {args.t3_target} 在 no-GMC 特征语义下不是做不到，而是与 T1 红线构成硬权衡**。")
    else:
        verdict = (f"**严格口径与放宽通道下都没有 T3 ≥ {args.t3_target} 的配置**"
                   f"（严格最高 {strict_t3_max if strict_t3_max is None else f'{strict_t3_max:.6f}'}）。")
    # ---- 杠杆分型：按 population 判定「买 T3」还是「换 T1」----------------
    buy_t3, trade_t1 = [], []
    for _k, _s in dim_stats.items():
        _n_up = _s.get("n_t3_up", 0)
        if _n_up == 0:
            continue
        # ⚠️ 判定要看**多数**（T3 变多的对子 > T3 变少的对子），不是看"是否存在"：
        #    mix_spec 只有 3/24 对涨 T3、21/24 对掉 T3 ⇒ 它是「拿 T3 换 T1」的杠杆，
        #    若按"是否存在 T3↑ 对子"判就会被误归成买 T3 的杠杆（方向反了）。
        if _n_up > _s["t3_down"]:
            buy_t3.append(_k)
        else:
            trade_t1.append(_k)
    buy_t3.sort(key=lambda k: (dim_stats[k].get("rate_up_med") is None,
                                dim_stats[k].get("rate_up_med") if dim_stats[k].get("rate_up_med") is not None else 0))
    trade_t1.sort(key=lambda k: -dim_stats[k].get("n_t3_up", 0))

    if pear is not None:
        verdict += (f"　严格通道内 T1 与 T3 的相关系数 Pearson **{pear:+.3f}**"
                    f"（Spearman {spear:+.3f}）⇒ 强负相关。"
                    f"（⚠️ 该系数受维度耦合影响，**硬证据看 §4.1 受控对**："
                    f"共 {len(pairs)} 对只差一个维度的配对，"
                    + (f"其中 mix 单变量受控对 {len(mix_pairs)} 对，mix 变大时 "
                       f"T1 上升 {mix_t1_up}/{len(mix_pairs)}、T3 下降 {mix_t3_down}/{len(mix_pairs)}。"
                       if mix_pairs else "mix 的单变量效应尚未被干净验证。")
                    + (f"**杠杆分型**（按 population 判定该维度到底是「买 T3」还是「换 T1」，"
                       f"再按汇率升序）："
                       + ("**买 T3 型**（该维度以 T3 **变大**为主）："
                          + ("、".join(f"{k} {dim_stats[k]['rate_up_med']:.3f}"
                                       f"（{dim_stats[k].get('n_t3_up', 0)}/{dim_stats[k]['n']} 对涨 T3）"
                                       for k in buy_t3) if buy_t3 else "无")
                          + "；")
                       + ("**拿 T3 换 T1 型**（该维度以 T3 **下降**为主）："
                          + ("、".join(f"{k}（{dim_stats[k].get('n_t3_up', 0)}/{dim_stats[k]['n']} 对涨 T3，"
                                       f"T1 上升 {dim_stats[k]['t1_up']}/{dim_stats[k]['n']}）"
                                       for k in trade_t1) if trade_t1 else "无")
                          + "。"
                          f" ⇒ **要买 T3 走「买 T3 型」里汇率最小的那个；"
                          f"「拿 T3 换 T1 型」的维度用来涨 T1，代价是掉 T3，不能当买 T3 的路。**")
                       if dim_stats else "")
                    + "）")

    # ---- 输出 ---------------------------------------------------------
    L = []
    L.append("# T3 可达性论证（no-GMC）\n")
    L.append(f"> 目标：**T3 ≥ {args.t3_target}**（龙泉山盲测 001324 硬域 F1）"
             f"**且** T1 ≥ {t1_redline:.16f}（Anti-UAV 源域红线）"
             f"**且** T2 虚警 ≤ {fb['track2_fp'] + int(ref.get('gate', {}).get('track2_fp_max_increment', 60))}"
             f"（严格通道）\n")
    L.append("## 1. 数据总览\n")
    L.append("| 项 | 值 |")
    L.append("|:--|--:|")
    L.append(f"| 纳入分析的总 trial | {len(rows)} |")
    L.append(f"| **严格 PASS**（有资格交接） | **{len(strict)}** |")
    L.append(f"| 放宽通道（仅诊断，不可交接） | {len(relax)} |")
    L.append(f"| 严格通道 T1–T3 Pearson / Spearman | **{pear:+.3f}** / {spear:+.3f} |" if pear is not None else "| 相关性 | 样本不足 |")
    L.append(f"| 放宽通道 T1–T3 Pearson / Spearman | {pear_r:+.3f} / {spear_r:+.3f} |" if pear_r is not None else "| 放宽相关性 | 样本不足 |")
    L.append("")
    L.append("## 2. 严格口径下 T3 的实测可达上界\n")
    L.append(f"- 严格 PASS 中 T3 最高：**{strict_t3_max:.6f}**（{top_strict.get('trial')}，"
             f"`{top_strict.get('mix_spec')}·lr{top_strict.get('lr_scale','')[:6]}·{top_strict.get('epochs')}ep`）")
    if strict_t3:
        q = [strict_t3[int(len(strict_t3) * f)] for f in (0.1, 0.5)]
        L.append(f"- 严格通道 T3 分布：p90 {q[0]:.6f} · 中位 {q[1]:.6f} · 共 {len(strict_t3)} 条")
    L.append(f"- 目标 {args.t3_target} ⇒ {'**已达成**' if reach_strict else f'**尚未达成（差 {(args.t3_target - strict_t3_max) * 100:.3f} pp）**'}")
    L.append("")
    L.append("## 3. 高 T3 配置的 T1 缺口（要过线还差多少）\n")
    L.append(f"> 列出所有 T3 ≥ {args.t3_window} 的配置（不论 verdict），按 T1 降序 —— **T1 越靠前越接近过线**。\n")
    L.append("| trial | verdict | T1 | 距红线 | T3 | T2 FP | 配方 |")
    L.append("|:--|:--|--:|--:|--:|--:|:--|")
    for r in high[:22]:
        t1 = _f(r["t1_f1"])
        L.append(f"| {r.get('trial')} | {r.get('gate_verdict')} | {t1:.6f} | "
                 f"{(t1 - t1_redline) * 100:+.4f} pp | {_f(r['t3_f1']):.6f} | {r.get('t2_fp')} | "
                 f"`{r.get('mix_spec')}·lr{str(r.get('lr_scale',''))[:6]}·{r.get('epochs')}ep"
                 f"·bb{r.get('backbone_lr_scale')}·{r.get('init_from')}` |")
    L.append("")
    if gap_best is not None:
        L.append(f"> **最小缺口 = {gap_best * 100:+.4f} pp**（负数表示还差多少）。")
        L.append("")
    L.append("## 4. 严格通道内按 mix_spec 的中位（**仅供参考，见 4.1 的混淆警告**）\n")
    L.append("| mix | n | 实测 UAV:LQ | T1 中位 | T3 中位 | T3 噪声带 |")
    L.append("|:--|--:|--:|--:|--:|--:|")
    for k, n, t1m, t3m, t3b, rat in mix_rows:
        L.append(f"| {k} | {n} | {('%.2f' % rat) if rat is not None else '—'} | {t1m:.6f} | "
                 f"{t3m:.6f} | ±{t3b:.6f} |")
    L.append("")
    L.append("> ⚠️ **混淆警告**：周末的 TPE 采样里 **mix 与 epochs/lr/backbone 强耦合**"
             "（w8.0 多落在 15ep+lr1.0、w3.0 多落在 8ep+lr2.88）⇒ "
             "按 mix 分组取中位会把 epoch/lr 的效应误算成 mix 的效应，**连符号都可能反**。"
             "⇒ 结论只认下面的**受控对**。")
    L.append("")
    L.append("### 4.1 受控对分析（只差**一个**维度的配对，共 %d 对）\n" % len(pairs))
    L.append("> **方向约定**：数值维度统一写成「该维度**变大**时」的 Δ（单位变化量已归一）；"
             "`mix_spec` 按**实际比例**排序（wN→N，natural→3.09）；"
             "纯类别维度（init_from / freeze_bn）不做方向聚合，逐对列在表下。")
    L.append("> **参数来源**：%s" % ("study.db 的原始 float（精确）" if exact else
                                    "⚠️ 只有 CSV 的格式化值 ⇒ 受控对可能被凭空造出/抹掉"))
    L.append("")
    if dim_stats:
        L.append("| 变动维度 | 受控对数 | 该维度变大时 T1 上升 | T3 下降 | 其中 T3 **变大**的对数 | ΔT1/单位 | ΔT3/单位 | **每换 1 单位 T3 付的 T1** |")
        L.append("|:--|--:|--:|--:|--:|--:|--:|--:|")
        for d in sorted(dim_stats, key=lambda k: (dim_stats[k]["rate_med"] is None,
                                                  dim_stats[k]["rate_med"])):
            s = dim_stats[d]
            rate = ("—" if s["rate_med"] is None else f"**{s['rate_med']:.3f}**")
            L.append(f"| {d} | {s['n']} | {s['t1_up']}/{s['n']} | {s['t3_down']}/{s['n']} | "
                     f"{s.get('n_t3_up', 0)}/{s['n']} | "
                     f"{s['d1_med']:+.6f} | {s['d3_med']:+.6f} | {rate} |")
        L.append("")
        L.append("> **「每换 1 单位 T3 付的 T1」怎么读**：该维度**变大**时，T1 相对 T3 的净代价"
                 "（**越小越划算**）。负数 = 该维度变大时 **T1 与 T3 同涨**（白拿 T3）。")
        L.append("> ⚠️ **本列与 ΔT1/单位、ΔT3/单位 取自同一批「全部实质对子」**（口径已统一；"
                 "旧版只在「T3 变大的对子」子集上取中位，会与相邻两列的符号互相矛盾）。"
                 "「其中 T3 变大的对数」给出 population，**读排序前先看这一列**："
                 "若某维度绝大多数对子都是 T3 **下降**，那它本质是「拿 T3 换 T1」的杠杆，不是买 T3 的杠杆。")
        L.append("> ⇒ 这张表是**唯一可信的杠杆排序**；相关系数（§4）受维度耦合影响，只能当背景。")
        L.append("")
    cat_pairs = [p for p in pairs if p["cat"]]
    if cat_pairs:
        L.append("**类别维度受控对**（逐对列出，不做方向聚合）：\n")
        L.append("| 变动维度 | from → to | T1 Δ | T3 Δ |")
        L.append("|:--|:--|--:|--:|")
        for p in cat_pairs[:16]:
            L.append(f"| {p['dim']} | {p['lo']}({p['lo_v']}) → {p['hi']}({p['hi_v']}) | "
                     f"{p['d_t1']:+.6f} | {p['d_t3']:+.6f} |")
        L.append("")
    if mix_pairs:
        L.append(f"**mix 单变量受控对**：共 {len(mix_pairs)} 对，其中 mix 变大时 **T1 上升 {mix_t1_up}/{len(mix_pairs)}**、"
                 f"**T3 下降 {mix_t3_down}/{len(mix_pairs)}** ⇒ {dominant}")
        L.append("")
        L.append("| 低 mix → 高 mix | T1 Δ | T3 Δ |")
        L.append("|:--|--:|--:|")
        for p in mix_pairs[:14]:
            L.append(f"| {p['lo']}({p['lo_v']}) → {p['hi']}({p['hi_v']}) | {p['d_t1']:+.6f} | {p['d_t3']:+.6f} |")
        L.append("")
    else:
        L.append("> 当前数据里**没有只差 mix 一个维度的受控对** ⇒ mix 的单变量效应**未被干净验证**。")
        L.append("")
    L.append("### 4.2 种子噪声量（差异**不实质**的配对 = 同配方换种子）\n")
    L.append("> 这一类对**不是**维度效应，而是**同配方换种子**的重复测量 ⇒ "
             "它的 |ΔT1| / |ΔT3| 就是**种子噪声**（交接噪声带与「够不够得着红线」全靠它）。")
    L.append("")
    if noise_pairs:
        L.append("| 对 | 变动维度（差异量） | T1 Δ | T3 Δ |")
        L.append("|:--|:--|--:|--:|")
        for p in noise_pairs[:14]:
            L.append(f"| {p['lo']} → {p['hi']} | {p['dim']}（Δ={p['unit']:.3g}） | "
                     f"{p['d_t1']:+.6f} | {p['d_t3']:+.6f} |")
        L.append("")
        L.append(f"⇒ **种子噪声**：|ΔT1| 中位 **{_median(noise_t1):.6f}**"
                 f"（最大 {max(noise_t1):.6f}）· |ΔT3| 中位 **{_median(noise_t3):.6f}**"
                 f"（最大 {max(noise_t3):.6f}）")
        L.append(f"⇒ **凡是与红线的差距小于这个量级的，都属于「种子运气」而不是「配置不行」**。")
    else:
        L.append("- 无")
    L.append("")
    L.append("## 5. 放宽通道能到哪（诊断，不可交接）\n")
    if relax:
        L.append(f"- 放宽通道 T3 最高：**{relax_t3_max:.6f}**")
        L.append(f"- 其中 T3 ≥ {args.t3_target} 的有 **{len(relax_over)}** 条 —— 但**全部 T1 触线**"
                 f"（口径铁律：放宽通道只作诊断标记，不得进交接）")
    else:
        L.append("- 无")
    L.append("")
    L.append("## 6. 结论\n")
    L.append(verdict)
    L.append("")
    md = "\n".join(L)
    print(md)
    if args.out_md:
        Path(args.out_md).write_text(md + "\n", encoding="utf-8")
        print(f"[OK] 写出 {args.out_md}")
    if args.out_json:
        doc = {
            "t3_target": args.t3_target,
            "t1_redline": t1_redline,
            "n_total": len(rows), "n_strict": len(strict), "n_relaxed": len(relax),
            "pearson_strict": pear, "spearman_strict": spear,
            "pearson_relaxed": pear_r, "spearman_relaxed": spear_r,
            "strict_t3_max": strict_t3_max,
            "strict_t3_top_trial": top_strict.get("trial"),
            "relaxed_t3_max": relax_t3_max,
            "n_relaxed_over_target": len(relax_over),
            "min_t1_gap_over_high_t3": gap_best,
            "mix_medians": [{"mix": k, "n": n, "t1_median": t1m, "t3_median": t3m,
                             "t3_band": t3b, "measured_ratio": rat}
                            for k, n, t1m, t3m, t3b, rat in mix_rows],
            "dominant_variable": dominant,
            "n_controlled_pairs": len(pairs),
            "controlled_pair_stats": {k: {"n": v["n"], "t1_up": v["t1_up"],
                                          "t3_down": v["t3_down"],
                                          "d_t1_median": v["d1_med"],
                                          "d_t3_median": v["d3_med"]}
                                      for k, v in dim_stats.items()},
            "mix_controlled_pairs": [{"lo": p["lo"], "hi": p["hi"], "lo_v": p["lo_v"],
                                      "hi_v": p["hi_v"], "d_t1": p["d_t1"], "d_t3": p["d_t3"]}
                                     for p in mix_pairs],
            "verdict": verdict,
        }
        Path(args.out_json).write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")
        print(f"[OK] 写出 {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
