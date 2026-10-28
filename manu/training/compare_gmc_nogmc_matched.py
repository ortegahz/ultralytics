#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GMC vs no-GMC 的**配对 A/B** 对照（同一 trial 号 = 同一超参向量 + 同一种子）。

为什么需要这个
--------------
§十五.5 那条结论（「no-GMC 语义下微调净效用 ≈ 0」）用的是
**基线 vs 基线** 的对照：474 去 P0 在 GMC 特征下 vs 474 去 P0 在 no-GMC 特征下。
它是干净的，但**只测了底模**，没测「同一条训练配方在两种语义下差多少」。

本脚本补上后一层：两个 study 的**前若干 trial 用的是同一套种子采样序列**
⇒ 同一个 trial 号 = **同一个 mix_spec + 同一组超参 + 同一种子 + 同一样本序列**
⇒ 两边指标可以直接相减，得到**受控的**「去掉 GMC 值多少」。

口径纪律
--------
· 只做**配对差**，不做任何横比 U（GMC 的 U 与 no-GMC 的 U 量纲不同，跨语义不可比）。
· 指标**一个都不重算**：全部取落盘 CSV 列。
· 配不上（参数不一致 / 缺行）的 trial **不参与**，并单独报出来。

用法
----
```bash
cd /tmp/pycharm_project_10ae9e2e
python manu/training/compare_gmc_nogmc_matched.py \
  --gmc-csv runs/optuna_mix_ratio/optuna_summary.csv \
  --nogmc-csv runs/optuna_nogmc/optuna_summary.csv \
  --max-trials 12
```
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

# 判定「同一个 trial」必须一致的**全部**调参维度（少一个就不算配对）
PARAMS = (
    "mix_spec", "lr_scale", "warmup_epochs", "weight_decay", "focal_beta",
    "offset_weight", "mosaic", "scale", "translate", "clip_max_norm", "freeze_bn",
)


def _f(row: dict, key: str) -> float:
    try:
        return float(row.get(key) or 0)
    except (TypeError, ValueError):
        return 0.0


def load_csv(path: Path) -> dict:
    out: dict = {}
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name = row.get("trial")
            if name:
                out[name] = row
    return out


# 三轨 GT 规模（两侧**完全相同** ⇒ 配对差的前提）：
# Anti-UAV val 25,111 ｜ 龙泉山黄金池 16,638 ｜ 盲测难例 10,535。
# 来源：`auto_epoch_gate_check.BASE[*]["gt"]`（运行期读模块常量，读不到才用这里的兜底值）。
GT_FALLBACK = {"t1": 25111, "t2": 16638, "t3": 10535}


def gt_counts() -> dict:
    try:
        from manu.training.auto_epoch_gate_check import BASE  # noqa: E402

        return {
            "t1": BASE["track1_uav"]["gt"],
            "t2": BASE["track2_golden"]["gt"],
            "t3": BASE["track3_blind"]["gt"],
        }
    except Exception:
        return dict(GT_FALLBACK)


def derive_tp_recall_prec(f1: float, fp: float, gt: int) -> tuple[float, float, float]:
    """由 (F1, FP, GT) 反解 (TP, Recall, Precision)。

    推导：F1 = 2TP/(2TP+FP+FN)，FN = GT − TP
         ⇒ TP(2−F1) = F1×(FP+GT) ⇒ TP = F1×(FP+GT)/(2−F1)
    ⚠️ F1 取自落盘 CSV（6 位小数）⇒ 反解值有 ~1e-4 量级的舍入误差，
    只用于**归因**（掉的是召回还是虚警），不当精确指标用。
    """
    if not gt or f1 <= 0:
        return 0.0, 0.0, 0.0
    tp = f1 * (fp + gt) / (2.0 - f1)
    rec = tp / gt
    den = tp + fp
    prec = (tp / den) if den > 0 else 0.0
    return tp, rec, prec


def gmc_base() -> dict:
    """GMC 侧冻结基线（474 去 P0）—— 从门禁脚本的模块常量读，不手抄。"""
    from manu.training.auto_epoch_gate_check import BASE  # noqa: E402

    return {"track1_f1": BASE["track1_uav"]["f1"], "track3_f1": BASE["track3_blind"]["f1"]}


def nogmc_base(ref_path: str) -> dict:
    """no-GMC 侧冻结基线 —— 从参照系包读，不手抄。"""
    d = json.loads(Path(ref_path).read_text(encoding="utf-8"))
    fb = d.get("frozen_baseline") or {}
    return {"track1_f1": fb["track1_f1"], "track3_f1": fb["track3_f1"]}


def _wilcoxon_exact_p(diffs: list[float]) -> tuple[float, float]:
    """无 scipy 时的**精确** Wilcoxon 符号秩（平均秩处理并列，子集和 DP）。

    返回 (W+, 双侧 p)。n 很小（配对数），2^n DP 足够。
    """
    nz = [d for d in diffs if d != 0]
    n = len(nz)
    if n == 0:
        return 0.0, 1.0
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w_plus = sum(r for r, d in zip(ranks, nz) if d > 0)
    # 子集和 DP：count[s] = 取若干秩之和为 s 的方案数
    counts = {0: 1}
    for r in ranks:
        nxt = dict(counts)
        for s, c in counts.items():
            nxt[s + r] = nxt.get(s + r, 0) + c
        counts = nxt
    total = 2 ** n
    le = sum(c for s, c in counts.items() if s <= w_plus)
    ge = sum(c for s, c in counts.items() if s >= w_plus)
    p = min(1.0, min(le, ge) / total)
    return w_plus, p


def paired_test(diffs: list[float]) -> dict:
    """配对检验（house rule §三·6c：**配对数据必须用配对检验判显著性**）。

    · 主检验：Wilcoxon 符号秩（双侧；n ≤ 25 用精确分布）
    · 辅检：符号检验（binomial，双侧）
    · 效应量：配对差中位数 + Hodges–Lehmann 估计量（中位数的成对平均）
    """
    nz = [d for d in diffs if d != 0]
    n = len(diffs)
    method = "scipy.wilcoxon(exact)"
    w = None
    p = None
    try:
        from scipy.stats import wilcoxon  # noqa: E402

        if nz:
            stat, pv = wilcoxon(nz, alternative="two-sided", zero_method="wilcox")
            w, p = float(stat), float(pv)
        else:
            p = 1.0
    except Exception:
        w, p = _wilcoxon_exact_p(diffs)
        method = "exact-DP(自实现)"
    pos = sum(1 for d in nz if d > 0)
    neg = len(nz) - pos
    m = len(nz)
    if m:
        k = min(pos, neg)
        p_sign = min(1.0, 2.0 * sum(_comb(m, i) for i in range(k + 1)) / (2.0 ** m))
    else:
        p_sign = 1.0
    hl = None
    if diffs:
        all_avg = [(a + b) / 2.0 for a in diffs for b in diffs]
        hl = statistics.median(all_avg)
    return {
        "n": n, "n_nonzero": m, "w": w, "p": p, "method": method,
        "sign_p": p_sign, "n_pos": pos, "n_neg": neg,
        "median": statistics.median(diffs) if diffs else None,
        "hodges_lehmann": hl,
    }


def _comb(n: int, k: int) -> float:
    from math import comb

    return float(comb(n, k))


def seed_noise(csv_path: Path, max_trials: int = 200) -> list[dict]:
    """**同超参不同种子**的重复组 ⇒ 跨种子极差（= 种子噪声带的实测值）。

    用途：判断「某候选比另一个高 X pp」是不是种子噪声。
    只有**完全相同**的 11 个调参维度才算同一配置；种子 = trial 号（同配置不同 trial 号 ⇒ 不同种子）。
    """
    rows = load_csv(csv_path)
    groups: dict = {}
    for name in sorted(rows, key=_trial_sort_key):
        if _trial_index(name) >= max_trials:
            continue
        groups.setdefault(_param_tuple(rows[name]), []).append(name)
    out: list[dict] = []
    for key, names in groups.items():
        if len(names) < 2:
            continue
        t1 = [_f(rows[n], "t1_f1") for n in names]
        t2 = [_f(rows[n], "t2_f1") for n in names]
        t3 = [_f(rows[n], "t3_f1") for n in names]
        fp2 = [_f(rows[n], "t2_fp") for n in names]
        fp3 = [_f(rows[n], "t3_fp") for n in names]
        out.append({
            "mix_spec": key[0], "lr_scale": key[1],
            "trials": names, "n_seeds": len(names),
            "t1_range_pp": (max(t1) - min(t1)) * 100.0,
            "t2_range_pp": (max(t2) - min(t2)) * 100.0,
            "t3_range_pp": (max(t3) - min(t3)) * 100.0,
            "t2_fp_range": max(fp2) - min(fp2),
            "t3_fp_range": max(fp3) - min(fp3),
            "t3_values": t3, "t1_values": t1,
        })
    out.sort(key=lambda r: -r["n_seeds"])
    return out


def lr_bands(gmc_csv: Path, nogmc_csv: Path, gmc_base: dict | None = None,
             nogmc_base: dict | None = None, max_trials: int = 200) -> list[dict]:
    """按 ``lr_scale`` 分档，给出**两个语义各自**的「源域代价 ↔ 难例增益」中位数。

    每个 trial 的 Δ 都相对**自己那一侧**的冻结基线（474 去 P0），
    ⇒ 两侧的 Δ 可直接比大小（「训练把模型从自己的起点推动了多少」）。

    档位（按 lr_scale）：
    ``<0.7`` / ``0.7~1.2`` / ``1.2~1.8`` / ``1.8~2.4`` / ``2.4~3.0+``
    """
    bands = [(0.0, 0.7), (0.7, 1.2), (1.2, 1.8), (1.8, 2.4), (2.4, 99.0)]
    labels = ["<0.7", "0.7~1.2", "1.2~1.8", "1.8~2.4", "≥2.4"]
    out: list[dict] = []
    for (lo, hi), label in zip(bands, labels):
        row: dict = {"band": label, "lo": lo, "hi": hi}
        for side, csv_path, base in (("gmc", gmc_csv, gmc_base), ("nogmc", nogmc_csv, nogmc_base)):
            if base is None:
                row[side] = None
                continue
            xs: list[tuple[float, float, float]] = []
            for name, r in load_csv(csv_path).items():
                if _trial_index(name) >= max_trials:
                    continue
                lr = _f(r, "lr_scale")
                if not (lo <= lr < hi):
                    continue
                d1 = (_f(r, "t1_f1") - base["track1_f1"]) * 100.0
                d3 = (_f(r, "t3_f1") - base["track3_f1"]) * 100.0
                xs.append((lr, d1, d3))
            if not xs:
                row[side] = None
                continue
            row[side] = {
                "n": len(xs),
                "lr_median": statistics.median([x[0] for x in xs]),
                "d_t1_median_pp": statistics.median([x[1] for x in xs]),
                "d_t3_median_pp": statistics.median([x[2] for x in xs]),
                "d_t1_min_pp": min(x[1] for x in xs),
                "d_t1_max_pp": max(x[1] for x in xs),
                "d_t3_min_pp": min(x[2] for x in xs),
                "d_t3_max_pp": max(x[2] for x in xs),
            }
        out.append(row)
    return out


def _rp_block(gr: dict, nr: dict, gts: dict) -> dict:
    """逐轨的召回/虚警分解（GMC 侧 vs no-GMC 侧 vs 差）。"""
    out: dict = {}
    for key, f1k, fpk in (("t1", "t1_f1", "t1_fp"), ("t2", "t2_f1", "t2_fp"), ("t3", "t3_f1", "t3_fp")):
        gt = gts[key]
        g_tp, g_rec, g_prec = derive_tp_recall_prec(_f(gr, f1k), _f(gr, fpk), gt)
        n_tp, n_rec, n_prec = derive_tp_recall_prec(_f(nr, f1k), _f(nr, fpk), gt)
        out[key] = {
            "gt": gt,
            "tp_gmc": g_tp, "tp_nogmc": n_tp,
            "recall_gmc": g_rec, "recall_nogmc": n_rec,
            "d_recall_pp": (n_rec - g_rec) * 100.0,
            "prec_gmc": g_prec, "prec_nogmc": n_prec,
            "d_prec_pp": (n_prec - g_prec) * 100.0,
        }
    return out


def _trial_index(name: str) -> int:
    try:
        return int(str(name).split("_")[-1])
    except ValueError:
        return 10 ** 9


def _trial_sort_key(name: str) -> int:
    return _trial_index(name)


def _param_tuple(row: dict) -> tuple:
    return tuple((row.get(k) or "") for k in PARAMS)


def collect_pairs(gmc_csv: Path, nogmc_csv: Path, max_trials: int,
                  mode: str = "index") -> dict:
    """收集**配对** trial（供 CLI 与汇总器共用）。

    ``mode``：
      · ``"index"``：同 trial 号配对（⇒ 同超参 **且** 同种子 **且** 同样本序列，最严格）；
      · ``"config"``：**同超参**配对（可跨索引 ⇒ 同配置不同种子，样本更多、更抗单点噪声）。

    返回 dict：``matched`` / ``mismatched`` / ``missing`` / ``rows`` / ``aggregate``。
    ``rows`` 每项含两侧的 T1/T2/T3 F1 与 FP 及其差值（指标**一个都不重算**，全取落盘 CSV 列）。
    """
    g, n = load_csv(gmc_csv), load_csv(nogmc_csv)

    def _max_idx(d: dict) -> int:
        best = -1
        for name in d:
            try:
                best = max(best, int(str(name).split("_")[-1]))
            except ValueError:
                continue
        return best

    # 只在**两侧 CSV 实际覆盖的范围**内找配对，
    # 否则「缺行」会把 max_trials 之外的全域也算进去（数字没有意义）。
    span = min(max(_max_idx(g), _max_idx(n)) + 1, max_trials)
    matched: list[str] = []
    mismatched: list[dict] = []
    missing: list[str] = []
    pairs: list[tuple[str, str]] = []
    if mode == "config":
        # 同超参（跨索引）配对：按参数元组分组，两侧各自按 trial 号排序后 1:1 配
        by_g: dict = {}
        for name in sorted(g, key=_trial_sort_key):
            if _trial_index(name) < span:
                by_g.setdefault(_param_tuple(g[name]), []).append(name)
        by_n: dict = {}
        for name in sorted(n, key=_trial_sort_key):
            if _trial_index(name) < span:
                by_n.setdefault(_param_tuple(n[name]), []).append(name)
        for key in sorted(set(by_g) & set(by_n)):
            for gt, nt in zip(by_g[key], by_n[key]):
                pairs.append((gt, nt))
                matched.append(f"{gt}<->{nt}")
    else:
        for i in range(span):
            t = f"trial_{i:04d}"
            if t not in g or t not in n:
                missing.append(t)
                continue
            diffs = {
                k: [g[t].get(k), n[t].get(k)]
                for k in PARAMS
                if (g[t].get(k) or "") != (n[t].get(k) or "")
            }
            if diffs:
                mismatched.append({"trial": t, "diff": diffs})
            else:
                pairs.append((t, t))
                matched.append(t)

    d1: list[float] = []
    d2: list[float] = []
    d3: list[float] = []
    f2: list[float] = []
    f3: list[float] = []
    gts = gt_counts()
    rows_out = []
    for gt, nt in pairs:
        gr, nr = g[gt], n[nt]
        a1, b1 = _f(gr, "t1_f1"), _f(nr, "t1_f1")
        a2, b2 = _f(gr, "t2_f1"), _f(nr, "t2_f1")
        a3, b3 = _f(gr, "t3_f1"), _f(nr, "t3_f1")
        dfp2 = _f(nr, "t2_fp") - _f(gr, "t2_fp")
        dfp3 = _f(nr, "t3_fp") - _f(gr, "t3_fp")
        d1.append(b1 - a1)
        d2.append(b2 - a2)
        d3.append(b3 - a3)
        f2.append(dfp2)
        f3.append(dfp3)
        rows_out.append({
            "trial": gt if gt == nt else f"{gt}<->{nt}",
            "trial_gmc": gt, "trial_nogmc": nt,
            "mix_spec": gr.get("mix_spec"), "lr_scale": _f(gr, "lr_scale"),
            "measured_ratio_gmc": _f(gr, "measured_ratio"),
            "t1_f1_gmc": a1, "t1_f1_nogmc": b1, "d_t1_f1": b1 - a1,
            "t2_f1_gmc": a2, "t2_f1_nogmc": b2, "d_t2_f1": b2 - a2,
            "t3_f1_gmc": a3, "t3_f1_nogmc": b3, "d_t3_f1": b3 - a3,
            "d_t2_fp": int(dfp2), "d_t3_fp": int(dfp3),
            "rp": _rp_block(gr, nr, gts),
        })

    aggregate: dict = {}
    if matched:
        aggregate = {
            "d_t1_f1_median_pp": statistics.median(d1) * 100,
            "d_t2_f1_median_pp": statistics.median(d2) * 100,
            "d_t3_f1_median_pp": statistics.median(d3) * 100,
            "d_t1_f1_min_pp": min(d1) * 100, "d_t1_f1_max_pp": max(d1) * 100,
            "d_t2_f1_min_pp": min(d2) * 100, "d_t2_f1_max_pp": max(d2) * 100,
            "d_t3_f1_min_pp": min(d3) * 100, "d_t3_f1_max_pp": max(d3) * 100,
            "d_t2_fp_median": statistics.median(f2),
            "d_t3_fp_median": statistics.median(f3),
        }
        for key in ("t1", "t2", "t3"):
            aggregate[f"d_{key}_recall_median_pp"] = statistics.median(
                [r["rp"][key]["d_recall_pp"] for r in rows_out])
            aggregate[f"d_{key}_prec_median_pp"] = statistics.median(
                [r["rp"][key]["d_prec_pp"] for r in rows_out])
        # ★ 配对检验（house rule §三·6c）
        aggregate["paired_tests"] = {
            "t1_f1": paired_test([r["d_t1_f1"] for r in rows_out]),
            "t2_f1": paired_test([r["d_t2_f1"] for r in rows_out]),
            "t3_f1": paired_test([r["d_t3_f1"] for r in rows_out]),
        }
    return {
        "matched": matched, "mismatched": mismatched, "missing": missing,
        "rows": rows_out, "aggregate": aggregate,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="GMC vs no-GMC 配对 A/B 对照")
    ap.add_argument("--gmc-csv", default="runs/optuna_mix_ratio/optuna_summary.csv")
    ap.add_argument("--nogmc-csv", default="runs/optuna_nogmc/optuna_summary.csv")
    ap.add_argument("--max-trials", type=int, default=12, help="只比前 N 个 trial（默认 12）")
    ap.add_argument("--pair-mode", choices=("index", "config"), default="index",
                    help="index=同 trial 号配对（同超参+同种子，最严格）；"
                         "config=同超参配对（可跨索引，样本更多、更抗单点噪声）")
    ap.add_argument("--out-json", default="", help="可选：把配对表写成 JSON")
    ap.add_argument("--ref", default="runs/nogmc_ref/reference_frame.json",
                    help="参照系包（读 no-GMC 冻结基线，用于 lr 分档）")
    args = ap.parse_args()

    g_path, n_path = Path(args.gmc_csv), Path(args.nogmc_csv)
    for p in (g_path, n_path):
        if not p.is_file():
            print(f"[FAIL] 找不到 {p}")
            return 3

    data = collect_pairs(g_path, n_path, args.max_trials, mode=args.pair_mode)
    matched = data["matched"]
    mismatched = data["mismatched"]
    missing = data["missing"]
    rows_out = data["rows"]

    print("=" * 118)
    print(f"GMC vs no-GMC 配对 A/B ｜ 配对 {len(matched)} 个 ｜ 参数不一致 {len(mismatched)} 个 ｜ 缺行 {len(missing)} 个")
    print(f"  配对模式：**{args.pair_mode}** ｜ 配对判据：mix_spec + 全部调参维度逐字相同"
          + ("（同 trial 号 ⇒ 同超参 + 同种子 + 同样本序列）" if args.pair_mode == "index"
             else "（可跨索引 ⇒ 同配置不同种子）"))
    print("=" * 118)
    for m in mismatched:
        print(f"  ⚠️ 不配对 {m['trial']}: {m['diff']}")
    if missing:
        print(f"  ⚠️ 缺行: {missing}")
    if not matched:
        print("[FAIL] 没有任何可配对的 trial")
        return 3

    print()
    print("%-11s %-8s %-7s %-6s | %-24s | %-24s | %-24s | %-13s" % (
        "trial", "mix", "lr_scale", "实测比", "T1 F1 (GMC / no-GMC / Δ)",
        "T2 F1 (GMC / no-GMC / Δ)", "T3 F1 (GMC / no-GMC / Δ)", "ΔFP T2 / T3"))
    d1 = [r["d_t1_f1"] for r in rows_out]
    d2 = [r["d_t2_f1"] for r in rows_out]
    d3 = [r["d_t3_f1"] for r in rows_out]
    f2 = [float(r["d_t2_fp"]) for r in rows_out]
    f3 = [float(r["d_t3_fp"]) for r in rows_out]
    for r in rows_out:
        print("%-11s %-8s %-7.4f %-6s | %-24s | %-24s | %-24s | %+d / %+d" % (
            r["trial"], r["mix_spec"] or "", r["lr_scale"],
            f"{r['measured_ratio_gmc']:.2f}",
            f"{r['t1_f1_gmc']:.6f} / {r['t1_f1_nogmc']:.6f} / {r['d_t1_f1']:+.6f}",
            f"{r['t2_f1_gmc']:.6f} / {r['t2_f1_nogmc']:.6f} / {r['d_t2_f1']:+.6f}",
            f"{r['t3_f1_gmc']:.6f} / {r['t3_f1_nogmc']:.6f} / {r['d_t3_f1']:+.6f}",
            r["d_t2_fp"], r["d_t3_fp"],
        ))

    def agg(xs, scale=1.0, unit="") -> str:
        if not xs:
            return "（无）"
        return (f"中位 {statistics.median(xs) * scale:+.6f}{unit}"
                f" | 均值 {statistics.mean(xs) * scale:+.6f}{unit}"
                f" | 区间 {min(xs) * scale:+.6f} ~ {max(xs) * scale:+.6f}{unit}")

    print()
    print(f"  T1 ΔF1（no-GMC − GMC）：{agg(d1, 100, ' pp')}")
    print(f"  T2 ΔF1（no-GMC − GMC）：{agg(d2, 100, ' pp')}")
    print(f"  T3 ΔF1（no-GMC − GMC）：{agg(d3, 100, ' pp')}")
    print(f"  ΔT2 虚警（no-GMC − GMC）：中位 {statistics.median(f2):+.1f} | 区间 {min(f2):+.0f} ~ {max(f2):+.0f}")
    print(f"  ΔT3 虚警（no-GMC − GMC）：中位 {statistics.median(f3):+.1f} | 区间 {min(f3):+.0f} ~ {max(f3):+.0f}")
    print()
    print("  召回/虚警分解（由落盘 F1+FP 与 GT 反解 TP，~1e-4 舍入误差，只用于归因）：")
    for key, name in (("t1", "T1 源域  "), ("t2", "T2 黄金池"), ("t3", "T3 难例  ")):
        dr = statistics.median([r["rp"][key]["d_recall_pp"] for r in rows_out])
        dp = statistics.median([r["rp"][key]["d_prec_pp"] for r in rows_out])
        print(f"    {name}：ΔRecall 中位 {dr:+.3f} pp ｜ ΔPrecision 中位 {dp:+.3f} pp")
    print()
    print("  ⚠️ 只做**配对差**，不做跨语义 U 横比（GMC 的 U 与 no-GMC 的 U 量纲不同）。")

    # ---- 配对检验（house rule §三·6c）----
    _pt = data.get("aggregate", {}).get("paired_tests")
    if _pt:
        print()
        print("  配对检验（Wilcoxon 符号秩，双侧；Δ = no-GMC − GMC）：")
        for key, name in (("t1_f1", "T1 F1"), ("t2_f1", "T2 F1"), ("t3_f1", "T3 F1")):
            t = _pt.get(key)
            if not t:
                continue
            print("    %-6s n=%-3d 中位差 %+.6f ｜ Hodges–Lehmann %+.6f ｜ W=%s p=%.3g（%s）｜ 符号检验 p=%.3g（负 %d / 正 %d）" % (
                name, t["n"], t["median"], t["hodges_lehmann"],
                ("%.1f" % t["w"]) if t["w"] is not None else "—",
                t["p"], t["method"], t["sign_p"], t["n_neg"], t["n_pos"]))

    # ---- lr 分档：源域代价 ↔ 难例增益（两侧同轴）----
    try:
        bands = lr_bands(g_path, n_path, gmc_base(), nogmc_base(args.ref), args.max_trials)
        print()
        print("  lr_scale 分档（Δ 均相对**各自**的冻结基线 ⇒ 两侧可同轴比）：")
        print("    %-8s | %-34s | %-34s" % ("lr 档", "GMC（n / ΔT1 / ΔT3）", "no-GMC（n / ΔT1 / ΔT3）"))
        for b in bands:
            def _fmt(x):
                if not x:
                    return "—"
                return "n=%-2d %+.3f pp / %+.3f pp" % (x["n"], x["d_t1_median_pp"], x["d_t3_median_pp"])
            print("    %-8s | %-34s | %-34s" % (b["band"], _fmt(b["gmc"]), _fmt(b["nogmc"])))
    except Exception as exc:
        print(f"  （lr 分档生成失败，已跳过：{exc}）")

    if args.out_json:
        out = {
            "matched": matched,
            "mismatched": mismatched,
            "missing": missing,
            "rows": rows_out,
            "aggregate": data["aggregate"],
        }
        Path(args.out_json).write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [OK] 已写出 {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
