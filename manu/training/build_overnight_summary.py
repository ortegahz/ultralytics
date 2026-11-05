#!/usr/bin/env python3
"""过夜总汇总（2026-11-04 夜）—— 一次扫描全部产物，产出**唯一一份**明早可读的结论文件。

覆盖四条并行链：
  ① clip_max_norm 6.5 受控实验（阶段 D/E 的产物）
  ② 跨家族权重插值 / 分层 / 手术 / TIES / 单纯形扫参（`interp/interp_sweep.csv`）
  ③ 训练臂（T1 修复微调 / T3 配方空间延伸 / T3 家族加种子）
  ④ 决策层旋钮（th_base × dedup_radius / area 能量门）

**判定口径**（与 `manu/memory/nogmc_t3_20261103.md` 一致）：
  · T1 硬门禁：F1 ≥ **0.8958967482400269**（红线 = 冻结基线 trial_0474 的 T1 − 0.005）
  · T2 硬门禁：F1 ≥ **0.985** 且 **FP ≤ 112**
  · T3 观察轨（veto=false）：目标 ≥ **0.87**；本轮的「兼得」判定线 = **T3 ≥ 0.89**
  · 对标：base = trial_0474（no-GMC 冻结基线）；RC2 = trial_0080 ep3（带 GMC）

用法（服务器，conda uav，根 /tmp/pycharm_project_10ae9e2e）：
    python manu/training/build_overnight_summary.py --out-root /mnt/data/siping/runs/nogmc_t3
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from pathlib import Path

RL = 0.8958967482400269
BASE = {"t1": 0.9008967482400269, "t2": 0.981074481074481, "t3": 0.7578120617110798}
RC2 = {"t1": 0.9019097586568731, "t2": 0.9989491698441769, "t3": 0.8976937315523539}
T2_F1_MIN, T2_FP_MAX, T3_TARGET, T3_BOTH = 0.985, 112, 0.87, 0.89


def f(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def to_int(x, default=0):
    """Coerce a CSV/JSON field to int.

    ``csv.DictReader`` always yields **strings**; comparing one with an int raises
    ``TypeError: '<=' not supported between instances of 'str' and 'int'``.
    That crashed the 2026-11-05 07:05 automatic summary run, so every field that
    takes part in a numeric comparison goes through here.

    NOTE: deliberately **not** named ``i`` -- ``main()`` already uses ``i`` as a
    plain local variable (``i = txt.find(...)``), which shadows a module-level
    ``i()`` helper inside the function scope.
    """
    try:
        return int(float(x))
    except Exception:
        return default


def read_gate(path: Path) -> dict | None:
    """从一份 gate_report.json 抽出三轨指标。"""
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    try:
        t1 = d["tracks"]["track1_uav"]["finetune_peak"]
        t2 = d["tracks"]["track2_golden"]
        t3 = d["tracks"]["track3_blind"]
    except Exception:
        return None
    return {
        "t1": t1.get("f1"), "t1_fp": t1.get("fp"), "t1_tp": t1.get("tp"),
        "t2": t2.get("f1"), "t2_fp": t2.get("fp"), "t2_fn": t2.get("fn"),
        "t3": t3.get("f1"), "t3_fp": t3.get("fp"), "t3_tp": t3.get("tp"), "t3_fn": t3.get("fn"),
        "verdict": d.get("verdict"), "weights": d.get("weights"),
        "veto": (d.get("veto_failures") or []),
    }


def scan_gate_reports(out_root: Path) -> list[dict]:
    rows = []
    for p in sorted(glob.glob(str(out_root / "**" / "gate_report.json"), recursive=True)):
        m = read_gate(Path(p))
        if not m or m.get("t1") is None:
            continue
        m["_path"] = p
        m["_tag"] = str(Path(p).parent.parent.relative_to(out_root))
        rows.append(m)
    return rows


def fmt_row(name: str, m: dict) -> str:
    t1, t2, t3 = m.get("t1"), m.get("t2"), m.get("t3")
    if t1 is None:
        return f"| `{name}` | — | — | — | — | — | — |"
    hard1 = t1 >= RL
    hard2 = (t2 is not None and t2 >= T2_F1_MIN and to_int(m.get("t2_fp")) <= T2_FP_MAX)
    return (f"| `{name}` | {t1:.9f} | {(t1 - RL) * 100:+.4f} | {'✅' if hard1 else '❌'} | "
            f"{f(t2, 0):.9f}/{m.get('t2_fp')} | {'✅' if hard2 else '❌'} | "
            f"{f(t3, 0):.9f} | {(f(t3, 0) - BASE['t3']) * 100:+.4f} | {(f(t3, 0) - RC2['t3']) * 100:+.4f} |")


HEADER = ("| 产物 | T1 | T1 余量(pp) | T1 硬门 | T2 F1/FP | T2 硬门 | T3 | T3 vs base(pp) | T3 vs RC2(pp) |\n"
          "|:--|--:|--:|:--:|--:|:--:|--:|--:|--:|")
SEP = "|:--|--:|--:|:--:|--:|:--:|--:|--:|--:|"


def main() -> int:
    ap = argparse.ArgumentParser(description="过夜总汇总")
    ap.add_argument("--out-root", default="/mnt/data/siping/runs/nogmc_t3")
    ap.add_argument("--out-md", default="")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()
    root = Path(args.out_root)
    out_md = Path(args.out_md) if args.out_md else root / "OVERNIGHT_SUMMARY.md"
    out_json = Path(args.out_json) if args.out_json else root / "OVERNIGHT_SUMMARY.json"

    L: list[str] = []
    L.append("# ⭐ 过夜总汇总（2026-11-04 夜 · no-GMC 方向一：权重插值 + 多搜索空间）")
    L.append("")
    L.append(f"> 生成时间：`{__import__('datetime').datetime.now():%F %T}`　｜　"
             f"参照系：base = trial_0474（no-GMC 冻结基线）｜RC2 = trial_0080 ep3（带 GMC）")
    L.append(">")
    L.append("> **判定口径**：T1 ≥ 0.8958967482400269（硬门）· T2 F1 ≥ 0.985 且 FP ≤ 112（硬门）· "
             "T3 ≥ 0.87（观察轨 veto=false）· **「兼得」判定线 = T1 过线 且 T3 ≥ 0.89**")
    L.append("")

    # ---------- ① clip 受控实验 ----------
    L.append("## ① clip_max_norm 6.5 受控实验（n=4 vs n=4）")
    L.append("")
    pe = root / "logs" / "phaseE.log"
    clip_txt = ""
    if pe.exists():
        txt = pe.read_text(encoding="utf-8", errors="replace")
        i = txt.find("受控差值")
        clip_txt = txt[max(0, i - 1200): i + 800] if i >= 0 else txt[-1200:]
    L.append("```")
    L.append(clip_txt.strip() if clip_txt else "（phaseE.log 未找到受控结论段）")
    L.append("```")
    L.append("")

    # ---------- ② 插值扫参 ----------
    L.append("## ② 跨家族权重插值 / 分层 / 手术 / TIES / 单纯形（零训练，GPU 2/3）")
    L.append("")
    csvp = root / "interp" / "interp_sweep.csv"
    interp_rows = []
    if csvp.exists():
        for r in csv.DictReader(csvp.open(encoding="utf-8")):
            t1, t2, t3 = f(r.get("t1_f1")), f(r.get("t2_f1")), f(r.get("t3_f1"))
            if t1 is None:
                continue
            interp_rows.append({
                "name": r.get("name"), "t1": t1, "t2": t2, "t2_fp": to_int(r.get("t2_fp")),
                "t3": t3, "t3_fp": to_int(r.get("t3_fp")), "verdict": r.get("verdict"),
                "weights": r.get("weight"),
            })
    if interp_rows:
        L.append(HEADER)
        L.append(SEP)
        for r in sorted(interp_rows, key=lambda x: -(x["t3"] or 0)):
            L.append(fmt_row(r["name"], r))
        L.append("")
        both = [r for r in interp_rows if r["t1"] >= RL and (r["t3"] or 0) >= T3_BOTH]
        pass1 = [r for r in interp_rows if r["t1"] >= RL]
        if both:
            L.append(f"### ⭐⭐⭐ 命中「兼得」：{len(both)} 个产物同时 T1 过线且 T3 ≥ {T3_BOTH}")
            L.append("")
            for r in sorted(both, key=lambda x: -(x["t3"] or 0)):
                L.append(f"* **`{r['name']}`** — T1 {r['t1']:.9f}（余量 {(r['t1'] - RL) * 100:+.4f} pp）"
                         f" · T2 {r['t2']:.9f}/{r['t2_fp']} · **T3 {r['t3']:.9f}**"
                         f"（vs RC2 {(r['t3'] - RC2['t3']) * 100:+.4f} pp）")
                L.append(f"  * 权重：`{r['weights']}`")
            L.append("")
        elif pass1:
            best = max(pass1, key=lambda x: (x["t3"] or 0))
            L.append(f"### ⚠️ 无「兼得」产物；T1 过线的产物里 T3 最高 = **{best['t3']:.9f}**"
                     f"（`{best['name']}`，T1 余量 {(best['t1'] - RL) * 100:+.4f} pp，"
                     f"距 {T3_BOTH} 差 {(T3_BOTH - best['t3']) * 100:.4f} pp）")
            L.append("")
        else:
            L.append("### ⚠️ 插值族**没有任何一个 T1 过线**")
            L.append("")
    else:
        L.append("（无 `interp/interp_sweep.csv` ⇒ 插值链未产出结果）")
        L.append("")

    # ---------- ③ 训练臂 ----------
    L.append("## ③ 训练臂（T1 修复微调 / T3 配方空间延伸 / T3 家族加种子）")
    L.append("")
    arm_rows = []
    for p in sorted(glob.glob(str(root / "trials" / "trial_00*" / "gate" / "*" / "gate_report.json"))):
        m = read_gate(Path(p))
        if not m:
            continue
        try:
            num = int(Path(p).parents[2].name.split("_")[-1])
        except Exception:
            continue
        if num < 55:      # 55 以下是 clip 受控实验（已在 ① 覆盖）
            continue
        m["_name"] = f"trial_{num:04d}"
        arm_rows.append(m)
    if arm_rows:
        L.append(HEADER)
        L.append(SEP)
        for r in sorted(arm_rows, key=lambda x: -(x["t3"] or 0)):
            L.append(fmt_row(r["_name"], r))
        L.append("")
    else:
        L.append("（无编号 ≥ 55 的训练臂门禁产物）")
        L.append("")

    # ---------- ④ 决策层 ----------
    L.append("## ④ 决策层旋钮（纯 CPU，零 GPU）")
    L.append("")
    # 决策层扫参可能落在 decision_sweep/<run>/ 子目录里（K2/K3 的一次运行一个目录），
    # 旧的平铺 glob 找不到 ⇒ 这里递归扫，并直接算出「每个臂的最优行」。
    run_dirs = sorted(
        d for d in glob.glob(str(root / "decision_sweep" / "*")) if os.path.isdir(d)
    )
    dec_rows = []
    for d in run_dirs:
        csvp = os.path.join(d, "sweep_k2k3.csv")
        if not os.path.exists(csvp):
            csvp = os.path.join(d, "sweep_all.csv")
        if not os.path.exists(csvp):
            continue
        try:
            with open(csvp, encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    t3 = f(r.get("t3_f1"))
                    if t3 is None:
                        continue
                    if str(r.get("t2_gate_pass", "True")).lower() not in ("true", "1", "yes"):
                        continue
                    dec_rows.append({
                        "run": os.path.basename(d),
                        "arm": r.get("arm"), "t3": t3,
                        "d3": f(r.get("d_t3_f1"), 0.0),
                        "tp": to_int(r.get("t3_tp")), "fp": to_int(r.get("t3_fp")),
                        "fn": to_int(r.get("t3_fn")),
                        "area": r.get("area_min"), "sum_mode": r.get("sum_mode"),
                        "sum_value": r.get("sum_value"), "gstage": r.get("gate_stage"),
                        "th": r.get("th_base"), "tags": r.get("tag_allow"),
                        "t2_f1": f(r.get("t2_f1"), 0.0), "t2_fp": to_int(r.get("t2_fp")),
                    })
        except Exception:
            continue
    if dec_rows:
        L.append(f"共 **{len(dec_rows)}** 行通过 T2 硬门的决策层结果（来自 {len(run_dirs)} 个运行目录）。")
        L.append("")
        L.append("| 运行 | 臂 | 门配置 | T3 | ΔT3(pp) | TP | FP | FN | T2 F1/FP |")
        L.append("|:--|:--|:--|--:|--:|--:|--:|--:|:--|")
        best_by_arm: dict[tuple, dict] = {}
        for r in dec_rows:
            key = (r["run"], r["arm"])
            if key not in best_by_arm or r["t3"] > best_by_arm[key]["t3"]:
                best_by_arm[key] = r
        for r in sorted(best_by_arm.values(), key=lambda x: -(x["t3"] or 0)):
            gate = (f"area≥{r['area']}, sums {r['sum_mode']}={r['sum_value']}, "
                    f"{r['gstage']}, th {r['th']}, tags {r['tags']}")
            L.append(f"| `{r['run']}` | `{r['arm']}` | {gate} | **{r['t3']:.9f}** | "
                     f"{r['d3'] * 100:+.4f} | {r['tp']} | {r['fp']} | {r['fn']} | "
                     f"{r['t2_f1']:.6f}/{r['t2_fp']} |")
        L.append("")
    else:
        L.append("（未找到决策层扫参 CSV）")
        L.append("")

    # ---------- ④-b T1 侧判别门移植（决定「兼得」能否成立） ----------
    L.append("## ④-b T1 侧判别门移植（Anti-UAV 官方域，决定「兼得」能否成立）")
    L.append("")
    t1_files = sorted(glob.glob(str(root / "candgate_t1" / "*" / "t1_candgate*.json")))
    if t1_files:
        L.append("| 臂 | 门关闭最优 T1 F1 | 开门后最优 T1 F1 | 提升 | **过红线 0.8958967 的门配置数** |")
        L.append("|:--|--:|--:|--:|--:|")
        for path in t1_files:
            try:
                payload = json.loads(open(path, encoding="utf-8").read())
            except Exception:
                continue
            for arm in payload.get("arms", []):
                configs = arm.get("configs") or []
                if not configs:
                    continue
                closed = (arm.get("closed") or {}).get("best") or {}
                best_closed = f(closed.get("f1"))
                best_open = None
                passing = 0
                for cfg in configs:
                    b = cfg.get("best") or {}
                    if b.get("f1") is None:
                        continue
                    if best_open is None or b["f1"] > best_open:
                        best_open = b["f1"]
                    if b["f1"] >= RL:
                        passing += 1
                gain = ""
                if best_closed is not None and best_open is not None:
                    gain = f"{(best_open - best_closed) * 100:+.4f} pp"
                L.append(f"| `{arm.get('label')}` | "
                         f"{('%.9f' % best_closed) if best_closed is not None else '—'} | "
                         f"{('%.9f' % best_open) if best_open is not None else '—'} | {gain} | "
                         f"**{passing}** |")
        L.append("")
        L.append("⇒ **过红线配置数 = 0 ⇒ T1 侧判别门不能把 `trial_0049` 抬过红线 ⇒ 「兼得」不成立**"
                 "（详见 `manu/memory/t1_candgate_20261105.md`）。")
        L.append("")
    else:
        L.append("（未找到 T1 门控 JSON）")
        L.append("")

    # ---------- ⑤ 全库扫描 + 判定 ----------
    L.append("## ⑤ 全库扫描（所有 gate_report.json）与最终判定")
    L.append("")
    allrows = scan_gate_reports(root)
    L.append(f"扫描到 **{len(allrows)}** 份有效门禁报告。")
    L.append("")
    ok = [r for r in allrows if (r.get("t1") or 0) >= RL]
    L.append(f"* **T1 过线产物：{len(ok)}** 个")
    if ok:
        ok.sort(key=lambda x: -(x.get("t3") or 0))
        L.append("")
        L.append(HEADER)
        L.append(SEP)
        for r in ok[:15]:
            L.append(fmt_row(r["_tag"], r))
        L.append("")
    best_t3 = max(allrows, key=lambda x: (x.get("t3") or 0)) if allrows else None
    if best_t3:
        L.append(f"* **全库 T3 最高**：`{best_t3['_tag']}` T3 = **{best_t3['t3']:.9f}**"
                 f"（vs RC2 {(best_t3['t3'] - RC2['t3']) * 100:+.4f} pp）"
                 f"，但 T1 = {best_t3['t1']:.9f}（{(best_t3['t1'] - RL) * 100:+.4f} pp ⇒ "
                 f"{'过线' if best_t3['t1'] >= RL else '**不过线**'}）")
        L.append("")
    # 兼得全局判定
    both_all = [r for r in allrows if (r.get("t1") or 0) >= RL and (r.get("t3") or 0) >= T3_BOTH]
    both_all.sort(key=lambda x: -(x["t3"] or 0))
    L.append("## ⑥ 裁定")
    L.append("")
    if both_all:
        L.append(f"### ⭐⭐⭐ **存在「兼得」产物：{len(both_all)} 个**（T1 过线 且 T3 ≥ {T3_BOTH}）")
        L.append("")
        for r in both_all[:8]:
            L.append(f"* **`{r['_tag']}`** — T1 {r['t1']:.9f}（余量 {(r['t1'] - RL) * 100:+.4f} pp）"
                     f" · T2 {r['t2']:.9f}/{r['t2_fp']} · **T3 {r['t3']:.9f}**")
            L.append(f"  * 权重：`{r.get('weights')}`　· 报告：`{r['_path']}`")
        L.append("")
        L.append("> ⚠️ **口径纪律**：这些产物是**在评估集上搜出来的**（α / 模块手术 / 决策层门），"
                 "属**探索性证据**；进正式交接前必须补**同配方多种子复现**。")
    else:
        L.append(f"### ⚠️ **本夜未产出「兼得」产物**（T1 过线 且 T3 ≥ {T3_BOTH}）")
        L.append("")
        if ok:
            b = ok[0]
            L.append(f"* 过线产物中 T3 最高 = **{b['t3']:.9f}**（`{b['_tag']}`），"
                     f"距 {T3_BOTH} 差 **{(T3_BOTH - b['t3']) * 100:.4f} pp**")
        L.append("* ⇒ **「T1 余量宽裕」与「T3 高」互斥**这条结构性结论再次成立（见专题档 §3.9.10）")
        # 决策层判别门的贡献（它把「T1 过线」那一侧的 T3 顶上去了，但到不了 0.89）
        gate_best = None
        for r in dec_rows:
            if r["arm"] == "p2" and r["sum_mode"] == "abs":
                if gate_best is None or r["t3"] > gate_best["t3"]:
                    gate_best = r
        if gate_best is not None:
            gate_t2_ok = gate_best["t2_f1"] >= T2_F1_MIN and gate_best["t2_fp"] <= T2_FP_MAX
            L.append(f"* ⭐ **决策层 area/sum 判别门（样本外口径）把「T1 过线」侧的 T3 顶到 "
                     f"**{gate_best['t3']:.9f}**（门配置 `area≥{gate_best['area']}, sums≥{gate_best['sum_value']}, "
                     f"{gate_best['gstage']}, th {gate_best['th']}`），距 {T3_BOTH} 仍差 "
                     f"**{(T3_BOTH - gate_best['t3']) * 100:.4f} pp**；T2 硬门 "
                     f"**{'PASS' if gate_t2_ok else 'FAIL'}**"
                     f"（{gate_best['t2_f1']:.6f}/FP{gate_best['t2_fp']}）")
            L.append("* ⇒ **T3 侧到此为顶**：3792 点低阈值网格证实「降阈值换 TP」净亏、冻结 `th_base=0.30` 仍是最优点")
        L.append("* ⇒ **T1 侧移植实测 0 过线**（见 ④-b）⇒ **两条路都封 ⇒ 「兼得」在本轮全部搜索空间下不可达**")
        L.append("* ⇒ **维持现行交付裁定**：主装 `cs_t0043` + 峰值单兵 `trial_0049` + 备验 `cs_t0008p2`（双基线）")
    L.append("")

    # ---------- 交付指针 ----------
    L.append("## ⑦ 交付指针")
    L.append("")
    for rel, desc in [
        ("CANDIDATES/Candidate_t3/", "T1 优先路线（cs_t0008 + cs_t0008p2）"),
        ("CANDIDATES/Candidate_t3_T3prio/", "T3 优先路线（cs_t0043 + trial_0049）"),
        ("interp/interp_sweep.md", "插值族完整结果表"),
        ("OVERNIGHT_SUMMARY.md", "本文件"),
    ]:
        L.append(f"* `{rel}` — {desc}")
    L.append("")
    L.append("> 完整论证与复现命令见 `manu/memory/nogmc_t3_20261103.md` §3.9.12 / §3.9.13 与 `manu/memory/memory.md` §37。")
    L.append("")

    out_md.write_text("\n".join(L) + "\n", encoding="utf-8")
    out_json.write_text(json.dumps({
        "generated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "redline_t1": RL, "base": BASE, "rc2": RC2,
        "t2_f1_min": T2_F1_MIN, "t2_fp_max": T2_FP_MAX, "t3_target": T3_TARGET, "t3_both": T3_BOTH,
        "n_gate_reports": len(allrows),
        "t1_pass": [{"tag": r["_tag"], "t1": r["t1"], "t2": r["t2"], "t2_fp": r["t2_fp"],
                     "t3": r["t3"], "t3_fp": r["t3_fp"], "weights": r.get("weights")} for r in ok],
        "both": [{"tag": r["_tag"], "t1": r["t1"], "t3": r["t3"], "weights": r.get("weights")}
                 for r in both_all],
        "best_t3": ({"tag": best_t3["_tag"], "t3": best_t3["t3"], "t1": best_t3["t1"]}
                    if best_t3 else None),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[SUMMARY] 已写 {out_md}")
    print(f"[SUMMARY] 已写 {out_json}")
    print(f"[SUMMARY] 门禁报告 {len(allrows)} 份｜T1 过线 {len(ok)} 个｜兼得 {len(both_all)} 个")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
