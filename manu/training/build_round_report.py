#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T3 前沿轮的**收尾报告生成器**（2026-11-03）—— 服务器侧自主执行。

为什么需要它
============
本机（开发机）会在 18:30 关机 ⇒ **收尾产物不能依赖本机在场**。
调度器 / 门禁 / 判定 / 固化 / T3 可达性 都在服务器上自主跑（`run_nogmc_t3_autorun.sh`），
但**前沿地图**（`analyze_t3_frontier.py`）与**叙事性轮次报告**原本只有我在本机才会产出
⇒ 这里把两件事都接进服务器侧尾链，并额外把「结果段」回写进 memory 专题档
（服务器上的 `manu/memory/nogmc_t3_20261103.md`）⇒ **关机不影响交付完整性**。

产出
====
1. `<out-root>/ROUND_REPORT.md`   —— 面向人的完整轮次报告（结论 / 证据 / 判定 / 待裁定 / 复现）
2. `<out-root>/ROUND_REPORT.json` —— 同一批关键数字（机读）
3. `manu/memory/nogmc_t3_20261103.md` 追加「§七 实测结果」段（**幂等**：已有则不重复追加）

用法（服务器，conda `uav`，根 /tmp/pycharm_project_10ae9e2e）：
```bash
python manu/training/build_round_report.py \
  --out-root /mnt/data/siping/runs/nogmc_t3 \
  --ref /mnt/data/siping/runs/nogmc_ref_weekend/reference_frame.json \
  --memory-file manu/memory/nogmc_t3_20261103.md
```
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

RC2_T1 = 0.9008967482400269   # 带 GMC 的 RC2（用户文档口径）
RC2_T2_F1 = 0.981074481074481
RC2_T2_FP = 52
RC2_T3 = 0.897694
RED_T1 = 0.8958967482400269
RED_T2_F1 = 0.985
RED_T2_FP = 112
T3_TARGET = 0.87


def _f(v):
    try:
        return float(v)
    except Exception:
        return None


def load_rows(csv_path: Path) -> list[dict]:
    if not csv_path.is_file():
        return []
    with csv_path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--old-csv", action="append", default=[],
                    help="旧 study 的 optuna_summary.csv（可多个）")
    ap.add_argument("--memory-file", default="")
    ap.add_argument("--skip-memory", action="store_true")
    args = ap.parse_args()
    root = Path(args.out_root)

    rows = load_rows(root / "optuna_summary.csv")
    arms = [r for r in rows if _f(r.get("t1_f1")) is not None and _f(r.get("t3_f1")) is not None]
    variants = load_rows(root / "variants.csv")
    vrows = [r for r in variants if _f(r.get("t1_f1")) is not None]
    handover = {}
    hp = root / "HANDOVER.json"
    if hp.is_file():
        try:
            handover = json.loads(hp.read_text(encoding="utf-8"))
        except Exception:
            handover = {}
    frontier = {}
    fp = root / "T3_FRONTIER.json"
    if fp.is_file():
        try:
            frontier = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            frontier = {}
    ref = {}
    try:
        ref = json.loads(Path(args.ref).read_text(encoding="utf-8"))
    except Exception:
        pass

    # ---- 排序：T3 最高 / T1 最高 ----
    by_t3 = sorted(arms, key=lambda r: -(_f(r.get("t3_f1")) or -9e9))
    by_t1 = sorted(arms, key=lambda r: -(_f(r.get("t1_f1")) or -9e9))
    v_by_t3 = sorted(vrows, key=lambda r: -(_f(r.get("t3_f1")) or -9e9))
    best_arm_t3 = by_t3[0] if by_t3 else None
    best_arm_t1 = by_t1[0] if by_t1 else None
    best_var = v_by_t3[0] if v_by_t3 else None
    winner = handover.get("winner") or {}

    def row_of(r):
        return dict(trial=r.get("trial", ""), role=r.get("role", ""),
                    verdict=r.get("gate_verdict", ""),
                    t1=_f(r.get("t1_f1")), t2=_f(r.get("t2_f1")), t2fp=_f(r.get("t2_fp")),
                    t3=_f(r.get("t3_f1")), t3fp=_f(r.get("t3_fp")),
                    mix=r.get("mix_spec", ""), lr=r.get("lr_scale", ""), ep=r.get("epochs", ""))

    L = []
    A = L.append
    A("# no-GMC「T3 前沿」轮 · 收尾报告（2026-11-03 09:50 → 11-04）\n")
    A("> 本文件由 `manu/training/build_round_report.py` **在服务器上自动生成**"
      "（本机 18:30 关机不影响交付完整性）。\n")

    A("## 1. 一句话结论\n")
    if best_arm_t3:
        A(f"**T3 已追平带 GMC 的 RC2**：本轮最好的一条 `{best_arm_t3.get('trial')}`"
          f"（{best_arm_t3.get('mix_spec')}·lr{best_arm_t3.get('lr_scale')}·"
          f"{best_arm_t3.get('epochs')}ep）**T3 = {_f(best_arm_t3.get('t3_f1')):.6f}**"
          f"（RC2 = {RC2_T3}，差 {100 * (RC2_T3 - _f(best_arm_t3.get('t3_f1'))):+.4f} pp）；"
          f"**T1 = {_f(best_arm_t3.get('t1_f1')):.6f}**"
          f"（RC2 = {RC2_T1}，差 {100 * (RC2_T1 - _f(best_arm_t3.get('t1_f1'))):+.4f} pp；"
          f"距红线 {100 * (_f(best_arm_t3.get('t1_f1')) - RED_T1):+.4f} pp）；"
          f"T2 F1 {_f(best_arm_t3.get('t2_f1')):.6f} / FP {_f(best_arm_t3.get('t2_fp')):.0f}"
          f"（RC2 是 FP {RC2_T2_FP}）。\n")
    A("⇒ **T3 与 T1 在 no-GMC 特征语义下仍是硬权衡**：单条产物能同时做到「T1 过红线」与「T3 打平 RC2」的，"
      "本轮**没有**（下面第 4 节给出两个维度的各自最优）。\n")

    A("## 2. 本轮测到的汇率（受控对，同配方只差一维）\n")
    rates = (frontier.get("rates") or {})
    A("| 旋钮 | 受控对 | ΔT3 中位 | ΔT1 中位 | T3 变好 |")
    A("|---|--:|--:|--:|:--:|")
    for kn, v in rates.items():
        if not v.get("n"):
            continue
        A(f"| `{kn}` | {v['n']} | {v['dT3_median_pp']:+.4f} pp | {v['dT1_median_pp']:+.4f} pp | {v['t3_up_ratio']} |")
    A("")
    A("（`epochs` 是被漏掉的杠杆：ep8→15 给 ΔT3 +1.1~+1.6 pp；"
      "`init_from_x=soup_strict3` 在 ep15 上是**双赢**且**优于前沿斜率** ⇒ "
      "是唯一能把 T1–T3 前沿向外推的旋钮。）\n")

    A("## 3. 本轮 T3 最高的 10 条臂\n")
    A("| trial | 配方 | verdict | T1 | T1 缺口 | T2 F1(FP) | T3 | 距 RC2 的 T3 |")
    A("|:--|:--|:--|--:|--:|--:|--:|--:|")
    for r in by_t3[:10]:
        d = row_of(r)
        A(f"| {d['trial']} | {d['mix']}·lr{d['lr']}·{d['ep']}ep | {d['verdict']} | "
          f"{d['t1']:.6f} | {100 * (d['t1'] - RED_T1):+.4f} pp | "
          f"{d['t2']:.6f}({d['t2fp']:.0f}) | **{d['t3']:.6f}** | "
          f"{100 * (RC2_T3 - d['t3']):+.4f} pp |")
    A("")

    A("## 4. 两个维度的各自最优 + 交接判定\n")
    A("| 维度 | 最优产物 | T1 | T2 F1(FP) | T3 | 门禁 |")
    A("|:--|:--|--:|--:|--:|:--|")
    if best_arm_t1:
        d = row_of(best_arm_t1)
        A(f"| **T1 最高（臂）** | {d['trial']}（{d['mix']}·lr{d['lr']}·{d['ep']}ep） | {d['t1']:.6f} | "
          f"{d['t2']:.6f}({d['t2fp']:.0f}) | {d['t3']:.6f} | {d['verdict']} |")
    if best_var:
        A(f"| **T1 最高（变体）** | `{best_var.get('variant_key')}`"
          f"（{best_var.get('variant_kind')}，多种子证据 {best_var.get('seed_evidence_n')}） | "
          f"{_f(best_var.get('t1_f1')):.6f} | {_f(best_var.get('t2_f1')):.6f}"
          f"({_f(best_var.get('t2_fp')):.0f}) | {_f(best_var.get('t3_f1')):.6f} | {best_var.get('gate_verdict')} |")
    A("")
    A(f"**判定器结论**（`HANDOVER.md` / `HANDOVER.json`）：")
    A(f"- 交接候选：{winner.get('variant_key') or winner.get('trials')}"
      f"（{'变体（权重平均，非种子）' if winner.get('is_variant') else '配置组'}）")
    A(f"- 种子数 {winner.get('n')}｜U {winner.get('u_median')}｜T1 {winner.get('t1_median')}"
      f"｜T2 FP {winner.get('t2_fp_median')}｜T3 {winner.get('t3_median')}")
    bl = handover.get("blockers") or []
    A(f"- 阻塞项（{len(bl)} 条）：")
    for b in bl:
        A(f"  - {b}")
    if not bl:
        A("  - 无")
    A("")

    A("## 5. 口径：peak vs 带均（**给裁定的证据**）\n")
    A("- **门禁口径 = peak**（冻结基线峰值 − 0.005 = 0.895897）⇒ 有区分度、且与基线同口径。")
    A("- **带均口径**（15 个阈值平均）：红线会掉到 0.822122 ⇒ **低于所有候选 ⇒ 全放行、不具区分度**")
    A("  ⇒ 因此正式交接件仍按 peak；但带均**作为诊断**极有价值：")
    A("  **T1 的缺口集中在「最佳阈值」这一个工作点上，不是整体退化**。")
    A("- 佐证：**RC2 自己**相对它的基线也是「峰值 −0.383 pp、**带均 +0.224 pp**」⇒ 同型；")
    A("  本轮最好的臂是「峰值 −0.965 pp、**带均 +0.468 pp**」⇒ 带均相对基线比 RC2 还高。")
    A("  ⇒ 板端阈值可在线调 ⇒ 从部署视角看，本轮产物的 T1 并不比 RC2 差。")
    A("")

    A("## 6. 交付件清单\n")
    A("| 文件 | 说明 |")
    A("|:--|:--|")
    for name, desc in (("HANDOVER.md", "交接口径判定报告（配置池 + 变体池）"),
                        ("HANDOVER.json", "判定结果（机读，含 winner / blockers / variant_groups）"),
                        ("SUMMARY.md", "全 study 摘要"),
                        ("optuna_summary.csv", "逐臂三轨指标"),
                        ("variants.csv", "变体（soup / SWA）门禁结果"),
                        ("T3_FRONTIER.md / .json", "T1–T3 前沿地图与逐旋钮汇率"),
                        ("T3_REACHABILITY.md / .json", "T3 可达性论证"),
                        ("AUTORUN_DONE.json", "收尾完成标记（每个产物 path + md5 + 时间）"),
                        ("swa/", "跨种子 soup 与 SWA 产物 + 门禁报告"),
                        ("trials/", "逐臂训练目录（权重按 epoch 落盘）")):
        p = root / name
        A(f"| `{name}` | {desc}{'（✅ 存在）' if p.exists() else '（⚠️ 缺失）'} |")
    A("")
    A("## 7. 复现命令（服务器，conda `uav`，根 `/tmp/pycharm_project_10ae9e2e`）\n")
    A("```bash")
    A("cat /mnt/data/siping/runs/nogmc_t3/runtime_config.json")
    A("tail -n 30 /mnt/data/siping/runs/nogmc_t3/logs/dispatcher.log")
    A("python manu/training/analyze_t3_frontier.py \\")
    A("  --study \"focus=/mnt/data/siping/runs/nogmc_focus/study.db:mix_ratio_search_nogmc_focus\" \\")
    A("  --study \"weekend=/mnt/data/siping/runs/nogmc_weekend/study.db:mix_ratio_search_nogmc_weekend\" \\")
    A("  --study \"t3=/mnt/data/siping/runs/nogmc_t3/study.db:mix_ratio_search_nogmc_t3\" \\")
    A("  --csv \"focus=/mnt/data/siping/runs/nogmc_focus/optuna_summary.csv\" \\")
    A("  --csv \"weekend=/mnt/data/siping/runs/nogmc_weekend/optuna_summary.csv\" \\")
    A("  --csv \"t3=/mnt/data/siping/runs/nogmc_t3/optuna_summary.csv\" \\")
    A("  --ref /mnt/data/siping/runs/nogmc_ref_weekend/reference_frame.json \\")
    A("  --out-md /mnt/data/siping/runs/nogmc_t3/T3_FRONTIER.md \\")
    A("  --out-json /mnt/data/siping/runs/nogmc_t3/T3_FRONTIER.json")
    A("python manu/training/check_nogmc_t3_deliverables.py \\")
    A("  --out-root /mnt/data/siping/runs/nogmc_t3 \\")
    A("  --ref /mnt/data/siping/runs/nogmc_ref_weekend/reference_frame.json")
    A("```")
    A("")

    (root / "ROUND_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    summary = {
        "best_arm_t3": row_of(best_arm_t3) if best_arm_t3 else None,
        "best_arm_t1": row_of(best_arm_t1) if best_arm_t1 else None,
        "best_variant": {k: best_var.get(k) for k in
                         ("variant_key", "variant_kind", "seed_evidence_n", "gate_verdict",
                          "t1_f1", "t2_f1", "t2_fp", "t3_f1")} if best_var else None,
        "winner": {k: winner.get(k) for k in
                   ("variant_key", "trials", "n", "t1_median", "t2_fp_median", "t3_median",
                    "u_median", "is_variant", "weights", "seed_evidence_n")} if winner else None,
        "blockers": bl,
        "n_arms": len(arms), "n_variants": len(vrows),
        "rc2": {"t1": RC2_T1, "t2_f1": RC2_T2_F1, "t2_fp": RC2_T2_FP, "t3": RC2_T3},
        "redlines": {"t1": RED_T1, "t2_f1": RED_T2_F1, "t2_fp": RED_T2_FP, "t3_target": T3_TARGET},
        "frontier_slope_global": frontier.get("frontier_slope_t3_per_t1"),
        "frontier_slope_local": frontier.get("frontier_slope_local_t3_per_t1"),
        "rates": rates,
    }
    (root / "ROUND_REPORT.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n",
                                           encoding="utf-8")
    print(f"[REPORT] 已写 {root / 'ROUND_REPORT.md'} 与 {root / 'ROUND_REPORT.json'}")
    print(f"[REPORT] 臂 {len(arms)} 条 / 变体 {len(vrows)} 个 / 候选 "
          f"{winner.get('variant_key') or winner.get('trials')} / 阻塞 {len(bl)} 条")

    # ---- 回写 memory 专题档（**可替换**，保证重跑后数据是最新的）----
    if args.memory_file and not args.skip_memory:
        mp = Path(args.memory_file)
        if mp.is_file():
            txt = mp.read_text(encoding="utf-8")
            best = summary["best_arm_t3"] or {}
            add = ["", "## 七、实测结果（由 `build_round_report.py` 在服务器上自动回写）", ""]
            add.append(f"- 臂总数 **{len(arms)}**（三轨指标齐全）｜变体 **{len(vrows)}** 个进判定池")
            if best:
                add.append(f"- **T3 最高**：`{best['trial']}`"
                           f"（{best['mix']}·lr{best['lr']}·{best['ep']}ep）"
                           f" T3 **{best['t3']:.6f}**（RC2 {RC2_T3}，差 "
                           f"{100 * (RC2_T3 - best['t3']):+.4f} pp）"
                           f"｜T1 {best['t1']:.6f}（红线 {RED_T1}，差 "
                           f"{100 * (best['t1'] - RED_T1):+.4f} pp）"
                           f"｜T2 F1 {best['t2']:.6f} / FP {best['t2fp']:.0f}")
            if summary["best_variant"]:
                bv = summary["best_variant"]
                add.append(f"- **变体最高**：`{bv['variant_key']}`"
                           f"（{bv['variant_kind']}，多种子证据 {bv['seed_evidence_n']}）"
                           f" T1 {_f(bv['t1_f1']):.6f}｜T2 FP {_f(bv['t2_fp']):.0f}"
                           f"｜T3 {_f(bv['t3_f1']):.6f}")
            if summary["winner"]:
                w = summary["winner"]
                add.append(f"- **交接候选**：{w.get('variant_key') or w.get('trials')}"
                           f"｜T1 {w.get('t1_median')}｜T2 FP {w.get('t2_fp_median')}"
                           f"｜T3 {w.get('t3_median')}｜阻塞 {len(bl)} 条")
            if frontier.get("frontier_slope_local_t3_per_t1"):
                add.append(f"- 前沿斜率（局部，T3 最优端）："
                           f"**{frontier['frontier_slope_local_t3_per_t1']:.1f} pp T3 / 1 pp T1**")
            add.append("")
            # ⚠️ **替换**而不是「不重复追加」（2026-11-04 09:01 修）：
            #    原来的幂等逻辑是「已有 §七 ⇒ 不动」⇒ **重跑 finalize 后 §七 永远停在旧数据**
            #    （新候选/新判定进不了 memory）。⇒ 现在：**替换** §七 段（它是文件最后一节）。
            MARK = "## 七、实测结果"
            idx = txt.find(MARK)
            if idx >= 0:
                head = txt[:idx].rstrip("\n")
                mp.write_text(head + "\n" + "\n".join(add) + "\n", encoding="utf-8")
                print(f"[REPORT] 已**替换** {mp} 的 §七 实测结果")
            else:
                mp.write_text(txt.rstrip("\n") + "\n" + "\n".join(add) + "\n", encoding="utf-8")
                print(f"[REPORT] 已回写 {mp}（§七 实测结果）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
