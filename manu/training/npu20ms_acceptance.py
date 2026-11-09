#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
「NPU ≤ 21 ms 且精度下降微小」搜索的**验收合并表**。

把三样东西并成一张表，直接回答目标达成没达成：
  ① **板卡 NPU 实测时间**（从结构→速度表查；结构变了权重不变 ⇒ 时间只由结构决定）
  ② **三轨门禁指标**（T1/T2/T3，来自各 trial 的 gate_report.json）
  ③ **相对 cs_t0043 的精度变化**（cs_t0043 = 3 粒子同配方 Model Soup，当前交付基线）

判据：
  硬约束 —— 板卡 rknn_run ≤ --limit-ms（默认 21）
  主判据 —— ΔT3 ≥ --t3-drop-max-pp（默认 1.0 pp）
  次判据 —— ΔT1 ≥ --t1-drop-max-pp（默认 0.5 pp）且 T2 FP 增量 ≤ --t2-fp-max

用法（宿主或服务器）：
    python manu/training/npu20ms_acceptance.py \
        --root /mnt/data/siping/runs/npu20ms \
        --speed-table manu/pipeline/npu/struct_speed_table.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

# cs_t0043 的实测三轨（no-GMC 语义，3 粒子同配方 Model Soup）
CS_T0043 = {"t1_f1": 0.890766573, "t1_fp": 1210,
            "t2_f1": 0.998859, "t2_fp": 34,
            "t3_f1": 0.903960496, "t3_fp": 341}


def load_speed_table(path: str) -> dict:
    """(head_src, head_conv, act, late, scale) → ms"""
    p = Path(path)
    if not p.is_file():
        print(f"[WARN] 速度表不存在：{p} ⇒ 速度列留空")
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    out = {}
    for a in d.get("arms", []):
        sp = a.get("spec") or {}
        if a.get("ms") is None:
            continue
        key = (sp.get("head_src", "p1"), int(sp.get("head_conv", 64)),
               sp.get("act", "relu"), bool(sp.get("late_upsample", False)),
               round(float(sp.get("p2_out_scale", 1.0) or 1.0), 3))
        out[key] = float(a["ms"])
    print(f"[OK] 速度表 {len(out)} 个结构 <- {p}")
    return out


def struct_key_of(params: dict) -> tuple:
    return (params.get("head_src") or "p1", int(params.get("head_conv", 64) or 64),
            params.get("act", "relu"), bool(params.get("late_upsample", False)),
            round(float(params.get("p2_out_scale", 1.0) or 1.0), 3))


def read_gate_report(trial_dir: Path) -> dict:
    """gate_report.json 的结构：tracks.{track1_uav,track2_golden,track3_blind}"""
    cands = sorted(trial_dir.glob("gate*/epoch*/gate_report.json"))
    if not cands:
        return {}
    try:
        d = json.loads(cands[-1].read_text(encoding="utf-8"))
    except Exception:
        return {}
    tr = d.get("tracks", {})
    t1 = tr.get("track1_uav", {})
    t1f = t1.get("finetune_peak", {})
    t2 = tr.get("track2_golden", {})
    t3 = tr.get("track3_blind", {})
    out = {
        "t1_f1": t1f.get("f1"), "t1_fp": t1f.get("fp"),
        "t1_mean_delta": t1.get("delta", {}).get("mean_f1"),
        "t2_f1": t2.get("f1"), "t2_fp": t2.get("fp"),
        "t3_f1": t3.get("f1"), "t3_fp": t3.get("fp"),
        "verdict": d.get("verdict"),
        # ★ 2026-11-06：显式标出「评测没跑出来」的轨。
        #   （以前门禁会把三条轨全挂写成 verdict="PASS" + gates={} ⇒ 表格里是
        #    一行「没有指标」的杠杠，看不出是崩了还是没指标。）
        "infra_failed": bool(d.get("infrastructure_failure")) or any(
            isinstance(v, dict) and v.get("status") not in (None, "ok")
            for v in tr.values()),
        "failed_tracks": sorted(k for k, v in tr.items()
                                if isinstance(v, dict) and v.get("status") not in (None, "ok")),
    }
    return out


def load_soup_candidates(path: str) -> dict:
    """读 soup 候选清单 ⇒ {soup 名: {"struct":…, "speed_ms":…, "family":…, "members":…}}。

    ★ **可选**：文件不存在 ⇒ 返回空 dict ⇒ 既有行为逐位不变。
    """
    p = Path(path)
    if not p.is_file():
        print(f"[WARN] soup 候选清单不存在：{p} ⇒ 验收表不含 soup 行")
        return {}
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] soup 候选清单解析失败：{exc} ⇒ 验收表不含 soup 行")
        return {}
    out = {}
    for c in d.get("candidates", []):
        name = c.get("name")
        if not name:
            continue
        out[name] = c
    print(f"[OK] soup 候选 {len(out)} 个 <- {p}")
    return out


def render_markdown(payload: dict, crit: dict | None = None) -> str:
    """acceptance.json 的内容 ⇒ 人读总表（Markdown）。

    格式沿用看门狗 inline 生成器的既有人读版（保持交付物观感一致），
    额外：① soup 行加 🍲 标记；② **没有达标候选时**给出「最接近达标」的排序
    （按三条精度判据通过的条数，再按 T3 降序）⇒ 即使一条都不达标，
    也能一眼看出**最接近**「≤21 ms ∧ 精度下降微小」的那个。
    """
    crit = crit or {"t3_drop_max_pp": 1.0, "t1_drop_max_pp": 0.5, "t2_fp_max": 10}
    base = payload["baseline"]
    rows = payload["results"]
    rows_ok = [r for r in rows if str(r.get("verdict", "")).startswith("🎯")]
    rows_sp = [r for r in rows if r.get("ms") is not None]

    def _f(v, n=6):
        return ("%.*f" % (n, v)) if isinstance(v, float) else "-"

    out = ["# 「NPU ≤ 21 ms 且精度下降微小」最终验收表", "",
           "基线 = cs_t0043（3 粒子同配方 Model Soup）："
           f"T1 {base['t1_f1']:.6f} | T2 {base['t2_f1']:.6f} (FP {base['t2_fp']}) | "
           f"T3 {base['t3_f1']:.6f}", "",
           "| 臂 | NPU ms | T1 | ΔT1(pp) | T2 | ΔFP_T2 | T3 | ΔT3(pp) | T3FP | 判定 |",
           "|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|"]

    for r in sorted(rows, key=lambda x: (x.get("ms") is None, x.get("ms") or 0)):
        label = ("🍲 " + r["trial"]) if r.get("soup") else r["trial"]
        out.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            label, _f(r.get("ms"), 2), _f(r.get("t1")), _f(r.get("d1"), 3),
            _f(r.get("t2")),
            ("%+d" % r["dFP2"]) if r.get("dFP2") is not None else "-",
            _f(r.get("t3")), _f(r.get("d3"), 3),
            r.get("t3fp") if r.get("t3fp") is not None else "-",
            r.get("verdict", "-")))
    out.append("")

    if rows_ok:
        out.append("## 🏆 达标候选（按 T3 降序）")
        for r in sorted(rows_ok, key=lambda x: -(x.get("t3") or 0)):
            out.append("- `%s` NPU %.2f ms | T1 %.6f | T2 %.6f | T3 %.6f" % (
                r["trial"], r["ms"], r.get("t1") or 0, r.get("t2") or 0, r.get("t3") or 0))
        return "\n".join(out) + "\n"

    out.append("## （暂无同时满足「NPU ≤ %.1f ms ∧ ΔT3 ≥ −%.1f pp ∧ ΔT1 ≥ −%.1f pp "
               "∧ ΔFP_T2 ≤ +%d」的候选）" % (
                   crit.get("limit_ms", 21.0), crit.get("t3_drop_max_pp", 1.0),
                   crit.get("t1_drop_max_pp", 0.5), crit.get("t2_fp_max", 10)))

    # ---- 「最接近达标」：三条精度判据各过一条算一分 ----
    def _passed(r) -> int:
        n = 0
        d1, d3, dfp = r.get("d1"), r.get("d3"), r.get("dFP2")
        if isinstance(d3, (int, float)) and d3 >= -crit.get("t3_drop_max_pp", 1.0):
            n += 1
        if isinstance(d1, (int, float)) and d1 >= -crit.get("t1_drop_max_pp", 0.5):
            n += 1
        if isinstance(dfp, int) and dfp <= crit.get("t2_fp_max", 10):
            n += 1
        return n

    scored = [(r, _passed(r)) for r in rows if r.get("ms") is not None]
    best = [x for x in scored if x[1] >= 2]
    if best:
        best.sort(key=lambda x: (-x[1], -(x[0].get("t3") or 0)))
        out.append("### 最接近达标（3 条精度判据过了 2 条以上，按 T3 降序）")
        out.append("")
        out.append("| 臂 | NPU ms | T1 | ΔT1(pp) | T2 | ΔFP_T2 | T3 | ΔT3(pp) | 过几条 |")
        out.append("|:--|--:|--:|--:|--:|--:|--:|--:|--:|")
        for r, n in best[:12]:
            out.append("| %s | %.2f | %s | %s | %s | %s | %s | %s | %d/3 |" % (
                ("🍲 " + r["trial"]) if r.get("soup") else r["trial"], r["ms"],
                _f(r.get("t1"), 6), _f(r.get("d1"), 3), _f(r.get("t2"), 6),
                ("%+d" % r["dFP2"]) if r.get("dFP2") is not None else "-",
                _f(r.get("t3"), 6), _f(r.get("d3"), 3), n))
        out.append("")
    if rows_sp:
        out.append("### 速度达标里最快的几条")
        out.append("")
        for r in sorted([x for x in rows_sp if x.get("ms")], key=lambda x: x["ms"])[:8]:
            out.append("- `%s` NPU %.2f ms | ΔT1 %s pp | ΔT3 %s pp" % (
                r["trial"], r["ms"], _f(r.get("d1"), 3), _f(r.get("d3"), 3)))
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="周末轮输出根")
    ap.add_argument("--speed-table", required=True)
    ap.add_argument("--limit-ms", type=float, default=21.0)
    ap.add_argument("--t3-drop-max-pp", type=float, default=1.0)
    ap.add_argument("--t1-drop-max-pp", type=float, default=0.5)
    ap.add_argument("--t2-fp-max", type=int, default=10)
    ap.add_argument("--study-name", default="mix_ratio_search_npu20ms_v2",
                    help="study 名（★改结构维度的 choices 必须换新名："
                         "Optuna 的 CategoricalDistribution 不支持动态值空间）")
    ap.add_argument("--soup-candidates", default="",
                    help="结构线 soup 候选清单（默认自动找 <root>/soups/soup_candidates.json）；"
                         "soup 的速度/结构从这里取（普通臂的行为不变）")
    args = ap.parse_args()

    root = Path(args.root)
    speed = load_speed_table(args.speed_table)
    soup_cand_path = args.soup_candidates or str(root / "soups" / "soup_candidates.json")
    soups = load_soup_candidates(soup_cand_path)

    rows = []
    # ---- 从 optuna_summary.csv 取臂的参数（plan 的 fixed_params 会落在 params 里）----
    summary = root / "optuna_summary.csv"
    if summary.is_file():
        for r in csv.DictReader(summary.open()):
            rows.append({"trial": r.get("trial"), "row": r})

    print("=" * 150)
    print(f"{'臂':<44} {'NPUms':>7} {'T1':>9} {'ΔT1pp':>7} {'T2':>9} {'dFP2':>6} "
          f"{'T3':>9} {'ΔT3pp':>7} {'T3FP':>6}  判定")
    print("-" * 150)

    # 逐 trial 目录走一遍（门禁报告 + plan 参数）
    plan_params = {}
    plan_by_trial = {}
    plan_path = root / "npu20ms_plan.json"
    if not plan_path.is_file():
        plan_path = Path("manu/training/npu20ms_plan.json")
    if plan_path.is_file():
        _plan = json.loads(plan_path.read_text(encoding="utf-8"))
        for a in _plan:
            plan_params[a["role"]] = a["params"]
        # ★ trial 编号 ↔ plan 顺序：enqueue 是 FIFO ⇒ plan 的第 N 条 ⇒ trial_000N
        #   （补投的臂排在队尾 ⇒ 也按追加顺序接在 plan 之后）
        for _i, a in enumerate(_plan):
            plan_by_trial[f"trial_{_i:04d}"] = a["params"]
    # 补投的种子臂计划（若存在）⇒ 接在主 plan 之后
    _seed_path = root / "seed_arms.json"
    if not _seed_path.is_file():
        _seed_path = Path("manu/training/npu20ms_seed_arms.json")
    _seed_plan = []
    if _seed_path.is_file():
        _seed_plan = json.loads(_seed_path.read_text(encoding="utf-8"))
        for a in _seed_plan:
            plan_params[a["role"]] = a["params"]
        for _j, a in enumerate(_seed_plan):
            plan_by_trial[f"trial_{len(plan_by_trial) + _j:04d}"] = a["params"]

    def _params_from_plan(trial_name: str) -> dict:
        """study.db 拿不到参数时的退路：按 trial 编号查 plan。"""
        return dict(plan_by_trial.get(trial_name) or {})

    results = []
    tdir = root / "trials"
    if tdir.is_dir():
        for td in sorted(tdir.iterdir()):
            if not td.is_dir():
                continue
            g = read_gate_report(td)
            if not g:
                continue
            # ★ soup 行（结构线 Model Soup）：门禁产物在 trials/<soup_name>/gate/epochN/
            #   ⇒ 与普通臂同一个循环就能收；参数（结构字段）取自 soup 候选清单。
            is_soup = td.name in soups
            if is_soup:
                sc = soups[td.name]
                # ⚠️ **None 要换成默认值**（`params.get(k, default)` 遇到 None 值不生效）
                params = {k: v for k, v in (sc.get("struct") or {}).items() if v is not None}
                params.setdefault("head_src", "p1")
                params.setdefault("act", "relu")
                params.setdefault("head_conv", 64)
                params.setdefault("late_upsample", False)
                params.setdefault("p2_out_scale", 1.0)
            else:
                # 从 study.db 取该 trial 的参数（plan 的 fixed_params）
                params = {}
                try:
                    import optuna
                    st = optuna.load_study(study_name=args.study_name,
                                          storage=f"sqlite:///{(root / 'study.db').resolve()}")
                    for t in st.trials:
                        if t.number == int(td.name.split("_")[-1]):
                            # ⚠️ **兜底：入队的臂参数存在 system_attrs["fixed_params"]**
                            #    （`t.params` 只有在 suggest_* 之后才有 ⇒ 对刚入队/被剪枝的臂是空的）
                            params = dict(t.params or {})
                            if not params:
                                params = dict((getattr(t, "system_attrs", None) or {})
                                              .get("fixed_params") or {})
                            break
                except Exception:
                    params = {}
                if not params:
                    # 退路：plan 的参数（按 seed 对齐 —— role 里带 seed 后缀）
                    params = _params_from_plan(td.name)
            ms = speed.get(struct_key_of(params)) if params else None
            if is_soup:
                # ★ soup 行的速度取**候选清单**里查好的值（构建时按 ms_source=measured 优先，
                #   与 build_struct_soups.py 的口径一致）；清单里没有才回退本地速度表。
                _sms = soups[td.name].get("speed_ms")
                ms = _sms if _sms is not None else ms
            t1 = g.get("t1_f1")
            t3 = g.get("t3_f1")
            d1 = (t1 - CS_T0043["t1_f1"]) * 100 if isinstance(t1, float) else None
            d3 = (t3 - CS_T0043["t3_f1"]) * 100 if isinstance(t3, float) else None
            dFP2 = (g.get("t2_fp") - CS_T0043["t2_fp"]) if isinstance(g.get("t2_fp"), int) else None

            verdict = "-"
            if ms is not None and d3 is not None and d1 is not None and dFP2 is not None:
                ok_spd = ms <= args.limit_ms
                ok_t3 = d3 >= -args.t3_drop_max_pp
                ok_t1 = d1 >= -args.t1_drop_max_pp
                ok_t2 = dFP2 <= args.t2_fp_max
                verdict = ("🎯 达标" if (ok_spd and ok_t3 and ok_t1 and ok_t2)
                           else ("⚠️ 速度达标/精度不达标" if ok_spd else "❌ 速度超限"))
            if g.get("infra_failed"):
                # ★ 评测没跑出来 ⇒ 明确标出，**不参与判定**（别让它混进候选池）
                verdict = f"🔧 门禁未产出指标({','.join(g.get('failed_tracks') or []) or '?'})"
            results.append({"trial": td.name, "ms": ms, "t1": t1, "d1": d1, "t2": g.get("t2_f1"),
                            "dFP2": dFP2, "t3": t3, "d3": d3, "t3fp": g.get("t3_fp"),
                            "verdict": verdict, "params": params,
                            "soup": is_soup,
                            "family": soups[td.name].get("family") if is_soup else None})

            def _f(v, fmt="{:.6f}"):
                return fmt.format(v) if isinstance(v, float) else "-"

            label = ("🍲 " + td.name) if is_soup else td.name
            print(f"{label:<44} {(f'{ms:.2f}' if ms else '-'):>7} "
                  f"{_f(t1):>9} {(f'{d1:+.3f}' if d1 is not None else '-'):>7} "
                  f"{_f(g.get('t2_f1')):>9} "
                  f"{(f'{dFP2:+d}' if dFP2 is not None else '-'):>6} "
                  f"{_f(t3):>9} {(f'{d3:+.3f}' if d3 is not None else '-'):>7} "
                  f"{(g.get('t3_fp') if g.get('t3_fp') is not None else '-'):>6}  {verdict}")

    print("=" * 150)
    print(f"基线 cs_t0043：T1 {CS_T0043['t1_f1']:.6f}（FP {CS_T0043['t1_fp']}）| "
          f"T2 {CS_T0043['t2_f1']:.6f}（FP {CS_T0043['t2_fp']}）| "
          f"T3 {CS_T0043['t3_f1']:.6f}（FP {CS_T0043['t3_fp']}）")
    print(f"判据：NPU ≤ {args.limit_ms} ms ∧ ΔT3 ≥ −{args.t3_drop_max_pp} pp ∧ "
          f"ΔT1 ≥ −{args.t1_drop_max_pp} pp ∧ ΔFP_T2 ≤ {args.t2_fp_max}")
    ok = [r for r in results if r["verdict"].startswith("🎯")]
    if ok:
        ok.sort(key=lambda r: (-(r["t3"] or 0)))
        print("\n🏆 达标候选（按 T3 降序）：")
        for r in ok:
            print(f"   {r['trial']:<44} NPU {r['ms']:.2f} ms  T1 {r['t1']:.6f}  T3 {r['t3']:.6f}")
    else:
        print("\n（暂无达标候选）")
    out = root / "acceptance.json"
    payload = {"baseline": CS_T0043, "results": results}
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"\n[OK] -> {out}")
    # ★★ 2026-11-06：**同一次运行**产出人读总表 ACCEPTANCE.md。
    #   为什么要搬进来：`ACCEPTANCE.md` 原先只由看门狗的 inline 脚本生成，
    #   而看门狗在**第一轮收口**就跑完了（早于 soup 门禁）
    #   ⇒ 人读表会停在「单臂表」，**soup 行永远进不了交付物**。
    #   ⇒ 验收脚本是唯一在「两轮 + soup 都完成后」还会再跑一次的地方 ⇒ 放这里。
    md = render_markdown(payload, {"limit_ms": args.limit_ms,
                                    "t3_drop_max_pp": args.t3_drop_max_pp,
                                    "t1_drop_max_pp": args.t1_drop_max_pp,
                                    "t2_fp_max": args.t2_fp_max})
    md_path = root / "ACCEPTANCE.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"[OK] -> {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
