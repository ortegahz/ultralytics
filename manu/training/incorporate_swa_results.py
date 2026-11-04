#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 **SWA 变体**（沿 epoch 轴的权重平均）的门禁结果并入交接口径报告（2026-11-02）。

为什么需要它
------------
SWA 产物**不是 study 里的 trial** ⇒ `judge_handover_candidate.py`（读 CSV）
和 `seal_candidate.py`（从 CSV 挑中位 trial）**都看不到它**。
⇒ 若不单独整合，周二 09:00 交出来的报告会**漏掉一条可能已经命中双目标的产物**。

它做什么
--------
1. 扫 `<out-root>/swa/gate/*/epoch*/gate_report.json` ⇒ 取每个 SWA 候选的
   T1 / T2(F1,FP) / T3；
2. 按**同一套硬门禁**判定（与 judge 一致）：
   T1 ≥ 红线 · T2 F1 ≥ 0.985 · T2 虚警 ≤ 112；
3. 追加一节到 `HANDOVER.md`（附录 C）+ 写 `SWA_SUMMARY.json`；
4. **若某个候选既过硬门禁、又是唯一 T3 ≥ 0.87 的 ⇒ 自动封存**（权重 + md5 + 登记 JSON），
   并在 stdout 打出醒目的「★ 命中」行。

用法（服务器，conda `uav`，项目根 /tmp/pycharm_project_10ae9e2e）：
```bash
python manu/training/incorporate_swa_results.py \
  --out-root /mnt/data/siping/runs/nogmc_focus \
  --ref /mnt/data/siping/runs/nogmc_ref_weekend/reference_frame.json \
  --handover-md /mnt/data/siping/runs/nogmc_focus/HANDOVER.md \
  --candidates-dir /mnt/data/siping/runs/CANDIDATES
```
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path


def _f(v):
    try:
        return float(v)
    except Exception:
        return None


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="/mnt/data/siping/runs/nogmc_focus")
    ap.add_argument("--ref", default="/mnt/data/siping/runs/nogmc_ref_weekend/reference_frame.json")
    ap.add_argument("--handover-md", default="")
    ap.add_argument("--candidates-dir", default="/mnt/data/siping/runs/CANDIDATES")
    ap.add_argument("--rc-id", default="Candidate_v2_swa")
    ap.add_argument("--t3-target", type=float, default=0.87)
    ap.add_argument("--seal", action="store_true", help="命中时自动封存")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    swa_dir = out_root / "swa"
    gate_root = swa_dir / "gate"
    ref = json.loads(Path(args.ref).read_text(encoding="utf-8"))
    fb = ref["frozen_baseline"]
    gate = ref.get("gate", {})
    t1_red = fb["track1_f1"] + float(gate.get("track1_min_delta", -0.005))
    t2_fp_max = fb["track2_fp"] + int(gate.get("track2_fp_max_increment", 60))
    t2_f1_min = float(gate.get("track2_f1_min", 0.985))

    rows = []
    if gate_root.is_dir():
        for name in sorted(p.name for p in gate_root.iterdir() if p.is_dir()):
            # ⚠️ **两种布局都要收**（2026-11-03 08:5x 实测踩到）：
            #    · 独立门禁脚本（gate_swa_variants.sh / gate_crossseed_soups.sh）用
            #      `--out-dir $GD` ⇒ 报告直接在 `<name>/gate_report.json`
            #    · worker 内嵌门禁用 `--out-dir $GD/epoch10` ⇒ `<name>/epoch*/gate_report.json`
            #    原来只扫 `epoch*/` ⇒ **真实布局下扫不到任何候选 ⇒ 脚本 rc=0 但
            #    SWA_SUMMARY.json 根本不写、HANDOVER.md 附录 C 空着 ⇒
            #    整条 SWA 路线的实测结果被**静默丢掉**（昨天夹具测试用的是 epoch*/ 布局）。
            reps = sorted((gate_root / name).glob("epoch*/gate_report.json"))
            if not reps:
                direct = gate_root / name / "gate_report.json"
                reps = [direct] if direct.is_file() else []
            if not reps:
                continue
            rep = reps[-1]
            d = json.loads(rep.read_text(encoding="utf-8"))
            tr = d.get("tracks") or {}
            t1 = _f((((tr.get("track1_uav") or {}).get("finetune_peak") or {}).get("f1")))
            t2 = tr.get("track2_golden") or {}
            t3 = tr.get("track3_blind") or {}
            t2f1 = _f(t2.get("f1"))
            t2fp = _f(t2.get("fp"))
            t3f1 = _f(t3.get("f1"))
            bad = []
            if t1 is None or t1 < t1_red:
                bad.append(f"T1 {t1} < 红线 {t1_red:.6f}")
            if t2f1 is None or t2f1 < t2_f1_min:
                bad.append(f"T2 F1 {t2f1} < {t2_f1_min}")
            if t2fp is None or t2fp > t2_fp_max:
                bad.append(f"T2 虚警 {t2fp} > {t2_fp_max}")
            rows.append({
                "name": name, "report": str(rep), "weights": str(swa_dir / f"{name}.pt"),
                "t1": t1, "t2_f1": t2f1, "t2_fp": t2fp, "t3": t3f1,
                "script_verdict": d.get("verdict"),
                "strict_ok": not bad, "bad": bad,
                "t3_ok": t3f1 is not None and t3f1 >= args.t3_target,
            })

    if not rows:
        print("[SWA] 还没有门禁结果（SWA 门禁在训练结束后才跑）⇒ 跳过整合")
        return 0

    rows.sort(key=lambda r: (-(r["t3"] or -1)))
    hit = [r for r in rows if r["strict_ok"] and r["t3_ok"]]
    strict_only = [r for r in rows if r["strict_ok"] and not r["t3_ok"]]

    print(f"[SWA] 读到 {len(rows)} 个候选 | 过硬门禁 {len([r for r in rows if r['strict_ok']])} 个"
          f" | 其中 T3 ≥ {args.t3_target} 的 **命中 {len(hit)} 个**")
    for r in rows:
        mark = "★命中" if (r["strict_ok"] and r["t3_ok"]) else ("严格PASS" if r["strict_ok"] else "未过")
        print(f"   {mark:8s} {r['name']:20s} T1={r['t1']} T2=({r['t2_f1']},FP{r['t2_fp']}) T3={r['t3']}")

    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "redlines": {"track1": t1_red, "track2_f1_min": t2_f1_min,
                     "track2_fp_max": t2_fp_max, "track3_target": args.t3_target},
        "n_candidates": len(rows),
        "n_strict_ok": len([r for r in rows if r["strict_ok"]]),
        "n_hit": len(hit),
        "candidates": rows,
    }
    (out_root / "SWA_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # ---- 追加到 HANDOVER.md -------------------------------------------
    if args.handover_md and Path(args.handover_md).exists():
        md = Path(args.handover_md).read_text(encoding="utf-8")
        # ⚠️ **必须幂等**（2026-11-03 08:5x 修）：原来是无条件 `md + 附录C` ⇒
        #    每次重跑 finalize 都会**再追加一份附录 C**（报告里出现两份同节）。
        #    ⇒ 追加前先把旧的「## 附录 C.」整段切掉。
        _mark = "## 附录 C."
        if _mark in md:
            md = md.split(_mark)[0].rstrip() + "\n"
        L = ["", "## 附录 C. 零训练成本变体（SWA 沿 epoch 轴 + 跨种子 soup 沿种子轴）\n"]
        L.append("> 变体 C 的 Model Soups 是**跨配置**平均（只是两端插值 ⇒ 判负）；"
                 "SWA 是**同一 run 相邻 epoch** 的平均（在同一盆地内 ⇒ 降方差、不改 T1/T3 权衡位置）；"
                 "`cs_*` 是**同配方不同种子**的末 epoch 平均（沿种子轴）。两者都是**零训练成本**。\n")
        L.append("| 候选 | T1 | T1 缺口 | T2 F1(FP) | T3 | T3 ≥ 0.87 | 硬门禁 |")
        L.append("|:--|--:|--:|--:|--:|:--:|:--|")
        for r in rows:
            gap = (r["t1"] - t1_red) * 100 if r["t1"] is not None else None
            L.append(f"| {r['name']} | {r['t1']:.6f} | {gap:+.4f} pp | "
                     f"{r['t2_f1']:.6f}({r['t2_fp']:.0f}) | {r['t3']:.6f} | "
                     f"{'✅' if r['t3_ok'] else '❌'} | "
                     f"{'✅' if r['strict_ok'] else '❌ ' + '; '.join(r['bad'])} |")
        L.append("")
        if hit:
            L.append(f"### ★ 命中：**{', '.join(r['name'] for r in hit)}**"
                     f"（T1 过线 + T2 达标 + T3 ≥ {args.t3_target}）")
            L.append("")
            for r in hit:
                L.append(f"- **{r['name']}**：权重 `{r['weights']}`"
                         f"（门禁报告 `{r['report']}`）")
            L.append("")
        elif strict_only:
            L.append(f"### 严格 PASS 但 T3 未达 {args.t3_target}："
                     f"{', '.join(r['name'] for r in strict_only)}")
            L.append("")
        else:
            L.append(f"### 结论：{len(rows)} 个 SWA 候选**均未同时过硬门禁与 T3 ≥ {args.t3_target}**")
            L.append("")
        Path(args.handover_md).write_text(md + "\n".join(L) + "\n", encoding="utf-8")
        print(f"[SWA] 已把附录 C 追加到 {args.handover_md}")

    # ---- 命中 ⇒ 自动封存 ---------------------------------------------
    if hit and args.seal:
        r = hit[0]
        src = Path(r["weights"])
        if not src.exists():
            print(f"[SWA] ★ 命中但权重不存在：{src} ⇒ 未封存")
            return 1
        out_dir = Path(args.candidates_dir) / args.rc_id
        out_dir.mkdir(parents=True, exist_ok=True)
        sealed = out_dir / f"{args.rc_id}_{r['name']}.pt"
        # ⚠️ **重封前必须先解除只读**（与 seal_candidate.py 同一套逻辑）：
        #    封存副本是 0444（防篡改）⇒ 第二次封存同一个 rc_id 时 `shutil.copy2`
        #    直接抛 **PermissionError** ⇒ 整个收尾的固化步骤 rc=1 失败。
        #    ⇒ 覆盖前先 chmod +w（只影响"能不能被覆盖"，封存后仍会再置回 0444）。
        if sealed.exists():
            try:
                os.chmod(sealed, 0o644)
            except OSError:
                pass
        shutil.copy2(src, sealed)
        os.chmod(sealed, 0o444)
        if _md5(sealed) != _md5(src):
            print("[SWA] [FATAL] 封存副本与源权重 md5 不一致 ⇒ 中止")
            return 5
        reg = {
            "rc_id": args.rc_id,
            "rc_name": f"SWA（沿 epoch 轴权重平均）命中候选 · {r['name']}",
            "sealed_at_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "artifact": {"sealed_weights": str(sealed), "source_weights": str(src),
                         "size_bytes": sealed.stat().st_size,
                         "md5": _md5(sealed), "sha256": _sha256(sealed),
                         "sealed_mode": "0444"},
            "metrics_of_sealed_trial": {"t1_f1": r["t1"], "t2_f1": r["t2_f1"],
                                        "t2_fp": r["t2_fp"], "t3_f1": r["t3"]},
            "gate_report": r["report"],
            "reference_frame": args.ref,
            "note": ("⚠️ 这是**权重后处理**产物（不是 study 里的 trial）⇒ "
                     "它的『多种子中位』证据来自底层配方的种子；"
                     "封存的是**该 SWA 权重自身**的三轨门禁结果。"),
            "reproduce": {"gate": (f"python manu/training/auto_epoch_gate_check.py --weights {src} "
                                   f"--out-dir {r['report'].rsplit('/', 1)[0]} --gpus 4 --device 0 "
                                   f"--infer-gpu 0 --cache-gpus 0,1,2,3 --baseline-cache "
                                   f"/mnt/data/siping/runs/nogmc_ref_weekend/uav_cache_base.pkl "
                                   f"--min-track2-f1 0.985 --max-track2-fp 112 "
                                   f"--min-track1-f1 0.8958967482400269 "
                                   f"--uav-data /mnt/data/siping/datasets/manu/uav_gmc_median_nogmc/data.yaml "
                                   f"--ft-root /mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_nogmc")},
        }
        (out_dir / f"{args.rc_id}.json").write_text(
            json.dumps(reg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[SWA] ★★★ 已封存命中候选：{sealed}")
        print(f"      md5={reg['artifact']['md5']} size={reg['artifact']['size_bytes']}")
        print(f"      T1={r['t1']:.6f} T2=({r['t2_f1']:.6f},FP {r['t2_fp']:.0f}) T3={r['t3']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
