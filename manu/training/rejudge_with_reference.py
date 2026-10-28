#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用（最终）参照系包**重新判定**所有 trial 的一级红线与 U，并同步落盘产物。

为什么需要它
------------
2026-10-27 11:12，用户授权把 ``gate.track2_fp_max_increment`` 从 10 重标定到 60
（no-GMC 语义下混训后的 ΔFP_T2 正常带是 [+43,+51]，沿用 +10 会让 100% trial 被剪枝）。

参照系阈值存在 ``runs/nogmc_ref/reference_frame.json`` 里，而 **worker 每次启动都重读它**
⇒ 新派单的 trial 自动用新线。但**重标定之前已经在跑的 trial**（本轮是 trial_0000~0007）
是在旧线下判定的，判定结果与最终口径不一致。

本脚本做一次**权威重判**：
  · 从每个 trial 的 ``gate/epoch3/gate_report.json`` 取回**真实指标**（不重新推理）；
  · 用最终参照系包的 ``frozen_baseline`` / ``rc1`` / ``gate`` 阈值重跑
    ``hard_gate`` 与 ``utility``（与 worker 的调用**逐参数一致**）；
  · 同步写回 ``trial_result.json`` 与 ``optuna_summary.csv`` 的
    ``status`` / ``gate_verdict`` / ``utility`` / ``utility_vs_rc1`` 四列。

口径纪律：指标本身**一个都不重算**（只用落盘的 gate_report），
被重算的只有**判定**（这正是口径改动的意义所在）。

用法
----
```bash
# 干跑（只报告会变什么）
python manu/training/rejudge_with_reference.py --out-root runs/optuna_nogmc --ref runs/nogmc_ref/reference_frame.json

# 真正落盘
python manu/training/rejudge_with_reference.py --out-root runs/optuna_nogmc --ref runs/nogmc_ref/reference_frame.json --apply
```
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from manu.training.mix_ratio_judgement import (  # noqa: E402
    hard_gate,
    metrics_from_gate_report,
    utility,
)


def _f6(v) -> str:
    """与调度器 write_summary_row 的 `g()` 同格式。"""
    return f"{v:.6f}" if isinstance(v, float) else str(v)


def _status_of(res: dict, tier1_ok: bool, utility_value) -> str:
    """与调度器 main() 的判定**完全一致**。"""
    if tier1_ok and utility_value is not None:
        return "SUCCESS"
    if res.get("infrastructure_failure"):
        return "CRASHED"
    return "PRUNED"


def main() -> int:
    ap = argparse.ArgumentParser(description="用最终参照系包重判所有 trial 的门禁与 U")
    ap.add_argument("--out-root", default="runs/optuna_nogmc")
    ap.add_argument("--ref", default="runs/nogmc_ref/reference_frame.json")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--apply", action="store_true", help="真正落盘（默认干跑）")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    ref_path = Path(args.ref)
    if not ref_path.is_file():
        print(f"[FAIL] 找不到参照系包：{ref_path}")
        return 3
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    base = ref.get("frozen_baseline") or {}
    rc1 = ref.get("rc1") or {}
    gate_cfg = ref.get("gate") or {}
    if not base:
        print("[FAIL] 参照系包缺 frozen_baseline")
        return 3
    print(f"[REF] 阈值：{gate_cfg}")
    print(f"[REF] 基线 T1={base.get('track1_f1')} | 新擂主 T1={(rc1 or {}).get('track1_f1')}")

    # ---------------------------------------------------------------- 逐 trial 重判
    changed = []
    total = 0
    judged: dict[str, dict] = {}   # trial_name → 重判后的 result（落盘用，勿再从磁盘重读）
    for tr in sorted(out_root.glob("trials/trial_*/trial_result.json")):
        rep = tr.parent / "gate" / f"epoch{args.epochs}" / "gate_report.json"
        if not rep.is_file():
            continue  # 基础设施失败：没有指标可重判，保持原样
        try:
            res = json.loads(tr.read_text(encoding="utf-8"))
        except Exception:
            continue
        try:
            m = metrics_from_gate_report(rep)
        except Exception as exc:
            print(f"  [SKIP] {tr.parent.name}: {exc}")
            continue
        total += 1

        ok, reason, gdetail = hard_gate(
            m, base,
            track2_f1_min=gate_cfg.get("track2_f1_min", 0.985),
            track2_fp_max_increment=gate_cfg.get("track2_fp_max_increment", 10),
            track1_min_delta=gate_cfg.get("track1_min_delta", -0.005),
        )
        u_base, _ = utility(m, base)
        u_rc1 = utility(m, rc1)[0] if rc1 else None
        new_status = _status_of(res, ok, u_base if ok else None)

        old = (
            res.get("status"), res.get("gate_verdict"), res.get("tier1_ok"),
            res.get("utility"), res.get("utility_vs_rc1"),
        )
        res["gate_verdict"] = reason
        res["tier1_ok"] = bool(ok)
        res["tier1_reason"] = reason
        res["status"] = new_status
        res["utility"] = u_base if ok else None
        res["utility_vs_rc1"] = u_rc1
        res["gate_detail"] = gdetail
        res["rejudged_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        res["rejudged_with_reference"] = str(ref_path)
        new = (res["status"], res["gate_verdict"], res["tier1_ok"], res["utility"], res["utility_vs_rc1"])
        judged[tr.parent.name] = res
        if old[:3] != new[:3] or old[3] != new[3]:
            changed.append((tr.parent.name, old, new))

    # ---------------------------------------------------------------- 落盘
    if args.apply:
        n_res = 0
        for tr in sorted(out_root.glob("trials/trial_*/trial_result.json")):
            res = judged.get(tr.parent.name)
            if res is None:
                continue
            tr.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            n_res += 1
        print(f"[REJUDGE] trial_result.json 已落盘 {n_res} 个")

    # ---------------------------------------------------------------- CSV 同步
    csv_path = out_root / "optuna_summary.csv"
    csv_rows = []
    if csv_path.is_file():
        with csv_path.open(encoding="utf-8") as fh:
            rd = csv.DictReader(fh)
            fieldnames = list(rd.fieldnames or [])
            csv_rows = list(rd)
        by_name = dict(judged)
        n_csv = 0
        for row in csv_rows:
            r = by_name.get(row.get("trial"))
            if not r:
                continue
            row["status"] = r.get("status", row.get("status"))
            row["gate_verdict"] = r.get("gate_verdict", row.get("gate_verdict"))
            row["utility"] = r.get("utility", "")
            row["utility_vs_rc1"] = r.get("utility_vs_rc1", "")
            n_csv += 1
        if args.apply and n_csv:
            bak = csv_path.with_suffix(csv_path.suffix + ".bak_rejudge")
            shutil.copy2(csv_path, bak)
            with csv_path.open("w", encoding="utf-8", newline="") as fh:
                wr = csv.DictWriter(fh, fieldnames=fieldnames)
                wr.writeheader()
                wr.writerows(csv_rows)
            print(f"[REJUDGE] CSV 已同步 {n_csv} 行（备份 {bak}）")

    # ---------------------------------------------------------------- 报告
    print(f"\n[REJUDGE] 共重判 {total} 个 trial，判定发生变化 {len(changed)} 个")
    for name, old, new in changed:
        print(f"  {name}: status {old[0]}→{new[0]} | verdict {old[1]}→{new[1]} | U {old[3]}→{new[3]}")
    if not args.apply:
        print("[REJUDGE] 干跑模式，未落盘（加 --apply 生效）")
    else:
        print("[REJUDGE] 已落盘")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
