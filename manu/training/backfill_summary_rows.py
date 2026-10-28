#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把"已结算但没进 optuna_summary.csv"的 trial 补回汇总表。

## 为什么需要这个

调度器在**主循环内**收割完成的 trial 并写 `optuna_summary.csv`
（`optuna_mix_ratio_search.py:1000`）。到达 `--max-hours` 截止时间时，主循环直接
`break`，剩下的在跑 trial 进入 `[WAIT]` 收尾块——那个块**只等进程退出，不收割、
不写 CSV 行**（已登记缺陷：SIGTERM 路径同款）。

后果：截止时刻正在跑的那一批 trial，即使正常跑完、产物齐全
（`trial_result.json` + gate 报告 + 权重），也会**从所有分析脚本的视野里消失**
——`status_mix_ratio_search.py` / `analyze_mix_spec_results.py` /
`check_control_repro.py` 三者都只读 `optuna_summary.csv`。

## 本脚本做什么

1. 扫 `runs/optuna_mix_ratio/trials/*/trial_result.json`（worker 写的，**调度器不依赖**）。
2. 读现有 `optuna_summary.csv`，找出**缺行**的 trial。
3. 从 Optuna study 取该 trial 的 `params`（保证与派单时完全一致）。
4. 用调度器**自己的** `write_summary_row()` 补写行——格式单一真值，不复制格式定义。

判定 SUCCESS / PRUNED / CRASHED 的逻辑与调度器 `main()` 内**完全一致**：
- `tier1_ok` 且 `utility` 非空 ⇒ SUCCESS
- `infrastructure_failure` ⇒ CRASHED（门禁基础设施失败，不是触红线）
- 其余 ⇒ PRUNED（一级红线失败被剪枝）

用法（**执行机：服务器**）：

    cd /tmp/pycharm_project_10ae9e2e
    python manu/training/backfill_summary_rows.py            # 先看会补哪些（dry-run）
    python manu/training/backfill_summary_rows.py --apply    # 真写（先自动备份 CSV）

退出码：0 = 正常（无论是否补写），1 = 有错误。
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

DEFAULT_OUT_ROOT = Path("runs/optuna_mix_ratio")
DEFAULT_STORAGE = "sqlite:///runs/optuna_mix_ratio/study.db"
DEFAULT_STUDY = "mix_ratio_search"


def _load_dispatcher_module():
    """复用调度器的 write_summary_row（格式单一真值）。"""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import optuna_mix_ratio_search as oms  # noqa: E402

    return oms


def read_existing_rows(csv_path: Path) -> dict[str, dict]:
    if not csv_path.exists():
        return {}
    with open(csv_path, encoding="utf-8") as fh:
        return {r.get("trial", ""): r for r in csv.DictReader(fh)}


def classify(res: dict) -> str:
    """与调度器 main() 内完全一致的 SUCCESS / PRUNED / CRASHED 判定。"""
    if res.get("tier1_ok") and res.get("utility") is not None:
        return "SUCCESS"
    if res.get("infrastructure_failure"):
        return "CRASHED"
    return "PRUNED"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--storage", default=DEFAULT_STORAGE)
    ap.add_argument("--study-name", default=DEFAULT_STUDY)
    ap.add_argument("--apply", action="store_true", help="真写（默认 dry-run）")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    trials_dir = out_root / "trials"
    csv_path = out_root / "optuna_summary.csv"

    if not trials_dir.is_dir():
        print(f"[FATAL] 找不到 {trials_dir}（必须在服务器项目目录下执行）")
        return 1

    oms = _load_dispatcher_module()

    existing = read_existing_rows(csv_path)
    print(f"[INFO] 现有 CSV 行数：{len(existing)}")

    # 1) 收集已结算的 trial（以 worker 写的 trial_result.json 为准）
    settled: dict[str, dict] = {}
    for d in sorted(trials_dir.glob("trial_*")):
        rj = d / "trial_result.json"
        if not rj.is_file():
            continue
        try:
            import json

            settled[d.name] = json.loads(rj.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] {d.name}/trial_result.json 解析失败：{exc}")

    missing = [t for t in settled if t not in existing]
    print(f"[INFO] 已结算 trial：{len(settled)} 个；CSV 缺行：{len(missing)} 个")
    if not missing:
        print("[OK] 无需补采")
        return 0

    # 2) 从 Optuna study 取 params
    import optuna  # noqa: E402

    study = optuna.load_study(storage=args.storage, study_name=args.study_name)
    params_by_name: dict[str, dict] = {}
    for t in study.trials:
        params_by_name[f"trial_{t.number:04d}"] = dict(t.params)
        # Optuna 4.x 的 FrozenTrial **没有** `.name` 属性（用 getattr 取，别直接访问）
        alt_name = getattr(t, "name", None)
        if alt_name:
            params_by_name.setdefault(alt_name, dict(t.params))

    # 3) 补写
    to_write = []
    for name in sorted(missing):
        res = settled[name]
        params = params_by_name.get(name)
        if params is None:
            print(f"[WARN] {name} 在 study 里找不到 params，跳过")
            continue
        status = classify(res)
        # 2026-10-27 修：原来写死 0.0 ⇒ 补写行的 duration_min 全是 0，
        # 会把汇总里的「单 trial 耗时」中位数/最短/最长拉成 0.00（误导）。
        # `trial_result.json` 的 `elapsed_s` 是 worker 的**全程墙钟**（训练 + 门禁），
        # 与调度器自己算的 dur（(now-start_time)/60）同口径、只差零点几分钟 ⇒ 直接换算。
        try:
            dur = float(res.get("elapsed_s") or 0.0) / 60.0
        except (TypeError, ValueError):
            dur = 0.0
        to_write.append((name, params, res, status, dur))
        print(
            f"  + {name}  status={status}  mix={params.get('mix_spec')}  "
            f"U={res.get('utility')}  判决={res.get('gate_verdict')}"
        )

    if not to_write:
        print("[WARN] 没有可补写的行")
        return 0

    if not args.apply:
        print("\n[DRY-RUN] 以上为待补写行；加 --apply 真写")
        return 0

    if csv_path.exists():
        bak = csv_path.with_suffix(".csv.bak")
        shutil.copy2(csv_path, bak)
        print(f"[INFO] 已备份原 CSV → {bak}")

    for name, params, res, status, dur in to_write:
        oms.write_summary_row(csv_path, name, params, res, status, dur)
    print(f"\n[OK] 已补写 {len(to_write)} 行 → {csv_path}")

    after = read_existing_rows(csv_path)
    print(f"[INFO] 补写后 CSV 行数：{len(after)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
