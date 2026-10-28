#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""混样「实测比」重放器（deterministic replay of the WeightedRandomSampler）。

为什么需要它
------------
2026-10-27 抓到一个 bug：`optuna_mix_ratio_search.py` 里统计"这一批有多少 LQ 样本"时
**硬编码**了 GMC 目录名 ``"longquanshan_ft_gmc_median_official"``：

    n_lq_batch = sum(1 for f in im_files if "longquanshan_ft_gmc_median_official" in str(f))

no-GMC 跑的目录名是 ``longquanshan_ft_gmc_median_nogmc`` ⇒ **永远匹配不上**
⇒ `n_lq_batch` 恒为 0 ⇒ 日志里 "mix LQ/UAV 0/32"，
   `measured_ratio = uav_seen / max(lq_seen, 1)` 变成 ~56926 的垃圾值。

**注意：采样本身是对的** —— `WeightedRandomSampler` 的权重是用**样本数**
（`n_lq` / `n_uav`）构造的，不看路径 ⇒ LQ 样本确实按 `P(LQ)=1/(1+r)` 进了 batch。
受影响的**只有上报字段**（日志混样列 / `measured_ratio` / weights 元数据），
**不影响**训练、权重、门禁三轨指标与 U 判定。

本工具把"实测比"从**确定性重放**里重新算出来
（`WeightedRandomSampler` 的抽样序列由 seeded generator 完全决定 ⇒ 可逐位复现），
用来：
  · `--verify` ：对 GMC 研究（硬编码字符串当时是对的）逐 trial 校验重放配方；
  · `--repair` ：修 no-GMC 侧被 bug 污染的 `trial_result.json` 与 `optuna_summary.csv`。

重放配方（**已在 GMC 侧 104/104 逐位验证通过**）
--------------------------------------------------
1. ``torch.manual_seed(trial_seed)``               —— worker 的全局播种（`torch.initial_seed()`）
2. ``g = torch.Generator()``；``g.manual_seed((6148914691236517205 + RANK + seed) % 2**64)``
   —— 来自 ultralytics `build_dataloader`（``RANK=-1`` ⇒ 减 1）
3. ``torch.empty((), dtype=torch.int64).random_(generator=g)``  —— **只抽一次**
   —— DataLoader 迭代器的 ``base_seed``；``InfiniteDataLoader`` **复用迭代器**，
   所以第 2、3 个 epoch **不再抽**
4. 每个 epoch 一次 ``torch.multinomial(weights, num_samples, True, generator=g)``
   —— ``WeightedRandomSampler.__iter__`` 就是这么干的（torch 1.13.1 源码）
5. **只取最后一个 epoch 的计数**（``lq_seen`` 在每个 epoch 开头被重置）
6. ``natural`` 模式没有 WeightedRandomSampler（普通 RandomSampler 全量无放回）
   ⇒ 每 epoch 恰好 LQ=n_lq / UAV=n_uav

用法
----
```bash
# 校验（对 GMC 研究；期望 104/104 逐位一致）
python manu/training/replay_measured_ratio.py --verify --trials-root runs/optuna_mix_ratio/trials

# 修复 no-GMC 侧的实测比（trial_result.json + optuna_summary.csv）
python manu/training/replay_measured_ratio.py --repair --out-root runs/optuna_nogmc --apply
```
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

# 与 optuna_mix_ratio_search.py 的 DATA-GATE 常量一致（满额帧数）
LQ_EXPECT = 13918
UAV_EXPECT = 43008

# ultralytics/data/build.py::build_dataloader
BUILD_DL_SEED_BASE = 6148914691236517205
RANK = -1


def replay_mix_counts(
    mix_spec: str,
    trial_seed: int,
    *,
    n_lq: int = LQ_EXPECT,
    n_uav: int = UAV_EXPECT,
    epochs: int = 3,
) -> tuple[int, int]:
    """重放某个 trial 最后一个 epoch 的 (LQ 抽样数, UAV 抽样数)。"""
    if mix_spec == "natural":
        # 普通 RandomSampler 全量无放回 ⇒ 每 epoch 恰好各一遍
        return n_lq, n_uav

    uav_per_lq = float(mix_spec[1:])
    p_lq = 1.0 / (1.0 + uav_per_lq)
    p_uav = 1.0 - p_lq
    weights = torch.as_tensor(
        np.concatenate(
            [
                np.full(n_lq, p_lq / n_lq, dtype=np.float64),
                np.full(n_uav, p_uav / n_uav, dtype=np.float64),
            ]
        ),
        dtype=torch.double,
    )
    num_samples = n_lq + n_uav

    torch.manual_seed(trial_seed)
    g = torch.Generator()
    g.manual_seed((BUILD_DL_SEED_BASE + RANK + trial_seed) % (1 << 64))
    # 迭代器的 base_seed：**只抽一次**（InfiniteDataLoader 复用迭代器）
    torch.empty((), dtype=torch.int64).random_(generator=g)

    lq = 0
    for _ in range(epochs):
        idx = torch.multinomial(weights, num_samples, True, generator=g)
        lq = int((idx < n_lq).sum())  # 只保留最后一个 epoch
    return lq, num_samples - lq


def _trial_number(trial_name: str) -> int:
    return int(trial_name.split("_")[1])


def cmd_verify(args) -> int:
    trials_root = Path(args.trials_root)
    rows = []
    # 只验**最后一个 epoch** 的权重：重放脚本复现的是"最后一个 epoch 的计数"
    # （lq_seen 每个 epoch 开头重置 ⇒ 落盘的 measured_* 就是末 epoch 的值）
    for d in sorted(trials_root.glob(f"trial_*/weights/epoch{args.epochs}.pt")):
        try:
            ft = torch.load(d, map_location="cpu").get("finetune", {})
        except Exception:
            continue
        if not ft.get("mix_spec") or ft.get("measured_lq") is None:
            continue
        rows.append((_trial_number(d.parts[-3]), ft["mix_spec"], ft["measured_lq"], ft["measured_uav"]))
    rows.sort()

    exact = mismatch = 0
    max_diff = 0
    bad = []
    for n, spec, mlq, muav in rows:
        plq, puav = replay_mix_counts(spec, n + args.seed_offset)
        if plq == mlq and puav == muav:
            exact += 1
        else:
            mismatch += 1
            max_diff = max(max_diff, abs(plq - mlq))
            bad.append((n, spec, mlq, muav, plq, puav))
    for n, spec, mlq, muav, plq, puav in bad[:20]:
        print(f"  MISMATCH trial_{n:04d} {spec}: truth {mlq}/{muav} vs replay {plq}/{puav}")
    print(f"[VERIFY] {trials_root}: 逐位一致 {exact} / 不一致 {mismatch}（共 {exact + mismatch}），最大偏差 {max_diff} 个样本")
    return 0 if mismatch == 0 else (0 if args.lenient else 1)


def cmd_repair(args) -> int:
    out_root = Path(args.out_root)
    csv_path = out_root / "optuna_summary.csv"

    # ---- 1. 收集每个 trial 的 mix_spec（从 CSV 或 trial_result.json）----
    specs: dict[int, str] = {}
    csv_rows: list[dict] = []
    if csv_path.is_file():
        with csv_path.open(encoding="utf-8") as fh:
            rd = csv.DictReader(fh)
            fieldnames = list(rd.fieldnames or [])
            for r in rd:
                csv_rows.append(r)
                try:
                    specs[_trial_number(r["trial"])] = r["mix_spec"]
                except Exception:
                    pass

    fixed_res = fixed_csv = 0
    details = []

    # ---- 2. 修 trial_result.json ----
    for tr in sorted(out_root.glob("trials/trial_*/trial_result.json")):
        n = _trial_number(tr.parts[-2])
        spec = specs.get(n)
        if not spec:
            try:
                spec = json.loads(tr.read_text(encoding="utf-8")).get("mix_spec")
            except Exception:
                spec = None
        if not spec:
            continue
        try:
            data = json.loads(tr.read_text(encoding="utf-8"))
        except Exception:
            continue
        mq = data.get("measured_lq")
        mr = data.get("measured_ratio")
        # 只修"明显被 bug 污染"的：lq 计数为 0 或 ratio 荒谬地大
        if not (mq == 0 or (mr is not None and mr > 1000)):
            continue
        lq, uav = replay_mix_counts(spec, n + args.seed_offset)
        ratio = uav / max(lq, 1)
        old = (mq, mr)
        data["measured_lq"] = lq
        data["measured_uav"] = uav
        data["measured_ratio"] = ratio
        data["measured_ratio_source"] = "deterministic_replay (计数器缺陷期，重放复原)"
        if not args.apply:
            print(f"  [DRY] trial_{n:04d} {spec}: {old} → ({lq}, {uav}, {ratio:.6f})")
            fixed_res += 1
            continue
        tr.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        fixed_res += 1
        details.append((n, spec, lq, uav, ratio))

    # ---- 3. 修 CSV 的 measured_ratio 列 ----
    if csv_rows and "measured_ratio" in (fieldnames or []):
        idx = fieldnames.index("measured_ratio")
        for r in csv_rows:
            try:
                n = _trial_number(r["trial"])
            except Exception:
                continue
            spec = r.get("mix_spec")
            if not spec:
                continue
            try:
                cur = float(r.get("measured_ratio") or 0)
            except Exception:
                cur = 0.0
            if not (cur > 1000):
                continue
            lq, uav = replay_mix_counts(spec, n + args.seed_offset)
            ratio = uav / max(lq, 1)
            r["measured_ratio"] = f"{ratio:.3f}"
            fixed_csv += 1
            details.append((n, spec, lq, uav, ratio))
        if args.apply and fixed_csv:
            bak = csv_path.with_suffix(csv_path.suffix + ".bak")
            bak.write_text(csv_path.read_text(encoding="utf-8"), encoding="utf-8")
            with csv_path.open("w", encoding="utf-8", newline="") as fh:
                wr = csv.DictWriter(fh, fieldnames=fieldnames)
                wr.writeheader()
                wr.writerows(csv_rows)
            print(f"[REPAIR] CSV 已重写（备份 {bak}）")

    for n, spec, lq, uav, ratio in details:
        print(f"  trial_{n:04d} {spec:8s} → LQ={lq:,} UAV={uav:,} 实测 1:{ratio:.6f}")
    print(
        f"[REPAIR] {out_root}: trial_result.json 修 {fixed_res} 个，CSV 修 {fixed_csv} 个"
        f"{'' if args.apply else '（dry-run，未落盘）'}"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="混样实测比的重放 / 校验 / 修复")
    ap.add_argument("--verify", action="store_true", help="校验重放配方（对已有真值的研究）")
    ap.add_argument("--repair", action="store_true", help="修复被计数器 bug 污染的实测比")
    ap.add_argument("--trials-root", default="runs/optuna_mix_ratio/trials", help="--verify 用")
    ap.add_argument("--out-root", default="runs/optuna_nogmc", help="--repair 用")
    ap.add_argument("--seed-offset", type=int, default=0, help="trial seed = trial_number + offset")
    ap.add_argument("--epochs", type=int, default=3, help="训练 epoch 数（重放按此数推进 generator）")
    ap.add_argument("--lenient", action="store_true", help="--verify 允许存在偏差")
    ap.add_argument("--apply", action="store_true", help="--repair 真正落盘（默认 dry-run）")
    args = ap.parse_args()

    if args.verify:
        return cmd_verify(args)
    if args.repair:
        return cmd_repair(args)
    ap.error("必须指定 --verify 或 --repair")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
