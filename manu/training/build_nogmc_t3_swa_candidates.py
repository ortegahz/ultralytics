#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 no-GMC T3 study 的 **SWA 候选清单**（沿 epoch 轴的末 N 个 epoch 权重平均）。

为什么
----
`gate_swa_variants.sh` / `gate_crossseed_soups.sh` 里把候选列表**写死**了
⇒ 换 study（weekend → focus → t3）就得改脚本。
本脚本把「候选清单」从脚本里搬到**数据**里：新 study 跑完后，
它只读 `study.db` + `trials/` 目录，吐出每个 trial 的末 N 个 epoch 权重路径，
门禁脚本照单全收 ⇒ **不用改脚本**就能对任何 study 做 SWA 门禁。

它做什么
--------
1. 只读打开 `study.db`（**不写、不训练、不占 GPU**）；
2. 对每个 trial：
   · `state == COMPLETE`（**用 `t.state.name` 字符串比**，不能用 `t.state.value` 枚举值）；
   · 找 `trials/trial_XXXX/weights/epoch*.pt`，取**最大 epoch 号** N；
   · **要求 N ≥ 3**（不足 3 个 epoch 的 trial 平均不出来，也没意义）；
   · 取 epoch{N-2, N-1, N} 三个权重；
3. 写清单 `<swa-dir>/swa_candidates.json`；
4. `--fuse` 时再调 `fuse_weights.py` 把三者平均成 `<swa-dir>/swa_<trial>_last3.pt`
   （门禁脚本用；**融合只是读权重+写文件，不占 GPU**）。

命名
----
`swa_<trial编号>_last<N>`，例如 `swa_0033_last3`。
产物放在 `<swa-dir>/`，门禁报告放在 `<swa-dir>/gate/<name>/gate_report.json`
⇒ 正好是 `incorporate_swa_results.py` 认得的布局（它两种布局都收）。

用法（服务器，conda `uav`，项目根 /tmp/pycharm_project_10ae9e2e）：
```bash
python manu/training/build_nogmc_t3_swa_candidates.py \
  --out-root /mnt/data/siping/runs/nogmc_t3 \
  --study-name mix_ratio_search_nogmc_t3 --fuse
```
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

DEFAULT_OUT_ROOT = "/mnt/data/siping/runs/nogmc_t3"
DEFAULT_STUDY = "mix_ratio_search_nogmc_t3"


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _epoch_files(trial_dir: Path) -> list:
    wdir = trial_dir / "weights"
    if not wdir.is_dir():
        return []
    return sorted(wdir.glob("epoch*.pt"), key=lambda p: int(p.stem.replace("epoch", "")))


def main() -> int:
    ap = argparse.ArgumentParser(description="生成 no-GMC study 的 SWA 候选清单")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT,
                    help="study 产物根（study.db / trials/ 都在它下面）")
    ap.add_argument("--storage", default="", help="Optuna storage URL（默认 sqlite:///<out-root>/study.db）")
    ap.add_argument("--study-name", default=DEFAULT_STUDY)
    ap.add_argument("--swa-dir", default="", help="SWA 产物目录（默认 <out-root>/swa）")
    ap.add_argument("--last-n", type=int, default=3, help="平均末尾几个 epoch（默认 3）")
    ap.add_argument("--min-epochs", type=int, default=3,
                    help="trial 至少要有几个 epoch 才做 SWA（默认 3）")
    ap.add_argument("--out-json", default="", help="清单输出路径（默认 <swa-dir>/swa_candidates.json）")
    ap.add_argument("--repo", default="/tmp/pycharm_project_10ae9e2e")
    ap.add_argument("--fuse", action="store_true",
                    help="顺带调用 fuse_weights.py 把末 N 个 epoch 平均成 .pt（不占 GPU）")
    ap.add_argument("--dry-run", action="store_true", help="只打印清单，不写任何文件")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    storage = args.storage or ("sqlite:///" + str(out_root / "study.db"))
    swa_dir = Path(args.swa_dir) if args.swa_dir else out_root / "swa"
    out_json = Path(args.out_json) if args.out_json else swa_dir / "swa_candidates.json"
    n_last = max(1, int(args.last_n))

    db = out_root / "study.db"
    if not db.is_file():
        print(f"[FATAL] 找不到 {db}（study 还没建？）")
        return 1
    try:
        st = optuna.load_study(storage=storage, study_name=args.study_name)
    except Exception as exc:  # noqa: BLE001
        print(f"[FATAL] 加载 study {args.study_name} 失败：{exc}")
        return 1

    print(f"[SWA-CAND] study={args.study_name} trials={len(st.trials)} "
          f"| 平均末尾 {n_last} 个 epoch | 要求 epoch ≥ {args.min_epochs}")

    cands = []
    skipped = {"not_complete": 0, "no_weights": 0, "too_few_epochs": 0, "no_dir": 0}
    for t in st.trials:
        # ⚠️ **必须用 state.name（字符串）**，不能用 state.value（枚举 COMPLETE=1）
        state = t.state.name
        tdir = out_root / "trials" / f"trial_{t.number:04d}"
        role = (t.user_attrs or {}).get("role")
        # ⚠️ **不能只认 COMPLETE**（2026-11-03 18:20 修正；与 `build_crossseed_soups.py`
        #    是**同一个根因的第二个现场**）：
        #    本轮调度器把「训练跑完、但门禁判定 FAIL」的臂标成 **PRUNED**
        #    （Optuna 语义：PRUNED = 被剪枝/未完成）⇒ 若只收 COMPLETE，
        #    **今晚跑出来的每一条臂都会被跳过** ⇒ **SWA 通道整条空转**
        #    （而门禁的价值排序表里点名的 `T3_W2_LR30_15EP` / `INS_T0033` 等，
        #     按老判据其实**一条都进不了** ⇒ 今晚 15 小时的产出在 SWA 通道同样全部失明）。
        # ⇒ 改用**产物判据**（与 Optuna state 解耦）：
        #    ① state 不能是 RUNNING / WAITING / FAIL（正在训练 ⇒ 权重是半成品）
        #    ② 权重存在，且**最大 epoch 号 == 计划的 epochs**（末 epoch 完整）
        #    ③ `trial_result.json` 存在（门禁真的跑完了）
        # ⇒ 手动剪掉的 WAITING 臂（无权重）与被 hard_cut 砍在半路的臂（epoch 不足）仍被排除。
        if state in ("RUNNING", "WAITING", "FAIL"):
            skipped["not_complete"] += 1
            continue
        if not tdir.is_dir():
            skipped["no_dir"] += 1
            continue
        fs = _epoch_files(tdir)
        if not fs:
            skipped["no_weights"] += 1
            continue
        if state == "PRUNED":
            plan_ep = t.params.get("epochs")
            got_ep = int(fs[-1].stem.replace("epoch", ""))
            if plan_ep and got_ep != int(plan_ep):
                skipped["too_few_epochs"] += 1
                print(f"  · t{t.number:04d}（{role}）状态=PRUNED 且只跑到 epoch{got_ep}/{int(plan_ep)} ⇒ 半成品 ⇒ 跳过")
                continue
            if not (tdir / "trial_result.json").is_file():
                skipped["not_complete"] += 1
                continue
        if len(fs) < max(n_last, int(args.min_epochs)):
            skipped["too_few_epochs"] += 1
            print(f"  · t{t.number:04d}（{role}）只有 {len(fs)} 个 epoch ⇒ 不足 {args.min_epochs} ⇒ 跳过")
            continue
        picked = fs[-n_last:]
        name = f"swa_{t.number:04d}_last{n_last}"
        cands.append({
            "name": name,
            "trial": f"trial_{t.number:04d}",
            "trial_number": t.number,
            "role": role,
            "state": state,
            "epochs": [int(p.stem.replace("epoch", "")) for p in picked],
            "weights": [str(p) for p in picked],
            "out": str(swa_dir / f"{name}.pt"),
        })
        print(f"  · {name}（{role or '无 role'}）epoch {cands[-1]['epochs']} ⇒ "
              f"{len(picked)} 个权重")

    print(f"[SWA-CAND] 可用候选 {len(cands)} 个 | 跳过：{skipped}")

    if not args.dry_run:
        swa_dir.mkdir(parents=True, exist_ok=True)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(
            {"study": args.study_name, "out_root": str(out_root),
             "swa_dir": str(swa_dir), "last_n": n_last,
             "min_epochs": int(args.min_epochs),
             "n_candidates": len(cands), "skipped": skipped,
             "candidates": cands},
            ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[SWA-CAND] 清单已写 {out_json}")

    if args.fuse and not args.dry_run:
        for c in cands:
            out = Path(c["out"])
            cmd = [sys.executable, str(Path(args.repo) / "manu/training/fuse_weights.py"),
                   "--inputs"] + c["weights"] + ["--out", str(out), "--name", c["name"]]
            print(f"[SWA-CAND] 融合 {c['name']} …")
            subprocess.run(cmd, check=True, cwd=args.repo)
            c["md5"] = _md5(out)
            c["fused"] = True
        if not args.dry_run:
            out_json.write_text(json.dumps(
                {"study": args.study_name, "out_root": str(out_root),
                 "swa_dir": str(swa_dir), "last_n": n_last,
                 "min_epochs": int(args.min_epochs),
                 "n_candidates": len(cands), "skipped": skipped,
                 "candidates": cands},
                ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[SWA-CAND] 融合完成 {len(cands)} 个")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
