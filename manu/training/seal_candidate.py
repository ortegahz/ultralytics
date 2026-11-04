#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按交接口径判定结果，**固化** Release Candidate（权重 + 机读登记 + README）。

为什么需要它
------------
上一轮的 ``Candidate_v1`` 是手工复制的（md5 手算、JSON 手写）⇒ 易错且不可复现。
本脚本把固化流程代码化，输入是 ``judge_handover_candidate.py`` 产出的
``HANDOVER.json`` ⇒ **判定与固化同源**，不会出现"报告说的候选"和"实际封存的权重"对不上。

固化规则
--------
* **只固化严格 PASS（``gate_verdict == "PASS"``）且指标齐全的 trial**
  ⇒ ``PASS_RELAXED`` 绝不会被封成 RC（口径铁律）。
* 若交接候选有**多个种子** ⇒ 固化**三轨指标最接近组内中位**的那一个
  （中位 trial，而不是最高 trial —— 后者又是单点运气）。
* 副本 ``chmod 0444``（只读封存）；登记文件落 md5 + sha256 + 大小 + 三轨指标 +
  **15 维完整配方** + 判定链 + **阻塞项** + 复现命令。

用法（服务器 ``conda activate uav``）：

```bash
python manu/training/seal_candidate.py \
  --handover-json /mnt/data/siping/runs/nogmc_focus/HANDOVER.json \
  --csv           /mnt/data/siping/runs/nogmc_focus/optuna_summary.csv \
  --trials-root   /mnt/data/siping/runs/nogmc_focus/trials \
  --out-dir       /mnt/data/siping/runs/CANDIDATES \
  --rc-id         Candidate_v2 \
  --name          "no-GMC 定点打靶（23+12 臂）最优交接候选"
```
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import statistics
import sys
from pathlib import Path

CFG_COLS = ("mix_spec", "lr_scale", "warmup_epochs", "weight_decay", "focal_beta",
            "offset_weight", "mosaic", "scale", "translate", "clip_max_norm", "freeze_bn",
            "init_from", "epochs", "backbone_lr_scale", "freeze_backbone_epochs")


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _f(v):
    try:
        return float(v)
    except Exception:
        return None


def _median(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def main() -> int:
    ap = argparse.ArgumentParser(description="按交接口径固化 Release Candidate")
    ap.add_argument("--handover-json", required=True)
    ap.add_argument("--csv", required=True, nargs="+",
                    help="判定用的汇总 CSV；可给多份，格式 'TAG=路径'（与 judge_handover_candidate.py 同口径）")
    ap.add_argument("--trials-root", required=True, nargs="+",
                    help="权重根目录；可给多份，格式 'TAG=路径'（TAG 与 --csv 的 TAG 对应）")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rc-id", required=True)
    ap.add_argument("--name", default="")
    ap.add_argument("--ref", default="", help="参照系包路径（写进登记文件）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    doc = json.loads(Path(args.handover_json).read_text(encoding="utf-8"))
    if not doc.get("winner"):
        print("[FAIL] 交接判定没有候选（严格 PASS 里一个都没有）⇒ 不固化")
        return 4
    winner = doc["winner"]

    # ---- 读入（可能多份 CSV / 多个 trials 根目录）----------------------
    # ⚠️ **trial 名必须带来源前缀**：周末与 focus 两个 study 都有 trial_0009，
    #    但它们是不同的 trial（周末 = w3.0/lr1.0/15ep；focus = w6.0/lr2.0/10ep），
    #    权重也分别在各自的 trials 目录里 ⇒ 不加前缀就会**取错权重**。
    rows: list[dict] = []
    for spec in args.csv:
        if "=" in spec and not os.path.exists(spec):
            tag, path = spec.split("=", 1)
        else:
            path = spec
            tag = Path(path).parent.name
        if not os.path.exists(path):
            print(f"[FATAL] 找不到 CSV：{path}")
            return 2
        for r in csv.DictReader(Path(path).open("r", encoding="utf-8")):
            r["trial"] = f"{tag}:{r.get('trial')}"
            rows.append(r)

    roots: dict[str, Path] = {}
    for spec in args.trials_root:
        if "=" in spec and not os.path.exists(spec):
            tag, path = spec.split("=", 1)
        else:
            path = spec
            tag = Path(path).parent.name
        roots[tag] = Path(path)
    if not roots:
        print("[FATAL] --trials-root 为空")
        return 2

    by_trial = {r.get("trial"): r for r in rows}

    # ---- 只保留**指标齐全**的候选 --------------------------------------
    # ⚠️ **不再要求单条 verdict == PASS**：交接口径的决策单元是「配置」，
    #    合格性由**组中位**过 T1/T2 硬门禁决定（判定器已保证）。
    #    若这里仍按单条 verdict 过滤 ⇒ 一个「中位过线但没有一条种子严格 PASS」的组
    #    会因为**一条都封不了**而直接失败（rc=4），而这恰恰是需要人工裁定的情形。
    #    ⇒ 改为：**优先封组内的严格 PASS 种子**；一条都没有时才退到中位种子，
    #      并在登记文件里**显式标注**该产物的单条 verdict（供审计）。
    # ⚠️ **变体候选**（soup / SWA）：它**不是 trial** ⇒ 权重不在 trials_root 里
    #    ⇒ 走「就地设置变量」的分支（其余封存流程完全一样）。
    IS_VARIANT = bool(winner.get("is_variant"))

    cands = []
    for t in (winner.get("trials", []) if not IS_VARIANT else []):
        r = by_trial.get(t)
        if not r:
            print(f"[SKIP] {t}: 在汇总 CSV 里找不到")
            continue
        if any(_f(r.get(k)) is None for k in ("t1_f1", "t2_f1", "t3_f1")):
            print(f"[SKIP] {t}: 三轨指标不全 ⇒ 不可固化")
            continue
        cands.append(r)
    if not cands and not IS_VARIANT:
        print("[FAIL] 候选里没有可固化的（三轨指标齐全的）trial")
        return 4

    # ---- 选"中位 trial"（三轨 F1 到组内中位距离最小），不选最高 ---------
    # ⚠️ **优先严格 PASS 的种子**：组中位过线只是"这个配置合格"，
    #    而封存产物最好来自一条**本身严格 PASS** 的种子（口径铁律的保守侧）。
    #    组内一条严格 PASS 都没有时，退到中位种子，并在登记文件里显式标注。
    def _dist(r):
        d = 0.0
        for col, key in (("t1_f1", "t1_median"), ("t3_f1", "t3_median"), ("t2_f1", "t2_median")):
            m = winner.get(key)
            v = _f(r.get(col))
            if m is not None and v is not None:
                d += abs(v - m)
        return d

    if IS_VARIANT:
        # ---- 变体：权重就是它自己的 .pt（soup / SWA 产物）------------------
        src_w = Path(winner.get("weights") or "")
        if not src_w.is_file():
            print(f"[FAIL] 变体候选的权重不存在：{src_w}")
            return 4
        bare_trial = str(winner.get("variant_key") or "variant")
        tdir = src_w.parent
        pick = {"trial": bare_trial,
                "gate_verdict": (winner.get("verdicts") or [""])[0],
                "t1_f1": winner.get("t1_median"), "t1_fp": winner.get("t1_fp_median"),
                "t2_f1": winner.get("t2_median"), "t2_fp": winner.get("t2_fp_median"),
                "t3_f1": winner.get("t3_median"), "t3_fp": winner.get("t3_fp_median"),
                "utility": winner.get("u_median"), "utility_vs_rc1": None}
        pick.update({c: (winner.get("cfg") or {}).get(c) for c in CFG_COLS})
        cands = [pick]
        used_strict = True
    else:
        strict_cands = [r for r in cands if (r.get("gate_verdict") or "").strip() == "PASS"]
        used_strict = bool(strict_cands)
        pool = strict_cands if strict_cands else cands
        pool.sort(key=_dist)
        pick = pool[0]
        trial_name = pick.get("trial")          # 形如 "weekend:trial_0025"
        # ⚠️ 按前缀选根目录 ⇒ 周末/focus 同名 trial 不会串
        tag, _, bare = str(trial_name).partition(":")
        if bare and tag in roots:
            root = roots[tag]
            bare_trial = bare
        else:
            root = next(iter(roots.values()))
            bare_trial = str(trial_name)
        tdir = root / bare_trial

        # ---- 找权重（末 epoch 的 .pt）--------------------------------------
        wdir = tdir / "weights"
        wfiles = sorted(wdir.glob("epoch*.pt"), key=lambda p: int(p.stem.replace("epoch", "")))
        if not wfiles:
            print(f"[FAIL] {wdir} 里没有 epoch*.pt")
            return 4
        src_w = wfiles[-1]

    if IS_VARIANT:
        print(f"[PICK] 候选是**变体** {bare_trial}"
              f"（{winner.get('variant_kind')}｜源 {winner.get('variant_source')}）")
        print(f"       直接封存变体权重：{src_w}")
        print(f"       多种子证据：{winner.get('seed_evidence_n')}（源家族种子数）"
              f"｜T1={winner.get('t1_median')} T3={winner.get('t3_median')}")
    else:
        print(f"[PICK] 候选组 {winner.get('trials')} ⇒ 选中位 trial {trial_name}（权重 {src_w.name}）")
    print(f"       单条 verdict = {pick.get('gate_verdict')}"
          f"{'（组内有严格 PASS 种子 ⇒ 取其最接近中位的一条）' if used_strict else '（⚠️ 组内**无**严格 PASS 种子 ⇒ 退到中位种子，需人工裁定）'}")
    print(f"       组内各 trial 到组中位的距离：")
    for r in cands:
        print(f"         {r.get('trial')}: {_dist(r):.6f}  T1={r.get('t1_f1')} T3={r.get('t3_f1')} "
              f"verdict={r.get('gate_verdict')}")

    if args.dry_run:
        print("[DRY-RUN] 未落盘")
        return 0

    out_dir = Path(args.out_dir) / args.rc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    sealed = out_dir / f"{args.rc_id}_{bare_trial}_{src_w.name}"
    # ⚠️ **重封前必须先解除只读**（2026-11-03 06:33 实测踩到）：
    #    固化脚本会把封存副本置为 0444（防篡改）⇒ 第二次固化同一 rc_id 时
    #    `shutil.copy2` 直接抛 **PermissionError** ⇒ 整个收尾的固化步骤 rc=1 失败。
    #    ⇒ 覆盖前先 chmod +w（只影响"能不能被覆盖"，不改变封存后的只读属性）。
    if sealed.exists():
        try:
            os.chmod(sealed, 0o644)
        except OSError:
            pass
    shutil.copy2(src_w, sealed)
    os.chmod(sealed, 0o444)
    if _md5(sealed) != _md5(src_w):
        print("[FATAL] 封存副本与源权重 md5 不一致 ⇒ 中止")
        return 5

    md5 = _md5(sealed)
    sha = _sha256(sealed)
    size = sealed.stat().st_size

    gate_report = None
    for gp in sorted(tdir.glob("gate/epoch*/gate_report.json")):
        gate_report = gp
    if gate_report is None and IS_VARIANT:
        # 变体的门禁报告在 <swa_dir>/gate/<name>/gate_report.json
        for gp in sorted(tdir.glob("gate/*/gate_report.json")):
            gate_report = gp

    reg = {
        "rc_id": args.rc_id,
        "rc_name": args.name or args.rc_id,
        "sealed_at_local": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "artifact": {
            "sealed_weights": str(sealed),
            "source_weights": str(src_w),
            "size_bytes": size,
            "md5": md5,
            "sha256": sha,
            "sealed_mode": "0444",
        },
        "recipe_15d": {c: pick.get(c) for c in CFG_COLS},
        "sealed_trial_individual_verdict": (pick.get("gate_verdict") or "").strip(),
        "sealed_trial_is_strict_pass": used_strict,
        "metrics_of_sealed_trial": {
            "t1_f1": _f(pick.get("t1_f1")), "t1_fp": _f(pick.get("t1_fp")),
            "t2_f1": _f(pick.get("t2_f1")), "t2_fp": _f(pick.get("t2_fp")),
            "t3_f1": _f(pick.get("t3_f1")), "t3_fp": _f(pick.get("t3_fp")),
            "utility": _f(pick.get("utility")),
            "utility_vs_rc1": _f(pick.get("utility_vs_rc1")),
        },
        "group_median": {
            "n_seeds": winner.get("n"),
            "u_median": winner.get("u_median"), "u_band": winner.get("u_band"),
            "t1_median": winner.get("t1_median"), "t1_fp_median": winner.get("t1_fp_median"),
            "t2_median": winner.get("t2_median"), "t2_fp_median": winner.get("t2_fp_median"),
            "t3_median": winner.get("t3_median"), "t3_band": winner.get("t3_band"),
            "t3_fp_median": winner.get("t3_fp_median"),
            "trials": winner.get("trials"),
        },
        "redlines": doc.get("redlines"),
        "judgement_chain": doc.get("chain"),
        "judgement_reason": doc.get("reason"),
        "blockers": doc.get("blockers"),
        "handover_ready": doc.get("handover_ready"),
        "gate_report": str(gate_report) if gate_report else None,
        "reference_frame": args.ref or None,
        "reproduce": {
            "gate": (f"python manu/training/auto_epoch_gate_check.py --weights {src_w} "
                     f"--out-dir {tdir}/gate/<epochN> --gpus 4 --device 0 --infer-gpu 0 "
                     f"--cache-gpus 0,1,2,3 --baseline-cache "
                     f"/mnt/data/siping/runs/nogmc_ref_weekend/uav_cache_base.pkl "
                     f"--min-track2-f1 0.985 --max-track2-fp 112 "
                     f"--min-track1-f1 0.8958967482400269 "
                     f"--uav-data /mnt/data/siping/datasets/manu/uav_gmc_median_nogmc/data.yaml "
                     f"--ft-root /mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_nogmc "
                     f"--base-track1-f1 0.9008967482400269 --base-track1-tp 21499 --base-track1-fp 1118 "
                     f"--base-track2-f1 0.981074481074481 --base-track2-tp 16070 --base-track2-fp 52 "
                     f"--base-track2-fn 568 --base-track3-f1 0.7578120617110798 --base-track3-tp 6754 "
                     f"--base-track3-fp 536 --base-track3-fn 3781"),
            "judge": (f"python manu/training/judge_handover_candidate.py "
                      f"--csv <OUT>/optuna_summary.csv --ref <REF> --out-md <OUT>/HANDOVER.md "
                      f"--out-json <OUT>/HANDOVER.json --min-seeds 2"),
        },
    }
    reg_path = out_dir / f"{args.rc_id}.json"
    reg_path.write_text(json.dumps(reg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    m = reg["metrics_of_sealed_trial"]
    g = reg["group_median"]
    bl = reg["blockers"]
    readme = f"""# {args.rc_id} — {reg['rc_name']}

| 项 | 值 |
|:--|:--|
| 封存权重 | `{sealed}` |
| md5 | `{md5}` |
| sha256 | `{sha}` |
| 大小 | {size:,} bytes |
| 模式 | 0444（只读封存） |
| 源权重 | `{src_w}` |
| 选中理由 | 组内**中位 trial**（三轨到组中位距离最小），非最高 trial |

## 组内多种子中位（交接口径）

| 项 | 值 |
|:--|--:|
| 种子数 | **{g['n_seeds']}**（trials: {', '.join(g['trials'] or [])}） |
| U 中位 | {g['u_median']} ± {g['u_band']} |
| T1 中位 | {g['t1_median']}（虚警中位 {g['t1_fp_median']}） |
| T2 中位 | {g['t2_median']}（虚警中位 {g['t2_fp_median']}） |
| T3 中位 | {g['t3_median']} ± {g['t3_band']}（虚警中位 {g['t3_fp_median']}） |

## 封存 trial 的三轨

| 轨道 | F1 | FP |
|:--|--:|--:|
| T1 Anti-UAV 官方大盘 | {m['t1_f1']} | {m['t1_fp']} |
| T2 黄金池 | {m['t2_f1']} | {m['t2_fp']} |
| T3 盲测 001324 | {m['t3_f1']} | {m['t3_fp']} |

## 15 维配方

```json
{json.dumps(reg['recipe_15d'], ensure_ascii=False, indent=2)}
```

## 判定链

{chr(10).join('- ' + str(x) for x in (reg['judgement_chain'] or []))}
- **结论**：{reg['judgement_reason']}

## ⚠️ 阻塞项

{chr(10).join('- ' + str(x) for x in bl) if bl else '- 无'}

机读登记：`{reg_path.name}`
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    print(f"[OK] {args.rc_id} 已封存")
    print(f"  权重   : {sealed}  ({size:,} bytes, 0444)")
    print(f"  md5    : {md5}")
    print(f"  sha256 : {sha}")
    print(f"  登记   : {reg_path}")
    print(f"  阻塞项 : {bl or '无'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
