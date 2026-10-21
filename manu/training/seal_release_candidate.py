#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Release Candidate 封存工具（RC1）。

用途
----
把一次微调运行中的某个 epoch 权重**固化为不可变的 Release Candidate**，
并产出一份机读的 `RC1.json` 登记文件（权重指纹 + 三轨指标 + 选择依据 + 复现命令），
供后续微调实验（冻结 BN / L2-SP / 新配比 …）直接对账。

设计原则
--------
1. **数字不手抄**：所有指标一律从 `gate/gate_history.jsonl` 读出后程序化提取，
   杜绝转写误差（铁律九：结论须带可复制命令与产物路径）。
2. **封存副本只读**：`chmod 0444`，原始 `weights/epochN.pt` 保持可写
   （训练脚本重跑会覆盖它，RC1 副本才是不可变基准）。
3. **登记文件跨机可核**：同一份 `RC1.json` 落地到服务器与本地仓库，md5 双侧比对。

用法（执行机：服务器 `ssh -p 32222 huangzhe@192.168.99.40`，`conda activate uav`，
项目根 `/tmp/pycharm_project_10ae9e2e`）：

```bash
python manu/training/seal_release_candidate.py \
  --run-dir runs/mixed_ft_route_b --epoch 3 \
  --out-dir runs/mixed_ft_route_b/RC1
```
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

# 路线 B（1:3.09 全帧对等混训）Epoch 3 = RC1 的选择依据。
# 全部数字可由 gate_history.jsonl 复核，此处仅承载**定性判断**。
RC1_SELECTION_RATIONALE = {
    "verdict": "Epoch 3 是路线 B 首轮试跑（5 epoch）全周期三轨综合最优的帕累托点（Sweet Spot）",
    "quantitative_basis": [
        "Track1 Anti-UAV 官方大盘：RC1 峰值 FP=941，为 Ep1~Ep5 全场最低（Ep1=1232/Ep2=988/Ep3=941/Ep4=1057/Ep5=1023），"
        "比基线 FP=1014 还少 73；Ep4/Ep5 已反弹至 1057/1023，Ep5 F1 跌破 0.901（0.900501）= 源域散架征兆",
        "Track2 黄金池 7 条：RC1 F1=0.9989791016094163 为全周期峰值（Ep1=0.998769/Ep2=0.998889/Ep3=0.998979/"
        "Ep4=0.998859/Ep5=0.998949），FN=3（Ep5=2，差 1 帧，无实质差异），FP=31 为 Ep1~Ep5 最低",
        "Track3 盲测 001324：RC1 FP=461 为全周期最低（Ep1=589/Ep2=531/Ep3=461/Ep4=517/Ep5=502），净砍 129 个误检；"
        "RC1 已完成 Ep5 涨幅的 91.4%（(0.8716843671438099-0.8373308251032032)/(0.8749235474006115-0.8373308251032032)=0.9139）",
        "Ep3→Ep5 边际收益递减：额外 2 个完整 epoch（约 40 分钟算力）仅换来 Track3 F1 +0.0032，"
        "代价是 Track1 再损 0.14pp 且全图虚警由 461 回升到 502 ⇒ 判定为边际收益递减/轻度过拟合",
    ],
    "locked_instead_of": "epoch5.pt（目标域 F1 最高但三轨虚警全面抬头、源域 F1 跌破 0.901）",
    "caveats": [
        "🔴 原始红线 Track1 F1 ≥ 0.905736（去 P0 冻结基线）在 Ep1~Ep5 **全部 5 个 epoch 均未通过**，"
        "RC1 自身的 Track1 F1=0.901917808219178 同样判负（veto_failures=[track1_uav]）。"
        "RC1 的产生依据是**帕累托最优 + 虚警最低**，不是**通过原始 Track1 红线**。"
        "任何把 RC1 表述为『通过三轨门禁』的说法都是错的。",
        "⚠️ Track1 峰值门限漂移：基线/Ep1 峰值在 th=0.25，Ep2~Ep5 峰值在 th=0.30。"
        "跨 epoch 的 FP/F1 对比是『各模型自身最优门限下』的比较（标准做法），"
        "但门限本身右移了 0.05，解读时应连同 delta.negative_bands=7/15 一并看。",
        "⚠️ 三轨评测口径天生不对称（Track1 中心点 8px 距离匹配 / Track2-3 框内包含匹配），"
        "三轨 F1 **不可横向比较**，只做轨道内纵向对比（详见 memory.md §四十三 §2）。",
        "⚠️ Track2 的「FP 增量 ≤ 10」红线的唯一攻击点是序列 000703（基线 FP 4 → Ep5 FP 29，Δ+25），"
        "RC1 之后须定点攻它（把 000703 失效机理并入训练集负样本）。",
    ],
    "future_use": (
        "后续任何进一步微调（冻结 BN、L2-SP 正则、重新配比、加负样本等）**一律以本 RC1 为对账基准**，"
        "不再回看 epoch5。比较时三轨分别纵向对比，Track1 需同时看 15 档门限曲线的整体位移"
        "（禁止用 --expect-* 基线复现语义，那套参数对任何微调权重都必然 FAIL）。"
    ),
}


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            blk = fh.read(chunk)
            if not blk:
                break
            h.update(blk)
    return h.hexdigest()


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        while True:
            blk = fh.read(chunk)
            if not blk:
                break
            h.update(blk)
    return h.hexdigest()


def _load_history(run_dir: Path) -> list[dict]:
    hist = run_dir / "gate" / "gate_history.jsonl"
    if not hist.is_file():
        raise SystemExit(f"[FATAL] 找不到门禁历史：{hist}")
    rows = []
    with hist.open("r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    if not rows:
        raise SystemExit(f"[FATAL] 门禁历史为空：{hist}")
    return rows


def _track_summary(entry: dict) -> dict:
    """把 gate_history 单条记录压成三轨可读摘要。"""
    t = entry["tracks"]
    t1 = t["track1_uav"]
    t2 = t["track2_golden"]
    t3 = t["track3_blind"]
    return {
        "track1_antiuav_official": {
            "peak_th": t1["finetune_peak"]["th"],
            "f1": t1["finetune_peak"]["f1"],
            "precision": t1["finetune_peak"]["precision"],
            "recall": t1["finetune_peak"]["recall"],
            "tp": t1["finetune_peak"]["tp"],
            "fp": t1["finetune_peak"]["fp"],
            "fn": t1["finetune_peak"]["fn"],
            "baseline": {
                "peak_th": t1["baseline_peak"]["th"],
                "f1": t1["baseline_peak"]["f1"],
                "tp": t1["baseline_peak"]["tp"],
                "fp": t1["baseline_peak"]["fp"],
            },
            "delta_peak_f1": t1["delta"]["peak_f1"],
            "delta_mean_f1": t1["delta"]["mean_f1"],
            "delta_min_f1": t1["delta"]["min_f1"],
            "delta_max_f1": t1["delta"]["max_f1"],
            "negative_bands": t1["delta"]["negative_bands"],
            "bands": t1["delta"]["bands"],
            "min_f1_required": t1["min_f1_required"],
        },
        "track2_golden_pool_7seq": {
            "f1": t2["f1"],
            "tp": t2["tp"],
            "fp": t2["fp"],
            "fn": t2["fn"],
            "gt": t2["baseline"]["gt"],
            "frames": t2["baseline"]["frames"],
            "baseline": {k: t2["baseline"][k] for k in ("f1", "tp", "fp", "fn")},
            "delta_f1": t2["delta_f1"],
            "delta_fp": t2["delta_fp"],
            "delta_fn": t2["delta_fn"],
        },
        "track3_blind_001324": {
            "f1": t3["f1"],
            "tp": t3["tp"],
            "fp": t3["fp"],
            "fn": t3["fn"],
            "gt": t3["baseline"]["gt"],
            "frames": t3["baseline"]["frames"],
            "baseline": {k: t3["baseline"][k] for k in ("f1", "tp", "fp", "fn")},
            "delta_f1": t3["delta_f1"],
            "delta_fp": t3["delta_fp"],
            "delta_fn": t3["delta_fn"],
        },
    }


def _curve(history: list[dict]) -> list[dict]:
    """全周期三轨曲线（Ep0=基线 … Ep5），供后续实验横向对账。"""
    out = []
    for e in history:
        s = _track_summary(e)
        out.append(
            {
                "epoch": e["epoch"],
                "weights": e["weights"],
                "gate_verdict": e["verdict"],
                "gate_rc": e["rc"],
                "track1_f1": s["track1_antiuav_official"]["f1"],
                "track1_th": s["track1_antiuav_official"]["peak_th"],
                "track1_tp": s["track1_antiuav_official"]["tp"],
                "track1_fp": s["track1_antiuav_official"]["fp"],
                "track2_f1": s["track2_golden_pool_7seq"]["f1"],
                "track2_tp": s["track2_golden_pool_7seq"]["tp"],
                "track2_fp": s["track2_golden_pool_7seq"]["fp"],
                "track2_fn": s["track2_golden_pool_7seq"]["fn"],
                "track3_f1": s["track3_blind_001324"]["f1"],
                "track3_tp": s["track3_blind_001324"]["tp"],
                "track3_fp": s["track3_blind_001324"]["fp"],
                "track3_fn": s["track3_blind_001324"]["fn"],
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="封存 Release Candidate 并生成机读登记")
    ap.add_argument("--run-dir", required=True, help="微调运行目录，如 runs/mixed_ft_route_b")
    ap.add_argument("--epoch", type=int, required=True, help="要封存的 epoch 序号")
    ap.add_argument("--out-dir", default=None, help="封存输出目录（默认 <run-dir>/RC1）")
    ap.add_argument("--rc-id", default="RC1", help="候选编号，默认 RC1")
    ap.add_argument("--run-name", default="路线 B 全帧对等混训（1:3.09）首轮试跑", help="运行名称")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / args.rc_id
    if not run_dir.is_dir():
        raise SystemExit(f"[FATAL] 运行目录不存在：{run_dir}")

    history = _load_history(run_dir)
    entry = next((h for h in history if h["epoch"] == args.epoch), None)
    if entry is None:
        raise SystemExit(f"[FATAL] gate_history 中无 epoch={args.epoch}（现有 {sorted(h['epoch'] for h in history)}）")

    src_w = Path(entry["weights"])
    if not src_w.is_absolute():
        cand = run_dir / "weights" / src_w.name
        src_w = cand if cand.is_file() else Path(src_w)
    if not src_w.is_file():
        raise SystemExit(f"[FATAL] 权重不存在：{src_w}")

    out_dir.mkdir(parents=True, exist_ok=True)
    sealed = out_dir / src_w.name
    shutil.copy2(src_w, sealed)
    os.chmod(sealed, 0o444)

    md5 = _md5(sealed)
    sha = _sha256(sealed)
    size = sealed.stat().st_size
    if md5 != _md5(src_w):
        raise SystemExit("[FATAL] 封存副本与源权重 md5 不一致，封存中止")

    gates = None
    rep = run_dir / "gate" / f"epoch{args.epoch}" / "gate_report.json"
    if rep.is_file():
        with rep.open("r", encoding="utf-8") as fh:
            gates = json.load(fh).get("gates")

    doc = {
        "rc_id": args.rc_id,
        "rc_name": f"{args.run_name} · Epoch {args.epoch}",
        "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "created_local": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run": {
            "run_name": args.run_name,
            "run_dir": str(run_dir),
            "epoch": args.epoch,
            "fuse_mode_used": "record（本轮按用户指令跑满 5 轮，熔断判定照常记录）",
            "note": "epoch5 未被选为 RC1：目标域 F1 略高但三轨虚警全面抬头、源域 F1 跌破 0.901",
        },
        "artifact": {
            "source_weights": str(src_w),
            "sealed_weights": str(sealed),
            "sealed_filename": sealed.name,
            "size_bytes": size,
            "md5": md5,
            "sha256": sha,
            "sealed_mode": "0444 (read-only)",
            "host": "server /tmp/pycharm_project_10ae9e2e",
        },
        "metrics": _track_summary(entry),
        "gate": {
            "verdict": entry.get("verdict"),
            "rc": entry.get("rc"),
            "gates": gates,
            "veto_failures": [k for k, v in (gates or {}).items() if v.get("veto") and not v.get("pass")],
        },
        "selection_rationale": RC1_SELECTION_RATIONALE,
        "full_cycle_curve": _curve(history),
        "reproduce": {
            "gate_entry": (
                f"bash manu/training/auto_epoch_gate_check.sh {src_w} {args.epoch}"
            ),
            "gate_entry_flagform": (
                f"bash manu/training/auto_epoch_gate_check.sh --weights {src_w} --epoch {args.epoch}"
            ),
            "baseline_repro": (
                "bash manu/data/validate_longquan_ft_dataset.sh "
                "--weights runs/optuna_p0_nas/trial_0474/weights/best.pt "
                "--out-dir runs/gmc_eval/baseline_repro --strict-gate 1"
            ),
            "weight_delta_check": (
                f"python manu/training/verify_finetune_delta.py --weights {src_w} "
                "--source runs/optuna_p0_nas/trial_0474/weights/best.pt"
            ),
            "per_sequence_attribution": (
                f"python manu/training/analyze_per_sequence.py --run-dir {run_dir} "
                "--epochs 0,1,2,3,4,5 --out-json " + str(run_dir / "per_sequence_attribution.json")
            ),
        },
        "sources": {
            "gate_history": str(run_dir / "gate" / "gate_history.jsonl"),
            "gate_report": str(rep) if rep.is_file() else None,
            "memory_main": "manu/memory/memory.md §四十六",
            "memory_compact": "manu/memory/memory_compact.md §32",
        },
    }

    out_json = out_dir / f"{args.rc_id}.json"
    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=2, sort_keys=False)
        fh.write("\n")

    m = doc["metrics"]
    readme = f"""# {args.rc_id} — {doc['rc_name']}

| 项 | 值 |
|:--|:--|
| 封存权重 | `{sealed}` |
| md5 | `{md5}` |
| sha256 | `{sha}` |
| 大小 | {size:,} bytes |
| 模式 | 0444（只读封存） |
| 源权重 | `{src_w}` |
| 门禁判定 | **{entry.get('verdict')}**（rc={entry.get('rc')}，veto 失败项：{doc['gate']['veto_failures'] or '无'}） |

## 三轨指标（RC1 = Epoch {args.epoch}）

| 轨道 | F1 | TP | FP | FN | ΔF1 vs 基线 |
|:--|--:|--:|--:|--:|--:|
| Track1 Anti-UAV 官方大盘（峰值 th={m['track1_antiuav_official']['peak_th']}） | {m['track1_antiuav_official']['f1']:.6f} | {m['track1_antiuav_official']['tp']} | {m['track1_antiuav_official']['fp']} | {m['track1_antiuav_official']['fn']} | {m['track1_antiuav_official']['delta_peak_f1']:+.6f} |
| Track2 黄金池 7 条 | {m['track2_golden_pool_7seq']['f1']:.6f} | {m['track2_golden_pool_7seq']['tp']} | {m['track2_golden_pool_7seq']['fp']} | {m['track2_golden_pool_7seq']['fn']} | {m['track2_golden_pool_7seq']['delta_f1']:+.6f} |
| Track3 盲测 001324 | {m['track3_blind_001324']['f1']:.6f} | {m['track3_blind_001324']['tp']} | {m['track3_blind_001324']['fp']} | {m['track3_blind_001324']['fn']} | {m['track3_blind_001324']['delta_f1']:+.6f} |

## 为什么是 Epoch {args.epoch} 而不是 Epoch 5

{chr(10).join('- ' + x for x in RC1_SELECTION_RATIONALE['quantitative_basis'])}

**锁定 {args.epoch} 而非 5**：`{RC1_SELECTION_RATIONALE['locked_instead_of']}`

## ⚠️ 必读限定

{chr(10).join('- ' + x for x in RC1_SELECTION_RATIONALE['caveats'])}

## 后续用法

{RC1_SELECTION_RATIONALE['future_use']}

## 复现命令（执行机：服务器 `ssh -p 32222 huangzhe@192.168.99.40`，`conda activate uav`，根目录 `/tmp/pycharm_project_10ae9e2e`）

```bash
{entry['weights']} → 门禁
{doc['reproduce']['gate_entry']}

# 逐序列归因
{doc['reproduce']['per_sequence_attribution']}
```

登记文件的机读版本：`{out_json.name}`
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    print(f"[OK] {args.rc_id} 已封存")
    print(f"  权重   : {sealed}  ({size:,} bytes, mode 0444)")
    print(f"  md5    : {md5}")
    print(f"  sha256 : {sha}")
    print(f"  登记   : {out_json}")
    print(f"  说明   : {out_dir / 'README.md'}")
    print(f"  门禁   : {entry.get('verdict')} (rc={entry.get('rc')}) veto={doc['gate']['veto_failures'] or '无'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
