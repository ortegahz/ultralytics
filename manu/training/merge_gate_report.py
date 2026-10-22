#!/usr/bin/env python
# ==============================================================================
# 门禁报告**补测合并**工具
#
# 为什么需要
# ----------
# `auto_epoch_gate_check.py` 的三轨里，**Track1（Anti-UAV 推理）最先跑、也最吃显存**。
# 当门禁与正在训练的 trial 同卡时，Track1 可能 CUDA OOM ⇒ `gate_report.json` 里
# `tracks.track1_uav.finetune_peak.f1` 为 null，而 Track2/Track3（龙泉山）仍然正常跑完。
#
# 此时该报告**不可用但也没全废**：只缺 Track1 的数值。
# 本工具把"单独重跑 Track1"的结果合并回主报告，使整份报告重新可用，
# 同时把**残缺原件另存为 `*.incomplete.json` 留档**（可审计，不掩盖事故现场）。
#
# 用法（执行机：服务器）：
#   # 1) 先单独补跑 Track1（复用已建好的龙泉山 val 缓存，--skip-longquan）
#   python manu/training/auto_epoch_gate_check.py \
#     --weights <trial>/weights/epoch3.pt --out-dir <trial>/gate/epoch3_retry \
#     --gpus 1 --device <gpu> --infer-gpu <gpu> --cache-gpus <gpu> \
#     --baseline-cache runs/mixed_ft_route_b/baseline/uav_cache_deP0.pkl \
#     --skip-longquan --infer-batch 4
#   # 2) 合并
#   python manu/training/merge_gate_report.py \
#     --main <trial>/gate/epoch3/gate_report.json \
#     --retry <trial>/gate/epoch3_retry/gate_report.json
#
# 退出码 0 = 合并成功；非 0 = 合并失败（且**不会**改动主报告）。
# ==============================================================================
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def _load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _has_track1(rep: dict) -> bool:
    t1 = (rep.get("tracks") or {}).get("track1_uav") or {}
    peak = t1.get("finetune_peak") or {}
    return peak.get("f1") is not None


def merge(main_path: Path, retry_path: Path, backup: bool = True) -> tuple[bool, str]:
    """把 retry 报告里的 Track1 并入 main 报告。返回 (是否成功, 说明)。"""
    if not main_path.is_file():
        return False, f"主报告不存在：{main_path}"
    if not retry_path.is_file():
        return False, f"补测报告不存在：{retry_path}"

    main_rep = _load(main_path)
    retry_rep = _load(retry_path)

    if _has_track1(main_rep):
        return False, "主报告的 Track1 本来就是完整的，无需合并（不会覆盖已有数据）"
    if not _has_track1(retry_rep):
        return False, "补测报告的 Track1 仍不完整（推理可能又失败了），合并中止"

    main_t1 = ((main_rep.get("tracks") or {}).get("track1_uav") or {})
    retry_t1 = ((retry_rep.get("tracks") or {}).get("track1_uav") or {})

    # 保留主报告原有的 status/delta 等结构，只补齐 finetune_peak / 峰值相关字段
    merged_t1 = dict(main_t1)
    for key, value in retry_t1.items():
        if key in ("finetune_peak", "status", "rc", "baseline_peak", "delta",
                   "min_f1_required", "infer_log"):
            merged_t1[key] = value
    main_rep.setdefault("tracks", {})["track1_uav"] = merged_t1

    # 重新计算 verdict：Track1 的 pass 要按合并后的 finetune_peak 判定
    gates = main_rep.setdefault("gates", {})
    g1 = gates.setdefault("track1_uav", {})
    peak = merged_t1.get("finetune_peak") or {}
    base_peak = merged_t1.get("baseline_peak") or {}
    min_req = merged_t1.get("min_f1_required", g1.get("value"))
    f1 = peak.get("f1")
    if f1 is not None and min_req is not None:
        g1["value"] = f1
        g1["pass"] = bool(f1 >= min_req)
        g1["veto"] = True
        g1["rule"] = g1.get("rule") or f"F1 ≥ {min_req:.6f}（去 P0 冻结基线）"
        if base_peak.get("f1") is not None:
            g1["delta_f1"] = f1 - base_peak["f1"]
    # veto_failures / verdict 重算
    veto = [name for name, g in gates.items()
            if g.get("veto") and g.get("pass") is False]
    main_rep["veto_failures"] = veto
    if veto:
        main_rep["verdict"] = "FAIL"
    elif main_rep.get("verdict") == "FAIL" and not main_rep.get("observation_warnings"):
        main_rep["verdict"] = "PASS"
    main_rep["merged_from_retry"] = str(retry_path)
    main_rep["merge_note"] = (
        "Track1 首次推理失败（通常是 CUDA OOM），已用 --skip-longquan 单独补跑并合并；"
        f"残缺原件见 {main_path.with_suffix('.incomplete.json') if backup else '(未备份)'}"
    )

    if backup:
        bak = main_path.with_suffix(".incomplete.json")
        shutil.copy2(main_path, bak)

    main_path.write_text(json.dumps(main_rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return True, (
        f"合并成功：Track1 F1={peak.get('f1')} FP={peak.get('fp')} "
        f"th={peak.get('th')} | verdict={main_rep.get('verdict')} | veto={veto}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="合并门禁补测的 Track1 结果回主报告")
    ap.add_argument("--main", required=True, help="主 gate_report.json（含残缺的 Track1）")
    ap.add_argument("--retry", required=True, help="补测产生的 gate_report.json（Track1 完整）")
    ap.add_argument("--no-backup", action="store_true", help="不备份残缺原件（不建议）")
    args = ap.parse_args()

    ok, msg = merge(Path(args.main), Path(args.retry), backup=not args.no_backup)
    print(f"[{'OK' if ok else 'FAIL'}] {msg}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
