#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""基线升轨工具（2026-11-11 用户裁决）：trial_9863 → Baseline_v2.0。

作用（幂等，可反复执行）：
  ① 在 nano_search_ledger.json 顶层写入 baseline_registry / baseline_v2 / beat_to_pass；
  ② 给 trial_9863 臂打 is_baseline_v2=true 与 status；
  ③ 把 trial_0031 / trial_0080 记入 baseline_refs（**不进 arms**，避免污染搜索臂汇总表）；
  ④ 打印自检结果并回读校验。

用法（服务器）：
  /home/huangzhe/anaconda3/envs/uav/bin/python manu/pipeline/npu/promote_baseline.py
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nano_search as ns  # noqa: E402

LEDGER = ns.LEDGER


BOARD_SESSION_DIRS = [
    ("session1", "/mnt/data/siping/runs/npu20ms/board_measure_20261111/hm_speed_results.txt"),
    ("session2", "/mnt/data/siping/runs/npu20ms/board_measure_20261111_s2/hm_speed_results.txt"),
]
TAG_RE = re.compile(r"rknn_heatmap_smoke :: (\S+) =====")
MS_RE = re.compile(r"rknn_run\s+avg\s+([\d.]+) ms")
# tag -> trial（如 arm_n9863_ep5_int8 => 9863）
ARM_TRIAL_RE = re.compile(r"arm_n(\d+)_ep(\d+)")


def parse_board_session(path: str) -> dict:
    """从 hm_speed_results.txt 解析每个 tag 的 rknn_run avg ms。

    版式：health check 段 → `===== rknn_heatmap_smoke :: TAG =====` → 该 tag 的
    rknn_run avg → 下一个 health check 段 ⇒ **tag 后的第一个 rknn_run 才属于它**。
    """
    out: dict = {}
    if not os.path.isfile(path):
        return out
    tag = None
    for line in open(path, encoding="utf-8", errors="ignore"):
        m = TAG_RE.search(line)
        if m:
            tag = m.group(1)
            continue
        m = MS_RE.search(line)
        if m and tag and tag not in out:
            out[tag] = float(m.group(1))
            tag = None  # 该 tag 已取值；下一条 rknn_run 属于下一个 health check
    return out


def import_board_sessions(led: dict) -> dict:
    """把两次板卡会话写进 ledger，并给对应臂补 board_ms。"""
    sess: dict = {}
    for key, path in BOARD_SESSION_DIRS:
        arms = parse_board_session(path)
        if arms:
            sess[key] = {"dir": os.path.dirname(path), "src": path, "arms": arms}
    if not sess:
        return {}
    led["board_sessions"] = [sess]
    arms_map = led.setdefault("arms", {})
    # session1 先写、session2 后写（覆盖）⇒ session2（含 t31 同会话参照）为准
    for key in ("session1", "session2"):
        s = sess.get(key)
        if not s:
            continue
        # 别名：台账里板卡 tag 带 "arm_" 前缀（arm_t31_silu_int8），
        # 而 phase_report / beat_report 查的是去前缀名（t31_silu_int8）
        # ⇒ 两个键都写，否则同会话参照永远取不到（真实踩过的坑）
        for tag in list(s["arms"]):
            if tag.startswith("arm_"):
                s["arms"].setdefault(tag[4:], s["arms"][tag])
        for tag, ms in list(s["arms"].items()):
            m = ARM_TRIAL_RE.search(tag)
            if not m:
                continue
            trial = int(m.group(1))
            st = arms_map.setdefault(str(trial), {"trial": trial})
            st["board_ms"] = ms
            st["board_session"] = key
            st["board_tag"] = tag
    return sess


def ensure_meta(led: dict) -> None:
    led["baseline_v2"] = ns.BASE_V2
    led["baseline_registry"] = ns.BASELINE_REGISTRY
    led["beat_to_pass"] = {
        "ref": "Baseline_v2.0(trial_9863)",
        "nano_min": ns.BASE_V2["nano"],
        "f1_min": ns.BASE_V2["f1"],
        "t1_min": ns.BASE_V2["t1"],
        "t2_fp_max": ns.BASE_V2["t2_fp"],
        "t3_min": ns.BASE_V2["t3"],
        "legacy_ref": "trial_0031",
        "legacy_nano_min": ns.BASELINE_NANO_FLOOR["0031"],
        "note": "后续所有探索统一以 Baseline_v2.0 的指标为 Beat-to-Pass 下限",
    }
    led.setdefault("baseline_refs", {})
    led["baseline_refs"]["0031"] = {
        "id": "trial_0031",
        "status": "SUPERSEDED / DEPRECATED BASELINE",
        "role": "旧交付件（部署链路与同图时延参照）",
        "nano": 0.6666, "micro": 0.8370, "f1": 0.887122, "nano4": 0.6033,
        "t1": 0.885997, "t2_fp": 10, "t3": 0.948330,
        "ms_delivery_board": 19.306, "ms_new_board": 23.528,
        "note": "不再作为 Beat-to-Pass 参照；仅保留部署链路/时延同图基准",
    }
    led["baseline_refs"]["0080"] = {
        "id": "trial_0080 (RC2)",
        "status": "RETIRED / OFFLINE HISTORICAL REFERENCE",
        "role": "离线精度天花板历史参照（离线语义，非板端）",
        "note": "已不参与 Beat-to-Pass；仅历史对照",
    }
    arms = led.setdefault("arms", {})
    st = arms.setdefault(str(ns.BASE_V2["trial"]), {"trial": ns.BASE_V2["trial"]})
    st["is_baseline_v2"] = True
    st["status"] = "ACTIVE / OFFICIAL BASELINE"
    st["baseline_name"] = ns.BASE_V2["name"]
    st["beat_to_pass_ref"] = "Baseline_v2.0(trial_9863)"


def main() -> None:
    led = ns.load_ledger()
    ensure_meta(led)
    sess = import_board_sessions(led)
    ns.save_ledger(led)

    # 回读校验
    led2 = ns.load_ledger()
    assert led2["baseline_v2"]["trial"] == 9863, "baseline_v2 未落盘"
    assert led2["baseline_registry"][0]["trial"] == 9863, "registry 顺序错"
    assert led2["beat_to_pass"]["nano_min"] == ns.BASE_V2["nano"], "beat_to_pass 未落盘"
    assert led2["arms"]["9863"]["is_baseline_v2"] is True, "9863 臂标记缺失"
    assert "0031" in led2["baseline_refs"] and "0080" in led2["baseline_refs"], "refs 缺失"
    print("[OK] 台账升轨完成：")
    print(f"  ledger      = {LEDGER}")
    print(f"  baseline_v2 = {led2['baseline_v2']['name']} (trial {led2['baseline_v2']['trial']}) "
          f"nano={led2['baseline_v2']['nano']} F1={led2['baseline_v2']['f1']}")
    for r in led2["baseline_registry"]:
        print(f"  - {r['id']:<20} trial={r['trial']:<6} {r['status']}")
    print(f"  beat_to_pass = {led2['beat_to_pass']['ref']} nano_min={led2['beat_to_pass']['nano_min']}")
    print(f"  arms n={len(led2['arms'])}  9863.is_baseline_v2={led2['arms']['9863']['is_baseline_v2']}")
    for key, s in (sess or {}).items():
        print(f"  board {key}: " + ", ".join(f"{k}={v}ms" for k, v in sorted(s["arms"].items())))
    print(f"  9863.board_ms = {led2['arms']['9863'].get('board_ms')}")


if __name__ == "__main__":
    main()
