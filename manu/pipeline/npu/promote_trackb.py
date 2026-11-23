#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Track-B 升轨工具（2026-11-18 16:40 新增）

**为什么需要它**：严格轨（六项全过）已触顶 ——
  Baseline_v2.7 = trial_1540030（nano .7218809 / nano4 .6568998 / F1 .8905687 / T1 .8906883 /
  T2_fp=1 / T3 .9487751 / 板卡 19.668 ms）是 **3 成员 sb3 家族里「T2_fp=1 区间」的 nano 上限**
  ⇒ 继续搜只能靠**放宽 T2 虚警**换召回。
  而**用户口径红线是 T2_fp ≤ 5**（trial_9863 就是以 T2_fp=5 成为 Baseline_v2.0 的，用户当时明确裁决认可），
  任务书交付红线 T2_fp ≤ 20（官方门禁 ≤ 36）。

**Track-B 升轨判据**（全部用台账精确读数）：
  ① nano / nano4 / f1 / t1 / t3 **五项全部 ≥ 当前基线**，且 **nano 严格更优**；
  ② **T2_fp ≤ 5**（可 --t2-max 调整，但默认锁死用户红线）；
  ③ **板卡 T4 ≤ 19.989 ms**（硬红线，必测）；
  ④ **同家族复现票 ≥ 2**（成员集 + 权重桶相同 ⇒ 不是单臂彩票）。

**安全设计**：
  * 升轨时把**严格轨旧基线**完整存进台账 `baseline_v2_strict_backup` ⇒ 一键可回退；
  * registry 注明「TRACK-B：T2_fp=N（严格轨前基线为 1）」⇒ 晨报/memory 同步提示，用户可一票否决；
  * `beat_to_pass` 随新基线更新（含 t2_fp_max = 新基线的 t2_fp）。

用法：
  $PY manu/pipeline/npu/promote_trackb.py --list
  $PY manu/pipeline/npu/promote_trackb.py --trial 1544008 --dry-run
  $PY manu/pipeline/npu/promote_trackb.py --trial 1544008
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

import nano_search as ns  # noqa: E402
from trackb_core import (AXES, LIMIT_MS, T2_DELIVER, T2_USER, base_of, collect_rows,  # noqa: E402
                         family_votes, gaps, load_ledger, qualifies)

REPO = "/tmp/pycharm_project_10ae9e2e"
RUNS = "/mnt/data/siping/runs/npu20ms/nano_search"
MEM = os.path.join(REPO, "manu", "memory", "memory.md")


def next_name(cur: str) -> str:
    m = re.match(r"Baseline_v2\.(\d+)", str(cur or ""))
    return "Baseline_v2.%d" % (int(m.group(1)) + 1 if m else 1)


def log(msg: str) -> None:
    print("[%s] %s" % (time.strftime("%F %T"), msg), flush=True)


def cmd_list(led: dict, base: dict, t2_max: int) -> None:
    rows = collect_rows(led)
    base_t = base_of(led)
    hits = []
    for r in rows:
        ok, miss, win = qualifies(r, base_t, t2_max, require_board=False)
        if ok:
            hits.append((r, win))
    hits.sort(key=lambda x: -(x[0].get("nano") or 0))
    print("基线 %s / trial_%s：nano=%.7f nano4=%.7f f1=%.7f t1=%.7f t3=%.7f t2_fp=%s"
          % (base_t["name"], base_t["trial"], base_t["nano"], base_t["nano4"], base_t["f1"],
             base_t["t1"], base_t["t3"], base_t["t2_fp"]))
    print("Track-B 合格（五项全 ≥ 且至少一项严格优，T2_fp≤%d）：%d 个" % (t2_max, len(hits)))
    for r, win in hits[:25]:
        print("  trial_%-8d nano=%.7f nano4=%.7f f1=%.7f t1=%.7f t2fp=%-2s t3=%.7f 板卡=%s 票=%d  %s"
              % (r["trial"], r["nano"], r["nano4"] or 0, r["f1"] or 0, r["t1"] or 0,
                 r["t2_fp"], r["t3"] or 0, r.get("board_ms"),
                 family_votes(rows, r.get("family"), base_t, t2_max), r.get("family") or ""))


def do_promote(led: dict, base_t: dict, rows: list, trial: int, name: str,
               t2_max: int, min_family: int, dry: bool) -> int:
    st = (led.get("arms") or {}).get(str(trial)) or {}
    if not st:
        log("trial_%d 不在台账里 ⇒ 拒绝" % trial)
        return 2
    row = next((r for r in rows if r["trial"] == trial), None)
    if row is None:
        log("trial_%d 读数不全（需 nano/nano4/f1 + T1/T2fp/T3）⇒ 拒绝" % trial)
        return 2
    ok, miss, win = qualifies(row, base_t, t2_max, require_board=True)
    log("── Track-B 预检 ───────────────────────────")
    log("  trial %d  %s" % (trial, (row.get("desc") or "")[:70]))
    log("  nano  %.7f (基线 %.7f)   nano4 %.7f (基线 %.7f)   F1 %.7f (基线 %.7f)"
        % (row["nano"], base_t["nano"], row["nano4"] or 0, base_t["nano4"],
           row["f1"] or 0, base_t["f1"]))
    log("  T1    %s (基线 %s)   T2FP %s (红线 %d)   T3 %s (基线 %s)"
        % (row.get("t1"), base_t["t1"], row.get("t2_fp"), t2_max, row.get("t3"), base_t["t3"]))
    log("  板卡  %s ms（红线 %.3f）  家族 %s" % (row.get("board_ms"), LIMIT_MS, row.get("family")))
    votes = family_votes(rows, row.get("family"), base_t, t2_max)
    log("  同家族复现票 %d（要求 ≥%d）  严格胜出：%s" % (votes, min_family, win))
    if not ok:
        log("  ⇒ Track-B 判据未过（缺：%s）⇒ 拒绝升轨" % (miss or "无"))
        return 1
    if votes < min_family:
        log("  ⇒ 同家族复现票不足（%d < %d）⇒ 拒绝升轨" % (votes, min_family))
        return 1
    if not (row["nano"] > base_t["nano"]):
        log("  ⇒ nano 未严格优于基线 ⇒ 拒绝升轨（防基线名通胀）")
        return 1
    if dry:
        log("  （--dry-run：不写台账）")
        return 0
    # ── 执行升轨 ──
    old = dict(led.get("baseline_v2") or {})
    epoch = int(row.get("epoch") or 5)
    ck = "%s/trials/trial_%d/weights/epoch%d.pt" % (ns.ROOT, trial, epoch)
    new = dict(old)
    new.update({
        "name": name, "trial": trial, "epoch": epoch, "ckpt": ck,
        "nano": row["nano"], "nano4": row.get("nano4"), "f1": row.get("f1"),
        "micro": row.get("micro"), "small": row.get("small"), "regular": row.get("regular"),
        "t1": row.get("t1"), "t1_fp": row.get("t1_fp"),
        "t2": None, "t2_fp": row.get("t2_fp"),
        "t3": row.get("t3"), "t3_fp": row.get("t3_fp"),
        "ms_new_board": row.get("board_ms"), "ms_delivery_board": old.get("ms_delivery_board"),
        "recipe": (row.get("desc") or "")[:200],
        "promoted_at": time.strftime("%F %T"),
        "track": "TRACK-B",
        "previous_baseline": {"name": old.get("name"), "trial": old.get("trial"),
                              "nano": old.get("nano"), "t2_fp": old.get("t2_fp")},
    })
    led["baseline_v2"] = new
    # 严格轨旧基线留档（一键可回退）
    led["baseline_v2_strict_backup"] = {
        "saved_at": time.strftime("%F %T"), "baseline_v2": old,
        "note": "Track-B 升轨前的**严格轨**基线；回退命令见 trackb_core 回退脚本",
    }
    reg = [x for x in (led.get("baseline_registry") or []) if x.get("trial") != trial]
    reg.insert(0, {
        "id": name, "trial": trial, "status": "ACTIVE / OFFICIAL BASELINE (TRACK-B)",
        "note": ("Track-B 轨升轨：五项精度指标（nano/nano4/F1/T1/T3）全部 ≥ 前基线且 nano 严格更优；"
                 "**T2_fp=%s（严格轨前基线为 %s）** ⇒ 仍在用户口径红线 ≤%d / 任务书交付红线 ≤%d 内；"
                 "严格轨前基线 %s 已留档于 baseline_v2_strict_backup，可一键回退。"
                 % (row.get("t2_fp"), old.get("t2_fp"), T2_USER, T2_DELIVER, old.get("name"))),
    })
    for x in reg:
        if str(x.get("status", "")).startswith("ACTIVE") and x.get("trial") != trial:
            x["status"] = "SUPERSEDED / DEPRECATED BASELINE"
    led["baseline_registry"] = reg
    led["beat_to_pass"] = {
        "ref": "%s(trial_%d)" % (name, trial), "nano_min": new["nano"],
        "f1_min": new["f1"], "t1_min": new["t1"],
        "t2_fp_max": new["t2_fp"], "t3_min": new["t3"],
        "legacy_ref": "trial_0031", "legacy_nano_min": 0.6666,
        "note": "后续所有探索统一以新基线的指标为 Beat-to-Pass 下限（Track-B 轨，T2_fp 上限=%s）" % new["t2_fp"],
    }
    st["is_baseline_v2"] = True
    st["status"] = "ACTIVE / OFFICIAL BASELINE (TRACK-B)"
    st["baseline_name"] = name
    st["beat_to_pass_ref"] = "%s(trial_%d)" % (name, trial)
    for k, ost in (led.get("arms") or {}).items():
        if k != str(trial) and ost.get("is_baseline_v2"):
            ost["is_baseline_v2"] = False
            ost["status"] = "SUPERSEDED / DEPRECATED BASELINE"
    ns.save_ledger(led)
    rec = {"promoted_at": time.strftime("%F %T"), "track": "TRACK-B", "from": old.get("name"),
           "to": name, "trial": trial, "epoch": epoch, "family": row.get("family"),
           "family_votes": votes, "t2_max": t2_max,
           "nano": row["nano"], "nano4": row.get("nano4"), "f1": row.get("f1"),
           "t1": row.get("t1"), "t2_fp": row.get("t2_fp"), "t3": row.get("t3"),
           "board_ms": row.get("board_ms"),
           "delta_vs_prev": {k: (float(row[k]) - float(old.get(k) or 0)) for k in AXES}}
    out = os.path.join(RUNS, "promotion_ready_trackb.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    # 常量回填（six_criteria_audit.BASE + nano_search.BASE_V2）
    try:
        from auto_promote_watch import backfill_constants
        backfill_constants(trial, epoch)
    except Exception as exc:  # noqa: BLE001
        log("常量回填失败（不影响台账）：%s" % exc)
    try:
        with open(MEM, "a", encoding="utf-8") as f:
            f.write(
                "\n### 🟢🟢 **Track-B 自动升轨记录**（%s）\n"
                "* **%s = trial_%d@%d**（家族 %s，同族复现票 %d）\n"
                "* 指标：nano %.6f / nano4 %.6f / F1 %.6f / T1 %.6f / **T2_fp %s** / T3 %.6f / 板卡 %s ms\n"
                "* 判据：五项精度指标全 ≥ 前基线（%s）且 nano 严格更优；T2_fp ≤ %d（用户口径红线）\n"
                "* ⚠ 严格轨前基线 %s 已留档 `baseline_v2_strict_backup` ⇒ 可一键回退\n"
                % (rec["promoted_at"], name, trial, epoch, row.get("family"), votes,
                   row["nano"], row.get("nano4") or 0, row.get("f1") or 0, row.get("t1") or 0,
                   row.get("t2_fp"), row.get("t3") or 0, row.get("board_ms"),
                   old.get("name"), T2_USER, old.get("name")))
    except Exception as exc:  # noqa: BLE001
        log("memory 写入失败：%s" % exc)
    log("  ✔ 台账已升轨 → %s" % ns.LEDGER)
    log("  ✔ 升轨记录 → %s" % out)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--trial", type=int)
    ap.add_argument("--name", default="")
    ap.add_argument("--t2-max", type=int, default=T2_USER)
    ap.add_argument("--min-family", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    led = load_ledger()
    base_t = base_of(led)
    rows = collect_rows(led)
    if args.list or args.trial is None:
        cmd_list(led, base_t, args.t2_max)
        return
    name = args.name or next_name(base_t.get("name"))
    rc = do_promote(led, base_t, rows, args.trial, name, args.t2_max, args.min_family, args.dry_run)
    sys.exit(rc)


if __name__ == "__main__":
    main()
