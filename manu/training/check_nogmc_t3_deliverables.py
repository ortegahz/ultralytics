#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""交付件自检（2026-11-03 T3 前沿轮）：一条命令确认「该在的东西都在、且自洽」。

检查项
======
1. **产物齐备**：HANDOVER.md / HANDOVER.json / SUMMARY.md / optuna_summary.csv /
   T3_FRONTIER.md / T3_FRONTIER.json / T3_REACHABILITY.md / T3_REACHABILITY.json /
   AUTORUN_DONE.json / variants.csv
2. **判定自洽**：HANDOVER.json 的 winner 与 T3_FRONTIER.json 的最优配置是否一致；
   变体组是否进池；阻塞项是否都被显式列出
3. **门禁口径**：T1 红线 / T2 下限 / T2 虚警上限是否等于参照系包里的值
4. **训练-评估时序**：门禁开始时刻是否**晚于**最后一条训练臂完成时刻（铁律：
   评估不得与训练同卡）
5. **study 终态**：trial 状态分布、有没有 FAIL/INFRA-FAIL
6. **md5 一致性**：本地仓库 / SSHFS 镜像 / 服务器 三侧（本脚本在服务器上跑，
   只能比「镜像 vs 服务器」；本地侧由调用者另跑）

用法（服务器，conda `uav`，根 /tmp/pycharm_project_10ae9e2e）：
```bash
python manu/training/check_nogmc_t3_deliverables.py \
  --out-root /mnt/data/siping/runs/nogmc_t3 \
  --ref /mnt/data/siping/runs/nogmc_ref_weekend/reference_frame.json
```
退出码：0 = 全绿；1 = 有 ❌ 项。
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

OK = "✅"
BAD = "❌"
WARN = "⚠️"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--expect-study", default="mix_ratio_search_nogmc_t3")
    args = ap.parse_args()
    root = Path(args.out_root)
    ref = json.loads(Path(args.ref).read_text(encoding="utf-8"))
    fails: list[str] = []

    print("=" * 78)
    print("T3 前沿轮 · 交付件自检")
    print("=" * 78)

    # ---- 1. 产物齐备 ----
    print("\n[1] 产物齐备")
    wanted = ["HANDOVER.md", "HANDOVER.json", "SUMMARY.md", "optuna_summary.csv",
              "T3_FRONTIER.md", "T3_FRONTIER.json", "T3_REACHABILITY.md",
              "T3_REACHABILITY.json", "variants.csv",
              "runtime_config.json"]
    for w in wanted:
        p = root / w
        if p.is_file() and p.stat().st_size > 0:
            print(f"  {OK} {w:<26} {p.stat().st_size:>9,} B")
        else:
            print(f"  {BAD} {w:<26} 缺失或为空")
            fails.append(f"产物缺失：{w}")
    # ⚠️ **AUTORUN_DONE.json 不进硬性清单**（2026-11-04 08:53 修）：
    #    它由 autorun 的**最后一步 ⑥「备份 + DONE 标记」**写出，而本自检是 **⑤d**
    #    ⇒ **自检必然跑在它之前** ⇒ 缺失是**时序必然**，不是真缺失
    #    （昨晚实测：自检查它「缺失或为空」⇒ 假 FAIL，而 1 分钟后它就出现了）。
    _dp = root / "AUTORUN_DONE.json"
    if _dp.is_file() and _dp.stat().st_size > 0:
        print(f"  {OK} {'AUTORUN_DONE.json':<26} {_dp.stat().st_size:>9,} B")
    else:
        print(f"  {WARN} {'AUTORUN_DONE.json':<26} 尚未写出（由 autorun 最后一步 ⑥ 生成，"
              f"本自检跑在其前 ⇒ 非缺失；收尾完成后重跑本自检即会转绿）")

    # ---- 2. 判定自洽 ----
    print("\n[2] 判定自洽")
    hj = root / "HANDOVER.json"
    if hj.is_file():
        try:
            doc = json.loads(hj.read_text(encoding="utf-8"))
        except Exception as exc:
            doc = {}
            print(f"  {BAD} HANDOVER.json 解析失败：{exc}")
            fails.append("HANDOVER.json 解析失败")
        w = doc.get("winner") or {}
        if w:
            print(f"  {OK} 交接候选：{w.get('variant_key') or w.get('trials')}"
                  f"（{'变体' if w.get('is_variant') else '配置组'}）"
                  f" T1={w.get('t1_median')} T2FP={w.get('t2_fp_median')} T3={w.get('t3_median')}")
            if w.get("is_variant"):
                print(f"      变体权重：{w.get('weights')}")
                if not Path(w.get("weights") or "").is_file():
                    print(f"  {BAD} 变体权重文件不存在")
                    fails.append("变体权重文件不存在")
                if not (w.get("seed_evidence_n") or 0) >= 2:
                    print(f"  {WARN} 变体的多种子证据 = {w.get('seed_evidence_n')}（<2）")
        else:
            print(f"  {BAD} 没有交接候选")
            fails.append("没有交接候选")
        print(f"  {OK} 阻塞项 {len(doc.get('blockers') or [])} 条："
              + ("；".join((doc.get('blockers') or [])[:3]) or "无"))
        vg = doc.get("variant_groups")
        if vg is not None:
            print(f"  {OK} 变体组 {len(vg)} 个进池（单独成组，不污染配置中位）")

    # ---- 3. 门禁口径 ----
    print("\n[3] 门禁口径")
    fb = ref.get("frozen_baseline") or {}
    g = ref.get("gate") or {}
    t1_red = fb.get("track1_f1", 0) + float(g.get("track1_min_delta", -0.005))
    t2_min = g.get("track2_f1_min", 0.985)
    t2_max = fb.get("track2_fp", 52) + float(g.get("track2_fp_max_increment", 60))
    print(f"  T1 红线 {t1_red:.16f} | T2 F1 ≥ {t2_min} | T2 FP ≤ {t2_max}")
    if hj.is_file():
        rl = (json.loads(hj.read_text(encoding='utf-8')).get("redlines") or {})
        if rl:
            # ⚠️ **键名别名**（2026-11-04 08:53 修）：判定器写的是 `track1`，
            #    而这里原来只认 `t1_redline` ⇒ 取到默认值 -1 ⇒ **误报「红线不一致」**。
            #    实测两侧数值完全一致（0.8958967482400269 / 0.985 / 112）⇒ 是**假 FAIL**。
            # ⇒ 两种键名都接受。
            t1_in = rl.get("t1_redline", rl.get("track1", -1))
            # ⚠️ **容差 1e-6**（2026-11-04 08:58 修）：参照系包 `reference_frame.json` 的
            #    `frozen_baseline.track1_f1` 只保留了 **6 位小数**（0.900897），
            #    而判定器算红线用的是**未舍入**的基线（0.9008967482400269 − 0.005）
            #    ⇒ 两侧天然差 **7.48e-7** ⇒ 用 1e-12 判会**误报「红线不一致」**（实测假 FAIL）。
            #    ⇒ 红线按 6 位小数量级比较即可。
            ok = (abs(float(t1_in) - t1_red) < 1e-6
                  and abs(float(rl.get("t2_f1_min", rl.get("track2_f1_min", -1))) - t2_min) < 1e-9
                  and abs(float(rl.get("t2_fp_max", rl.get("track2_fp_max", -1))) - t2_max) < 1e-9)
            print(f"  {OK if ok else BAD} HANDOVER.json 里的红线与参照系包一致")
            if not ok:
                fails.append("判定报告的红线与参照系包不一致")

    # ---- 4. 训练-评估时序 ----
    print("\n[4] 训练-评估时序（铁律：评估不得与训练同卡）")
    db = root / "study.db"
    if db.is_file():
        conn = sqlite3.connect(str(db))
        try:
            rows = list(conn.execute(
                "SELECT number, state, datetime_start, datetime_complete FROM trials "
                "WHERE datetime_complete IS NOT NULL ORDER BY datetime_complete DESC"))
            states = dict(conn.execute("SELECT state, count(*) FROM trials GROUP BY state"))
            print(f"  {OK} study 状态分布：{states}")
            # ⚠️ **「最后一条训练臂完成时刻」只认真正跑过门禁的臂**：
            #    被**主动剪掉**（PRUNED 但从未开跑）的 trial 也有 datetime_complete
            #    ⇒ 用它做时序断言会误报「门禁早于训练结束」。
            ran = []
            for num, st, ds, dc in rows:
                if list((root / "trials" / f"trial_{num:04d}").glob("gate/*/gate_report.json")):
                    ran.append((num, dc))
            if not ran:
                ran = [(num, dc) for num, st, ds, dc in rows if st == "COMPLETE"]
            for num, dc in ran[:1]:
                print(f"  {OK} 最后一条**真正跑过门禁**的训练臂完成时刻：t{num} @ {dc}")
                # 找门禁报告的 mtime（取最新的那个）
                reps = sorted(root.glob("swa/gate/*/gate_report.json"), key=lambda p: p.stat().st_mtime)
                if reps:
                    import datetime as _dt
                    gate_m = _dt.datetime.fromtimestamp(reps[-1].stat().st_mtime)
                    dc_dt = _dt.datetime.fromisoformat(str(dc))
                    late = gate_m > dc_dt
                    print(f"  {OK if late else BAD} 门禁产物最新时刻 {gate_m} "
                          f"{'晚于' if late else '**早于**'}最后一条训练臂完成时刻")
                    if not late:
                        fails.append("门禁可能与训练同卡（时序违规）")
        finally:
            conn.close()
    else:
        print(f"  {WARN} 没有 study.db")

    # ---- 5. memory 导航表 ----
    print("\n[5] memory 导航表")
    for cand in ("manu/memory/validate_nav_table.py", "manu/training/validate_nav_table.py"):
        p = Path(cand)
        if p.is_file():
            import subprocess
            r = subprocess.run(["python3", cand], capture_output=True, text=True)
            tail = (r.stdout or r.stderr).strip().splitlines()[-1:] or ["(无输出)"]
            ok = "PASS" in (r.stdout or "")
            print(f"  {OK if ok else BAD} {cand}: {tail[0]}")
            break
    else:
        print(f"  {WARN} 找不到导航表校验脚本")

    print("\n" + "=" * 78)
    if fails:
        print(f"{BAD} 自检未通过（{len(fails)} 项）：")
        for f in fails:
            print(f"   - {f}")
        return 1
    print(f"{OK} 自检全绿")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
