#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""周一 09:00 交接用「**升轨命令生成器**」（🔴 **只读**，2026-11-20 新增）。

为什么必须有它（全部实测，不是推测）：

  ① 自动升轨链要求**三关同时**满足：同家族 n≥4 ＋ 板端 T4 ≤ 19.989 ms ＋ 种子复现
     （`promote_candidate.check_arm` 的口径）；
  ② 周末新轴产出的是**新成员 / 单臂**（§268 分辨率地板轴、§270「基线骨架 + T1 供体」
     微剂量轴、§270 PT1「T1 供体再抛光」轴）⇒ 同一配方作用在 ≥2 个锚点上的票数
     **天然不足** ⇒ `auto_promote_watch` **不会**把它们自动升为基线；
  ③ §270 §18 的结论：**「周一 09:00 前必须有人工兜底」**。
     而兜底的形式必须是一条**能直接复制粘贴执行**的命令 ——
     否则 09:00 接手的人还得在 24 MB 台账 + 几十个门禁日志里考古半小时。

⇒ 本脚本把「**哪个臂 / 六项读数 / 差哪一关 / 复制粘贴跑哪条命令**」
   直接印进周一晨报，让 09:00 接手的人 **30 秒**内完成升轨。

🔴 **只读铁律**：本脚本只做 `json.load`（台账）＋ `glob`（门禁报告）＋ 读文件，
   **不写台账、不写任何文件、不碰权重**。真升轨由
   `promote_candidate.py --trial N [--force] [--require-board]` 完成。
   ⇒ 本脚本可以安全地在**任何时刻**运行（包括流水线正在写台账时）。

用法（服务器）：
    cd /tmp/pycharm_project_10ae9e2e && export PYTHONPATH=.
    /home/huangzhe/anaconda3/envs/uav/bin/python manu/pipeline/npu/promote_ready_cmd.py
    /home/huangzhe/anaconda3/envs/uav/bin/python manu/pipeline/npu/promote_ready_cmd.py --top 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

import promote_candidate as PC  # noqa: E402
import six_criteria_audit as SCA  # noqa: E402

LEDGER = SCA.LEDGER
GATE_DIR = SCA.GATE_DIR
REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
PY_BIN = "/home/huangzhe/anaconda3/envs/uav/bin/python"
PROMOTE_REL = "manu/pipeline/npu/promote_candidate.py"
SIX_KEYS = ("nano", "nano4", "f1", "t1", "t2fp", "t3")


def next_baseline_name(led: dict) -> str:
    """建议的新基线名 = 当前基线名的次版本（`Baseline_v3.1` → `Baseline_v3.2`）。"""
    cur = ((led.get("baseline_v2") or {}).get("name") or "").strip()
    if not cur:
        return "Baseline_v3.2"
    head, _, tail = cur.rpartition(".")
    if head and tail.isdigit():
        return f"{head}.{int(tail) + 1}"
    return cur + "_next"


def cmd_line(trial: int, name: str, force: bool, require_board: bool, dry: bool) -> str:
    bits = [f"cd {REPO} && {PY_BIN} {PROMOTE_REL}", f"--trial {trial}", f"--name {name}"]
    if dry:
        bits.append("--dry-run")
    else:
        if force:
            bits.append("--force")
        if require_board:
            bits.append("--require-board")
    return " ".join(bits)


def main() -> None:
    ap = argparse.ArgumentParser(description="生成周一交接用的升轨命令（只读）")
    ap.add_argument("--ledger", default=LEDGER)
    ap.add_argument("--gate-dir", default=GATE_DIR)
    ap.add_argument("--top", type=int, default=3, help="最多列几个冠军臂")
    args = ap.parse_args()

    print("===== 🧭 周一升轨命令（复制粘贴即可 · 本脚本只读，不改任何状态）=====")
    try:
        led = json.load(open(args.ledger, encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️ 读台账失败：{exc}")
        return

    base = SCA.live_base(led)
    name = next_baseline_name(led)
    bt = (led.get("baseline_v2") or {}).get("trial")
    print(f"当前基线：{((led.get('baseline_v2') or {}).get('name') or '?')}（trial {bt}）")
    print(f"  判据口径：nano {base['nano']:.7f} / nano4 {base['nano4']:.7f} / F1 {base['f1']:.7f}"
          f" / T1 {base['t1']:.7f} / T2_fp ≤ {SCA.t2_cap(base)} / T3 {base['t3']:.7f}")
    print(f"建议新基线名：{name}")

    rows = SCA.collect(args.ledger, args.gate_dir)
    if not rows:
        print("\n（没有可配对的 (臂, epoch) 读数 ⇒ 本周末暂无门禁结果）")
        return
    for r in rows:
        r["npass"], r["miss"] = SCA.judge(r, base)

    # 🔴🔴🔴 2026-11-23 —— **「六项全过」（`>=` 口径）≠ 可升轨**。
    #   与基线**逐位打平**的克隆臂也算「六项全过」（现场 57 臂全部如此 ⇒ 零改进）
    #   ⇒ 候选榜只收**至少一轴严格优于基线**的臂，否则本脚本会把基线克隆臂
    #     当成「Baseline_v3.2 升轨命令」交给人粘贴执行（**基线名通胀**）。
    ties = [r for r in rows if r["npass"] == 6 and not SCA.strictly_better(r, base)]
    champs = [r for r in rows if r["npass"] == 6 and SCA.strictly_better(r, base)]
    champs.sort(key=lambda r: (-(r["nano"] or 0.0), -(r["f1"] or 0.0)))
    near = sorted((r for r in rows if r["npass"] == 5),
                  key=lambda r: (-(r["nano"] or 0.0), -(r["f1"] or 0.0)))

    print(f"\n六项全过臂：{len(champs) + len(ties)} 个"
          f"（其中与基线逐位打平、**不可升轨**的 {len(ties)} 个）"
          f" ｜ 5/6 近失臂：{len(near)} 个"
          f" ｜ 有读数的臂：{len(rows)} 个")

    if not champs:
        print("\n⚠️ **当前没有「六项全过 ∧ 至少一轴严格优于基线」的臂** ⇒ 升轨链此时无法手工兜底。")
        print("   最近的 5/6 臂（仅供观察，不是候选）：")
        for r in near[:5]:
            print(f"   - trial {r['trial']}@{r['ep']}：差 {r['miss']}"
                  f"（nano {r['nano']:.7f} / n4 {r['nano4']:.7f} / F1 {r['f1']:.7f}"
                  f" / T1 {r['t1']:.7f} / T2fp {r['t2_fp']} / T3 {r['t3']:.7f}）")
        print("\n⇒ 六项全过的臂一旦出现，本节会在同一份晨报里自动给出命令。")
        return

    print("\n----- 🏆 可升轨候选（六项全过 ∧ 至少一轴严格优于基线 · 按 nano 排序）-----")
    for i, r in enumerate(champs[: max(1, args.top)], 1):
        trial = int(r["trial"])
        try:
            chk = PC.check_arm(led, base, trial)
        except Exception as exc:  # noqa: BLE001
            print(f"{i}) trial {trial}@{r['ep']}：check_arm 异常 {exc}")
            continue
        seed_ok = bool(chk.get("seed_replicated"))
        board_ms = chk.get("board_ms")
        stale = bool(chk.get("board_stale"))
        checks = chk.get("checks") or {}
        six_local = all(bool(checks.get(k)) for k in SIX_KEYS if k in checks)
        print(f"{i}) trial {trial} @ epoch {r['ep']}　desc: {(led.get('arms') or {}).get(str(trial), {}).get('desc', '')}")
        print(f"   六项：nano {r['nano']:.7f} / nano4 {r['nano4']:.7f} / F1 {r['f1']:.7f}"
              f" / T1 {r['t1']:.7f} / T2_fp {r['t2_fp']} / T3 {r['t3']:.7f}　⇒ 6/6 ✅")
        if board_ms is None:
            print(f"   板端 T4：**未测** ⇒ 升轨前必须先让 `board_t4_watch` 补测（无读数 ⇒ 不加 --require-board）")
        elif stale:
            print(f"   板端 T4：{board_ms} ms，但打的是**旧基线**参照 ⇒ 判定过期，需重测")
        else:
            print(f"   板端 T4：{board_ms} ms　通过={chk.get('board_pass_t4')}"
                  f"（红线 {getattr(PC.T4, 'LIMIT_MS', 19.989)} ms）")
        print(f"   种子复现：{'✅ 已复现' if seed_ok else '⚠️ 未复现（需 --force 强推）'}"
              f"　{chk.get('seed_note', '')}")
        if not six_local:
            print(f"   ⚠️ 注意：check_arm 的六项判定未全绿（checks={checks}）⇒ 建议先跑下面的预检")
        force = not seed_ok
        need_board = board_ms is not None and not stale
        print(f"   ① 预检（不改台账）：")
        print(f"      {cmd_line(trial, name, force, need_board, dry=True)}")
        print(f"   ② 真升轨：")
        print(f"      {cmd_line(trial, name, force, need_board, dry=False)}")
        if force:
            print(f"      （带 --force：{'该臂未过「同配方 ≥2 锚点」种子复现关，' if not seed_ok else ''}"
                  f"人工确认后强推）")
        print()

    print("ℹ️ 升轨后请回填基线常量并核对：新基线应同时出现在台账 `baseline_v2` 与")
    print("   `manu/pipeline/npu/nano_search.py` 的 BASE 常量里（脚本会打印待回填值）。")
    print("ℹ️ 若上表为空但确实有更好的臂 ⇒ 多半是「六项判据」口径问题，"
          "先看本节上方的判据口径行（t2_cap = max(基线 t2_fp, 5)）。")


if __name__ == "__main__":
    main()
