#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**分辨率地板过滤器**（配方级）—— 2026-11-23 新增。

★ 为什么需要（实测依据）
────────────────────────
台账 13,765 臂实测：把臂展平成「基成员权重向量」后与基线比 **maxΔ**：
  · maxΔ < 0.012            ⇒ nano/nano4 读数与基线**逐位相同** ⇒ 纯烧 GPU
  · 0.012 ≤ maxΔ ≤ 0.030     ⇒ nano / T1 / T3 **最大增益 = 0**（单调损失区）
  · 0.03 ≤ maxΔ ≲ 0.16      ⇒ 「活带」（唯一可能动读数的区间）

现场（2026-11-23 09:50）：待打队列 64 臂里 **55 臂 maxΔ<0.012** ⇒ ≈8 小时门禁机时纯浪费。

⇒ 本脚本在**合成前**把死区配方直接丢掉（`weight_space.prune_below_floor`），
  **任何配方生成器都受保护**（不必手工调剂量下限）。

用法：`cat recipes.txt | python filter_live_recipes.py [floor]`
  · stdin  = `trial:epoch:w,trial:epoch:w;...`（`night_soup3.py --recipes` 格式）
  · stdout = 只含活带配方（同样格式，可直接喂 night_soup3）
  · stderr = 剔除统计（可审计）
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/tmp/pycharm_project_10ae9e2e/manu/pipeline/npu/scripts")
try:
    import weight_space as WS
except Exception as exc:  # noqa: BLE001
    sys.stderr.write("[filter_live] weight_space 不可用（%s）⇒ **放行全部**（保守）\n" % exc)
    WS = None

FLOOR = float(os.environ.get("GATE_MIN_MAXDELTA", "0.030"))


def main() -> None:
    text = sys.stdin.read()
    recs = [x.strip() for x in text.split(";") if x.strip()]
    if not recs:
        sys.exit(0)
    if WS is None:
        print(";".join(recs))
        return
    try:
        led = WS.load_ledger()
        fl = WS.Flattener(led.get("arms") or {})
        base = WS.baseline_flat(led, fl)
        kept, dropped = WS.prune_below_floor(recs, base, fl, floor=FLOOR)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write("[filter_live] 校验异常（%s）⇒ **放行全部**（保守）\n" % exc)
        print(";".join(recs))
        return
    # 🔴 `prune_below_floor` 返回 [(rec, maxΔ), ...] ⇒ 拆包取 rec（否则 join 报 tuple 错）
    kept = [r for r, _mx in kept]
    dropped = [(r, mx) for r, mx in dropped]
    if dropped:
        sys.stderr.write("[filter_live] 分辨率地板：剔除 %d/%d 条（maxΔ<%.3f ⇒ 读数必与基线同格）\n"
                         % (len(dropped), len(recs), FLOOR))
        for r, mx in dropped[:5]:
            sys.stderr.write("[filter_live]   DROP maxΔ=%s  %s\n" % ("?" if mx is None else ("%.4f" % mx), r))
    if kept:
        print(";".join(kept))


if __name__ == "__main__":
    main()
