#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把三路汤（SB3）臂的 `soup_family` 重打成**粗粒度家族标签**。

为什么必须做（2026-11-18 09:55 实测）：
  · 升轨的两道家族关（`promote_candidate.soup_family_stats` / `soup_family_six_ok`
    与 `auto_promote_watch` 的 n≥4）**都按台账里的 `soup_family` 字符串分组**；
  · 初版 sb3 标签把权重精确到 1e-4 ⇒ **每份配方自成一族** ⇒ 每族 n=1 ⇒ **永远升不了轨**；
  · 两路汤的老口径是 `bs<parent>:<权重档>`（权重取整 1%）⇒ 邻域臂天然同族。
⇒ 统一改成 `sb3:<成员集合>:bw<增益子权重档>`（与 `night_soup3.fam_tag` 完全一致）。
"""
from __future__ import annotations

import os
import sys

REPO = "/tmp/pycharm_project_10ae9e2e"
sys.path.insert(0, REPO)
os.chdir(REPO)
from manu.pipeline.npu import nano_search as NS  # noqa: E402


def main() -> None:
    led = NS.load_ledger()
    n = 0
    fams: dict[str, int] = {}
    for _k, st in (led.get("arms") or {}).items():
        if not st.get("synth"):
            continue
        mem = st.get("soup_members") or []
        w = st.get("soup_weights") or []
        if len(mem) < 2:
            continue
        is_sb3 = (str(st.get("channel") or "") == "SB3"
                  or str(st.get("desc") or "").startswith("[SB3]")
                  or str(st.get("soup_family") or "").startswith("sb3:"))
        if not is_sb3:
            continue
        parts = [f"{int(t)}@{int(e)}" for (t, e) in mem]
        bw = int(round(float(w[-1]) * 100)) if w else 0
        tag = "sb3:" + "+".join(parts) + f":bw{bw:02d}"
        if st.get("soup_family") != tag:
            st["soup_family"] = tag
            n += 1
        fams[tag] = fams.get(tag, 0) + 1
    NS.save_ledger(led)
    print(f"[retag] 重打 {n} 个 sb3 臂；家族分布：")
    for k, v in sorted(fams.items(), key=lambda x: -x[1]):
        print(f"    n={v:<3} {k}")


if __name__ == "__main__":
    main()
