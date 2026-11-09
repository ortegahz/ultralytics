#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按成员自己的门禁分数，从家族里挑「最好的 k 个」重写家族定义（2026-11-06）。

为什么需要
----------
F_MAIN 家族计划攒到 **n≈40** 个成员（`p3/hc64/relu/late`，13.131 ms）⇒
`build_struct_soups.py` 会做**均匀平均**。均匀平均对种子噪声最稳，但：
* 若个别成员明显差（种子运气差）���均匀平均会把它带进来；
* 若大 soup 的 ΔT1 差一点点没过 −0.5 pp 门槛，**换更少的、更好的成员**常常能救回来。

⇒ 本脚本做一件事：读每个成员自己的门禁报告（与验收表**同一个**口径），
按指定指标排序，只保留 top-k，写出一份新的家族定义 JSON ⇒ 交给
`build_struct_soups.py` 重跑即可得到另一个 soup（结构自描述不变 ⇒ 速度列不变）。

用法（服务器，conda uav）：
    # ① 看每个家族成员的分数（不写文件）
    python manu/training/pick_soup_members.py \
        --families-json /mnt/data/siping/runs/npu20ms/struct_families.json \
        --root /mnt/data/siping/runs/npu20ms --metric t1_f1 --top-k 8 --dry-run

    # ② 写出只含 top-k 的新家族定义（原子写，先备份）
    python manu/training/pick_soup_members.py \
        --families-json /mnt/data/siping/runs/npu20ms/struct_families.json \
        --root /mnt/data/siping/runs/npu20ms --metric t1_f1 --top-k 8 \
        --out /mnt/data/siping/runs/npu20ms/struct_families_top8.json

    # ③ 用它重建 soup（结构自描述逐字段校验 ⇒ 结构不一致会直接报错）
    python manu/training/build_struct_soups.py \
        --out-root /mnt/data/siping/runs/npu20ms --study-name mix_ratio_search_npu20ms_v2 \
        --families-json /mnt/data/siping/runs/npu20ms/struct_families_top8.json \
        --out-dir /mnt/data/siping/runs/npu20ms/soups_top8 \
        --speed-table manu/pipeline/npu/struct_speed_table.json

⚠️ **口径提示**：`--metric` 取的是**门禁报告里那个轨的 F1**（与验收表同一份数字），
不是 baseline 对比后的 Δ ⇒ 想按"相对 cs_t0043 的退化"排序时，
`t1_f1` 本身已经是相对同一基线算出来的 F1，**排序等价于按 ΔT1 排**。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "manu" / "training"))

try:
    from build_struct_soups import resolve_member  # noqa: E402
    from npu20ms_acceptance import read_gate_report  # noqa: E402
except Exception as _exc:  # pragma: no cover
    raise SystemExit(f"[FATAL] 导入助手失败：{_exc}") from _exc


def _f1_of(gate_report: dict, metric: str) -> float | None:
    """从门禁报告里取指定指标（t1_f1 / t3_f1 / t2_f1），口径与验收表一致。"""
    if not gate_report:
        return None
    v = gate_report.get(metric)
    if isinstance(v, (int, float)):
        return float(v)
    # 兜底：从 tracks 里取 finetune_peak.f1
    track = {"t1_f1": "track1_uav", "t3_f1": "track3_blind", "t2_f1": "track2_golden"}.get(metric)
    if track:
        node = (gate_report.get("tracks") or {}).get(track) or {}
        fp = node.get("finetune_peak") or {}
        if isinstance(fp.get("f1"), (int, float)):
            return float(fp["f1"])
        if isinstance(node.get("f1"), (int, float)):
            return float(node["f1"])
    g = (gate_report.get("gates") or {}).get(
        {"t1_f1": "track1_uav", "t3_f1": "track3_blind", "t2_f1": "track2_golden"}.get(metric, "")
    ) or {}
    if isinstance(g.get("value"), (int, float)):
        return float(g["value"])
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families-json", required=True)
    ap.add_argument("--root", required=True, help="study 产物根（trials/ 在它下面）")
    ap.add_argument("--study-name", default="", help="（仅用于日志；成员解析靠 role→trial 映射）")
    ap.add_argument("--metric", default="t1_f1", choices=("t1_f1", "t2_f1", "t3_f1"))
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--require-pass", action="store_true",
                    help="★只保留**门禁没有 veto 失败**的成员（压 T2 假阳的关键）；"
                         "若通过者不足 2 个 ⇒ 自动回落到全部成员并打印告警")
    ap.add_argument("--out", default="", help="写出的新家族定义（留空 = 只打印）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = Path(args.families_json)
    doc = json.loads(src.read_text(encoding="utf-8"))
    fams = doc.get("families") or {}
    if not fams:
        print(f"[FATAL] {src} 里没有 families")
        return 2

    class _A:
        """``resolve_member`` 需要的最小 args 影子（它只读这三个属性）。"""
        epoch = 0
        out_root = args.root
        study_name = args.study_name or "mix_ratio_search_npu20ms_v2"
    a = _A()

    out_fams: dict[str, dict] = {}
    for fam, fs in fams.items():
        members = fs.get("members") or []
        scored: list[tuple[float, dict, dict]] = []
        for m in members:
            w, info = resolve_member(m, a, int(fs.get("epoch") or 0))
            if w is None:
                print(f"  · {fam}: 成员 {info.get('role')} ⇒ {info.get('skip')}")
                continue
            gr_path = info.get("gate_report")
            gr = {}
            tdir = info.get("trial_dir")
            if tdir:
                try:
                    # ★ 传 **trial 目录**（read_gate_report 内部自己 glob "gate*/epoch*/gate_report.json"）
                    gr = read_gate_report(Path(tdir))
                except Exception:
                    gr = {}
            val = _f1_of(gr, args.metric)
            # ★ 判定「门禁是否通过」必须用 **verdict**（read_gate_report 归一化后**没有**
            #   veto_failures 字段 ⇒ 用它判断会静默失效，把 FAIL 臂当成通过者）。
            #   口径与验收表一致：verdict == "PASS" 且没有 infra_failed。
            ok = (str(gr.get("verdict") or "").upper() == "PASS") and not gr.get("infra_failed")
            info = dict(info)
            info["_gate_ok"] = ok
            info["_gate_verdict"] = gr.get("verdict")
            scored.append((val if val is not None else -1.0, m, info))
            print(f"  · {fam}: {info.get('role')} ⇒ {args.metric}={val} ({Path(w).name})"
                  f"{'' if ok else '  [门禁未通过 ⇒ 排除]'}")

        if args.require_pass:
            passing = [t for t in scored if t[2].get("_gate_ok", True)]
            if len(passing) >= 2:
                print(f"[PICK] {fam}: 门禁通过者 {len(passing)}/{len(scored)} ⇒ 只在通过者里选")
                scored = passing
            else:
                print(f"[WARN] {fam}: 门禁通过者只有 {len(passing)} 个（<2）⇒ 回落到全部成员排序")

        scored.sort(key=lambda t: -t[0])
        keep = scored[: max(2, args.top_k)]
        if len(keep) < 2:
            print(f"  · {fam}: 可用成员只有 {len(keep)} 个 ⇒ 不改")
            out_fams[fam] = fs
            continue
        kept_roles = [k[1].get("role") for k in keep]
        print(f"[PICK] {fam}: {len(scored)} 个成员 ⇒ 保留 top-{len(keep)}：{kept_roles}")
        new_fs = dict(fs)
        new_fs["members"] = [k[1] for k in keep]
        new_fs["picked_by"] = {
            "metric": args.metric, "top_k": args.top_k,
            "scores": [{"role": k[1].get("role"), args.metric: k[0]} for k in keep],
            "source_families": str(src),
        }
        out_fams[fam] = new_fs

    if args.out and not args.dry_run:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        if outp.is_file():
            shutil.copy2(outp, outp.with_suffix(outp.suffix + ".bak"))
        doc2 = dict(doc)
        doc2["families"] = out_fams
        doc2["picked_members_by"] = {"metric": args.metric, "top_k": args.top_k,
                                     "source": str(src), "ts": int(__import__("time").time())}
        outp.write_text(json.dumps(doc2, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] 新家族定义 -> {outp}（原文件已备份）")
    else:
        print("[DRY-RUN] 未写文件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
