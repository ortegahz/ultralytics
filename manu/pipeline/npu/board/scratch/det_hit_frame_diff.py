#!/usr/bin/env python3
"""det_hit_frame_diff.py -- 命中帧对比：两条后处理路径的逐帧检测差异。

WHAT THIS TOOL IS FOR
---------------------
后处理稀疏化（dense 全图 8 连通域 -> 稀疏路径 / GPU 路径）之后，
「精度有没有变」不能靠肉眼看��频，要靠**逐帧逐目标**的对照。
本工具做的是**命中帧对比**：

    命中帧 = 基准路径和候选路径**都产出了检出**的帧。
    在这些帧上，逐个目标对齐后比较坐标、峰值、框。

为什么只看命中帧：这是「精度有没有漂」的**必要且敏感**的观测口径。
命中帧上的残差如果已经超过 GT 框的容差，那非命中帧（一边有、一边没有）
只会让结果更糟，不可能更好。

口径说明（与 PC 评分器一致，见 manu/evaluation/eval_multiscale_cache_th.py）：

* 每个检测按 ``peak`` 降序排序，``--top-k`` 个参与逐个配对（默认 1，
  因为门禁口径 dedup_radius=768 实际等于每帧只留 top-1）。
* ``inbox`` 判定 = 预测**中心点**是否落在 GT 框内：
  ``abs(px-gx) <= gw/2 and abs(py-gy) <= gh/2``。
  预测自己的 w/h **不参与**判定（这一点在报告里必须写清楚）。
* GT 标签是归一化格式 ``class cx cy w h``（0..1），native 画布 640x512。

输入是两个 e2e/demo 同格式的 TSV：

    <stem>\\t<count>\\t<tag> <cx> <cy> <w> <h> <area> <peak> <sum>\\t...
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

CANVAS_W = 640
CANVAS_H = 512


def parse_mosaic_tsv(path: Path) -> dict[str, list[tuple]]:
    """-> {stem: [(tag, cx, cy, w, h, area, peak, sum), ...]}"""
    out: dict[str, list[tuple]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        stem = parts[0]
        n = int(parts[1])
        dets = []
        for field in parts[2:]:
            v = field.split()
            if len(v) != 8:
                raise ValueError(f"{path}: bad detection field {field!r}")
            dets.append(
                (
                    int(v[0]),
                    float(v[1]),
                    float(v[2]),
                    int(v[3]),
                    int(v[4]),
                    int(v[5]),
                    float(v[6]),
                    float(v[7]),
                )
            )
        if len(dets) != n:
            raise ValueError(f"{path}: {stem} declares {n} detections but carries {len(dets)}")
        out[stem] = dets
    return out


def frame_index(stem: str) -> int:
    return int(stem.rsplit("__", 1)[1])


def load_gt(labels_root: Path, stem: str) -> list[tuple[float, float, float, float]] | None:
    """-> [(cx, cy, w, h)] in native pixels, or None when there is no label."""
    path = labels_root / f"{stem}.txt"
    if not path.is_file():
        return None
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cx, cy, w, h = (float(x) for x in p[1:5])
        boxes.append((cx * CANVAS_W, cy * CANVAS_H, w * CANVAS_W, h * CANVAS_H))
    return boxes


def inbox(px: float, py: float, box: tuple[float, float, float, float]) -> bool:
    gx, gy, gw, gh = box
    return abs(px - gx) <= gw / 2.0 and abs(py - gy) <= gh / 2.0


def pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    a = sorted(values)
    if len(a) == 1:
        return a[0]
    idx = min(len(a) - 1, max(0, int(round(q * (len(a) - 1)))))
    return a[idx]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="基准路径的 TSV（dense）")
    ap.add_argument("--new", required=True, help="候选路径的 TSV（sparse / gpu）")
    ap.add_argument("--labels-root", default="", help="GT 标签目录（可选，用于 inbox 判定）")
    ap.add_argument("--seq", default="", help="序列名（只用于报告标题）")
    ap.add_argument("--top-k", type=int, default=1, help="每帧参与配对的检测数（按 peak 降序）")
    ap.add_argument("--out-md", default="", help="markdown 报告输出路径")
    ap.add_argument("--out-json", default="", help="json 报告输出路径")
    args = ap.parse_args()

    base = parse_mosaic_tsv(Path(args.base))
    new = parse_mosaic_tsv(Path(args.new))
    labels = Path(args.labels_root) if args.labels_root else None

    base_stems = set(base)
    new_stems = set(new)
    common = base_stems & new_stems
    only_base = base_stems - new_stems
    only_new = new_stems - base_stems

    hit_stems = sorted(
        (s for s in common if base[s] and new[s]), key=frame_index
    )  # 命中帧：两边都有检出
    base_only_empty = sorted(
        (s for s in common if base[s] and not new[s]), key=frame_index
    )
    new_only_empty = sorted(
        (s for s in common if new[s] and not base[s]), key=frame_index
    )

    d_cx: list[float] = []
    d_cy: list[float] = []
    d_dist: list[float] = []
    d_peak: list[float] = []
    d_w: list[float] = []
    d_h: list[float] = []
    d_area: list[float] = []
    d_sum: list[float] = []
    tag_mismatch = 0
    n_pairs = 0
    n_base_dets = 0
    n_new_dets = 0

    inbox_base = 0
    inbox_new = 0
    inbox_both = 0
    inbox_flip = 0
    inbox_base_only = 0
    inbox_new_only = 0
    gt_frames = 0
    gt_missing_base = 0
    gt_missing_new = 0

    for stem in hit_stems:
        bd = sorted(base[stem], key=lambda d: -d[6])[: args.top_k]
        nd = sorted(new[stem], key=lambda d: -d[6])[: args.top_k]
        n_base_dets += len(base[stem])
        n_new_dets += len(new[stem])
        for b, n in zip(bd, nd):
            n_pairs += 1
            d_cx.append(n[1] - b[1])
            d_cy.append(n[2] - b[2])
            d_dist.append(math.hypot(n[1] - b[1], n[2] - b[2]))
            d_peak.append(n[6] - b[6])
            d_w.append(n[3] - b[3])
            d_h.append(n[4] - b[4])
            d_area.append(n[5] - b[5])
            d_sum.append(n[7] - b[7])
            if b[0] != n[0]:
                tag_mismatch += 1
        if labels is not None:
            boxes = load_gt(labels, stem)
            if boxes:
                gt_frames += 1
                b0 = sorted(base[stem], key=lambda d: -d[6])[:1]
                n0 = sorted(new[stem], key=lambda d: -d[6])[:1]
                hb = bool(b0) and any(inbox(b0[0][1], b0[0][2], box) for box in boxes)
                hn = bool(n0) and any(inbox(n0[0][1], n0[0][2], box) for box in boxes)
                inbox_base += int(hb)
                inbox_new += int(hn)
                inbox_both += int(hb and hn)
                inbox_flip += int(hb != hn)
                inbox_base_only += int(hb and not hn)
                inbox_new_only += int(hn and not hb)
                gt_missing_base += int(not hb)
                gt_missing_new += int(not hn)

    def stats(v: list[float]) -> str:
        if not v:
            return "n/a"
        return (
            f"p50={pct(v, 0.5):.4f} p90={pct(v, 0.9):.4f} p95={pct(v, 0.95):.4f} "
            f"p99={pct(v, 0.99):.4f} max={max(v):.4f} mean={st.mean(v):.4f}"
        )

    lines = []
    lines.append("# 命中帧对比报告")
    lines.append("")
    if args.seq:
        lines.append(f"* 序列：`{args.seq}`")
    lines.append(f"* 基准：`{args.base}`")
    lines.append(f"* 候选：`{args.new}`")
    lines.append(f"* 每帧配对数 top-k：**{args.top_k}**（按 peak 降序）")
    lines.append("")
    lines.append("## 1. 帧集合")
    lines.append("")
    lines.append("| 项 | 帧数 |")
    lines.append("|:--|--:|")
    lines.append(f"| 基准 TSV 帧数 | {len(base_stems)} |")
    lines.append(f"| 候选 TSV 帧数 | {len(new_stems)} |")
    lines.append(f"| 共有帧 | {len(common)} |")
    lines.append(f"| 仅基准有 | {len(only_base)} |")
    lines.append(f"| 仅候选有 | {len(only_new)} |")
    lines.append(f"| **命中帧**（两边都有检出） | **{len(hit_stems)}** |")
    lines.append(f"| 共有帧中：基准有 / 候选空 | {len(base_only_empty)} |")
    lines.append(f"| 共有帧中：候选有 / 基准空 | {len(new_only_empty)} |")
    lines.append("")
    lines.append("## 2. 命中帧上的逐目标差异")
    lines.append("")
    lines.append(f"配对目标数：**{n_pairs}**（命中帧里基准总检出 {n_base_dets} / 候选总检出 {n_new_dets}）")
    lines.append("")
    lines.append("| 量 | 统计 |")
    lines.append("|:--|:--|")
    lines.append(f"| Δcx (px) | {stats(d_cx)} |")
    lines.append(f"| Δcy (px) | {stats(d_cy)} |")
    lines.append(f"| **Δ中心距 (px)** | {stats(d_dist)} |")
    lines.append(f"| Δpeak | {stats(d_peak)} |")
    lines.append(f"| Δw (px) | {stats(d_w)} |")
    lines.append(f"| Δh (px) | {stats(d_h)} |")
    lines.append(f"| Δarea (px) | {stats(d_area)} |")
    lines.append(f"| Δsum | {stats(d_sum)} |")
    lines.append(f"| tag 不一致的配对数 | {tag_mismatch} |")
    lines.append("")
    if labels is not None:
        lines.append("## 3. inbox 判定（中心点是否落在 GT 框内）")
        lines.append("")
        lines.append(
            "> 判定式 `abs(px-gx) <= gw/2 and abs(py-gy) <= gh/2`；"
            "预测自身的 w/h **不参与**判定（与 PC 评分器 `_hits` 同口径）。"
        )
        lines.append("")
        lines.append("| 项 | 帧数 |")
        lines.append("|:--|--:|")
        lines.append(f"| 有 GT 的命中帧 | {gt_frames} |")
        lines.append(f"| 基准命中 inbox | {inbox_base} |")
        lines.append(f"| 候选命中 inbox | {inbox_new} |")
        lines.append(f"| 两边都命中 | {inbox_both} |")
        lines.append(f"| **判定翻转（一边命中一边不中）** | **{inbox_flip}** |")
        lines.append(f"| 仅基准命中（候选丢） | {inbox_base_only} |")
        lines.append(f"| 仅候选命中（候选多） | {inbox_new_only} |")
        lines.append("")
    else:
        lines.append("## 3. inbox 判定")
        lines.append("")
        lines.append("> 未提供 `--labels-root`，跳过。")
        lines.append("")

    text = "\n".join(lines) + "\n"
    if args.out_md:
        Path(args.out_md).write_text(text, encoding="utf-8")
    print(text)

    if args.out_json:
        payload = {
            "base": str(args.base),
            "new": str(args.new),
            "frames": {
                "base": len(base_stems),
                "new": len(new_stems),
                "common": len(common),
                "only_base": len(only_base),
                "only_new": len(only_new),
                "hit": len(hit_stems),
                "base_only_empty": len(base_only_empty),
                "new_only_empty": len(new_only_empty),
            },
            "pairs": n_pairs,
            "delta_cx": d_cx,
            "delta_cy": d_cy,
            "delta_dist": d_dist,
            "delta_peak": d_peak,
            "tag_mismatch": tag_mismatch,
            "inbox": {
                "gt_frames": gt_frames,
                "base": inbox_base,
                "new": inbox_new,
                "both": inbox_both,
                "flip": inbox_flip,
                "base_only": inbox_base_only,
                "new_only": inbox_new_only,
            },
        }
        Path(args.out_json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
