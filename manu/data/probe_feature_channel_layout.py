#!/usr/bin/env python
"""判定各特征集**磁盘上的通道序**（用原始帧做锚点，判定哪一个通道是 I_t）。

背景
----
同一个模型不可能同时适配两种通道序。混训要求龙泉山与 Anti-UAV 两个训练集
**磁盘通道序完全一致**，否则 batch 内两侧不同构，实验直接报废。

判定规则：某通道与原始帧的皮尔逊相关系数最高 ⇒ 该通道就是 I_t。

⚠️ 锚点必须与特征帧**同序列同帧号**，否则相关系数毫无意义（历史踩坑：
拿 014053 的原始帧去比 002001 的特征，得出 B=-0.72 的假负相关）。
本脚本因此**从特征文件名自动推导原始帧路径**，不再硬编码锚点。

用法::

    python manu/data/probe_feature_channel_layout.py \\
        --out-json runs/logs/feature_channel_layout.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np

D = Path("/mnt/data/siping/datasets/manu")
LONGQUAN_RAW_ROOT = D / "龙泉山/frames_ir_jpg"
ANTIUAV_RAW_ROOT = D / "anti-uav/train"


def derive_raw_path(feat_path: Path) -> Path | None:
    """从特征文件名推导同序列同帧号的原始帧路径。"""
    stem = feat_path.stem
    m_long = re.match(r"^(?P<seq>.+?)__frame_(?P<frame>\d+)$", stem)
    if m_long:
        return LONGQUAN_RAW_ROOT / m_long.group("seq") / f"frame_{int(m_long.group('frame')):06d}.jpg"
    m_uav = re.match(r"^(?P<seq>.+?)__(?P<frame>\d+)$", stem)
    if m_uav:
        return ANTIUAV_RAW_ROOT / m_uav.group("seq") / f"{int(m_uav.group('frame')):06d}.jpg"
    return None


def analyse(name: str, feat_path: Path, raw_path: Path | None) -> dict:
    if not feat_path.is_file():
        return {"name": name, "status": "feature_missing", "path": str(feat_path)}
    feat = cv2.imread(str(feat_path), cv2.IMREAD_COLOR)
    if feat is None:
        return {"name": name, "status": "feature_unreadable", "path": str(feat_path)}
    channels = [feat[:, :, i].astype(np.float32) for i in range(3)]
    out = {
        "name": name,
        "status": "ok",
        "path": str(feat_path),
        "channel_mean": [round(float(c.mean()), 3) for c in channels],
        "channel_max": [int(c.max()) for c in channels],
        "channel_nonzero_ratio": [round(float((c > 0).mean()), 4) for c in channels],
    }
    if raw_path is None:
        out["status"] = "raw_anchor_not_derivable"
        return out
    if not raw_path.is_file():
        out["status"] = "raw_anchor_missing"
        out["raw_path"] = str(raw_path)
        return out

    raw = cv2.imread(str(raw_path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        out["status"] = "raw_unreadable"
        out["raw_path"] = str(raw_path)
        return out
    raw = raw.astype(np.float32)
    if raw.shape != channels[0].shape:
        out["status"] = "raw_shape_mismatch"
        out["raw_path"] = str(raw_path)
        out["raw_shape"] = list(raw.shape)
        out["feature_shape"] = list(channels[0].shape)
        return out

    corr = []
    for c in channels:
        a = c.ravel() - c.mean()
        b = raw.ravel() - raw.mean()
        denom = float(np.sqrt((a * a).sum() * (b * b).sum())) or 1.0
        corr.append(round(float((a * b).sum() / denom), 6))
    out["raw_path"] = str(raw_path)
    out["pearson_with_raw"] = corr
    out["it_channel_index_in_BGR_read"] = int(np.argmax(corr))
    out["it_channel_name"] = ["B", "G", "R"][int(np.argmax(corr))]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="判定各特征集的磁盘通道序")
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    D = Path("/mnt/data/siping/datasets/manu")
    cases = [
        ("龙泉山 FPV 旧缓存(train)", D / "longquanshan_ir_gmc_median/images/train/VIDEO00029_19700101_014053__frame_000300.jpg"),
        ("龙泉山 official 新数据集(train)", D / "longquanshan_ft_gmc_median_official/images/train/VIDEO00008_19700101_002001__frame_000000.jpg"),
        ("龙泉山 FPV 新数据集(train)", D / "longquanshan_ft_gmc_median/images/train/VIDEO00008_19700101_002001__frame_000000.jpg"),
        ("龙泉山 official 新数据集(val)", D / "longquanshan_ft_gmc_median_official/images/val/VIDEO00029_19700101_014053__frame_000300.jpg"),
        ("Anti-UAV uav_gmc_median(train)", D / "uav_gmc_median/images/train/01_1751_0250-1750__000006.jpg"),
        ("Anti-UAV uav_gmc_median(val)", D / "uav_gmc_median/images/val/01_4485_1167-2666__000003.jpg"),
        ("Anti-UAV official_rebuild(val)", D / "uav_gmc_median_official_rebuild/images/val/01_4485_1167-2666__000003.jpg"),
    ]

    report = {"cases": []}
    it_index: dict[str, int] = {}
    print("=" * 104)
    for name, feat in cases:
        raw = derive_raw_path(feat)
        info = analyse(name, feat, raw)
        report["cases"].append(info)
        if info.get("status") != "ok":
            print(f"  {name:<36} -> {info['status']}  ({info.get('raw_path', '-')})")
            continue
        corr = info["pearson_with_raw"]
        idx = info["it_channel_index_in_BGR_read"]
        it_index[name] = idx
        print(
            f"  {name:<36} 相关 B={corr[0]:+.4f} G={corr[1]:+.4f} R={corr[2]:+.4f} "
            f"⇒ I_t 在 {info['it_channel_name']} 通道 (idx={idx})"
        )
    print("=" * 104)

    # 同构性判定：参与混训的两个训练集必须落在同一通道
    train_sets = ["龙泉山 official 新数据集(train)", "Anti-UAV uav_gmc_median(train)"]
    idxs = [it_index.get(n) for n in train_sets]
    if all(i is not None for i in idxs):
        homogeneous = len(set(idxs)) == 1
        report["train_sets_homogeneous"] = homogeneous
        report["train_set_it_channel_indices"] = dict(zip(train_sets, idxs))
        print(
            f"  混训两侧磁盘通道同构: {'是 ✅' if homogeneous else '否 ❌'}  "
            f"({', '.join(f'{n}=idx{it_index[n]}' for n in train_sets)})"
        )
    else:
        report["train_sets_homogeneous"] = None
        print("  混训两侧磁盘通道同构: 无法判定（锚点缺失）")

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"报告已写入: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
