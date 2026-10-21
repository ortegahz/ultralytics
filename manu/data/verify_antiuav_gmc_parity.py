#!/usr/bin/env python
"""证明本项目构建器的 ``official`` 模式 ≡ Anti-UAV 官方特征集构建方式。

做法（**直接对标官方产物，不靠"看起来一样"**）：
    1. 取 Anti-UAV 官方特征集 ``uav_gmc_median/images/val/{seq}__{idx:06d}.jpg`` 中的样本帧号；
    2. 用本项目 ``build_longquan_ft_dataset.py::iter_sequence_features``（``semantics="official"``）
       在**同样的 Anti-UAV 原始帧**上重算特征；
    3. 与官方产物逐字节（sha256）比对。

只要抽样全部逐字节相等，就说明 official 模式产出的特征**与 Anti-UAV 官方特征集完全同源**，
不存在通道序、GMC 拟合方式、中值窗口、编码参数等任何差异。

用法::

    python manu/data/verify_antiuav_gmc_parity.py \\
        --raw-root "/mnt/data/siping/datasets/manu/anti-uav/train" \\
        --canonical-root "/mnt/data/siping/datasets/manu/uav_gmc_median/images/val" \\
        --sequences 01_4485_1167-2666,01_1751_0250-1750,02_6321_0274-2773 \\
        --sample 25 --out-json runs/logs/antiuav_gmc_parity.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu.data.build_full_median_dataset import natural_key  # noqa: E402
from manu.data.build_longquan_ft_dataset import iter_sequence_features  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def trailing_digits(stem: str) -> int | None:
    match = re.search(r"(\d+)$", stem)
    return int(match.group(1)) if match else None


def main() -> int:
    parser = argparse.ArgumentParser(description="验证 official 语义 ≡ Anti-UAV 官方特征集")
    parser.add_argument("--raw-root", required=True, help="Anti-UAV 原始帧根（含 <seq>/ 子目录）")
    parser.add_argument("--canonical-root", required=True, help="uav_gmc_median/images/val")
    parser.add_argument("--sequences", default="", help="逗号分隔；为空则取 canonical 目录里前 3 条")
    parser.add_argument("--sample", type=int, default=25, help="每条序列抽样帧数")
    parser.add_argument("--window", type=int, default=21)
    parser.add_argument("--stride-step", type=int, default=2)
    parser.add_argument("--downscale", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    canonical = Path(args.canonical_root)
    if not canonical.is_dir():
        print(f"[FAIL] 官方特征集目录不存在: {canonical}")
        return 2

    if args.sequences:
        sequences = [s.strip() for s in args.sequences.split(",") if s.strip()]
    else:
        sequences = sorted({p.stem.split("__")[0] for p in canonical.glob("*.jpg")})[:3]

    rng = random.Random(args.seed)
    report = {
        "raw_root": str(raw_root),
        "canonical_root": str(canonical),
        "window": args.window,
        "stride_step": args.stride_step,
        "downscale": args.downscale,
        "sequences": {},
    }
    total = identical = 0
    failures: list[str] = []

    for seq in sequences:
        seq_dir = raw_root / seq
        if not seq_dir.is_dir():
            failures.append(f"{seq}: 原始目录不存在 {seq_dir}")
            continue
        frames = [f for f in seq_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES]
        frames.sort(key=natural_key)
        index_map: dict[int, int] = {}
        for position, frame in enumerate(frames):
            digits = trailing_digits(frame.stem)
            index_map.setdefault(digits if digits is not None else position, position)

        # 官方特征集里该序列的帧号
        cand = sorted(p for p in canonical.glob(f"{seq}__*.jpg"))
        frame_ids = []
        for path in cand:
            digits = trailing_digits(path.stem.split("__", 1)[1])
            if digits is not None:
                frame_ids.append(digits)
        if not frame_ids:
            failures.append(f"{seq}: 官方特征集中找不到该序列")
            continue
        picks = rng.sample(frame_ids, min(args.sample, len(frame_ids)))

        # 用 official 语义重算这些帧
        computed = {
            frame_id: img
            for frame_id, img in iter_sequence_features(
                seq_dir, picks, args.window, args.stride_step, args.downscale, semantics="official"
            )
        }

        seq_identical = 0
        details = []
        for frame_id in picks:
            total += 1
            img = computed.get(frame_id)
            canon_path = canonical / f"{seq}__{frame_id:06d}.jpg"
            if img is None:
                failures.append(f"{seq}___{frame_id:06d}: 重算失败")
                details.append({"frame": frame_id, "status": "compute_failed"})
                continue
            canon_bytes = canon_path.read_bytes()
            mine_bytes = cv2.imencode(".jpg", img)[1].tobytes()
            same = sha256_bytes(mine_bytes) == sha256_bytes(canon_bytes)
            if same:
                seq_identical += 1
                identical += 1
            else:
                canon_img = cv2.imdecode(np.frombuffer(canon_bytes, np.uint8), cv2.IMREAD_COLOR)
                diff = np.abs(img.astype(np.int16) - canon_img.astype(np.int16))
                details.append(
                    {
                        "frame": frame_id,
                        "status": "mismatch",
                        "decoded_max_abs_diff": int(diff.max()),
                        "decoded_mean_abs_diff": round(float(diff.mean()), 4),
                    }
                )
                failures.append(f"{seq}___{frame_id:06d}: 与官方特征集不一致 maxdiff={int(diff.max())}")

        report["sequences"][seq] = {
            "raw_frames": len(frames),
            "canonical_frames": len(frame_ids),
            "sampled": len(picks),
            "byte_identical": seq_identical,
            "mismatches": [d for d in details if d.get("status") == "mismatch"][:10],
        }
        print(
            f"  {seq:<26} 原始 {len(frames):>5} 帧 | 官方 {len(frame_ids):>5} 帧 | 抽样 {len(picks):>3} "
            f"→ 逐字节相同 {seq_identical}/{len(picks)}"
        )

    report["summary"] = {
        "sampled": total,
        "byte_identical": identical,
        "mismatch": total - identical,
        "verdict": "PARITY_OK" if total and identical == total else "PARITY_FAIL",
    }

    print("=" * 92)
    print(f"判定: {report['summary']['verdict']}  逐字节相同 {identical}/{total}")
    if failures:
        print("不一致明细：")
        for item in failures[:20]:
            print(f"   - {item}")

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"报告已写入: {out}")
    return 0 if report["summary"]["verdict"] == "PARITY_OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
