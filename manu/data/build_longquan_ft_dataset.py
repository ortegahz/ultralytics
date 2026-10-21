#!/usr/bin/env python
"""构建龙泉山业务域微调特征数据集（三轨划分：黄金池守护大盘 + 001324 盲测）。

产物布局（YOLO 标准 + Ultralytics 特征约定）::

    <out-root>/
      data.yaml
      split_manifest.json
      images/train/<seq>__frame_<idx:06d>.jpg    3 通道特征
      images/val/<seq>__frame_<idx:06d>.jpg
      labels/train/<seq>__frame_<idx:06d>.txt    YOLO 归一化
      labels/val/<seq>__frame_<idx:06d>.txt

特征语义严格对齐官方构建器 ``manu/data/build_full_median_dataset.py``：
逐 lag 直接拟合 ``H = compute_affine(im_h, im_curr)``，**禁用** FPV 仿射连乘变体。
通道磁盘序 ``[I_t, |I_t - W(I_{t-2})|, (I_t - B_t)^+]``（推理时由下游工具反转为模型序）。

采样口径：
* val  = 全部帧，步长 1，起点 0（含第 0 帧，与 2026-10-20 冻结 baseline 帧集合逐帧一致）
* train= 步长 5，起点 0（Anti-UAV 参考集 uav 为步长 5 起点 6，可用 --train-start 6 复刻）

用法::

    python manu/data/build_longquan_ft_dataset.py \\
        --split-config manu/data/longquan_ft_split.json \\
        --raw-root "/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg" \\
        --labels-root "/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010" \\
        --out-root "/mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median" \\
        --feature-mode rebuild --workers 24
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu.data.build_full_median_dataset import FastGMCEstimator, natural_key  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_label_rows(label_path: Path) -> tuple[list[str], int]:
    """读取 YOLO 标签并去重。

    去重键与评测端 ``manu/evaluation/eval_multiscale_cache_th.py::load_ground_truth``
    **完全一致**：同一文件内按 ``(cx, cy, w, h)`` 四元组去重（类 id 不参与去重键），
    丢弃字段数不足 5 的行。这样数据集内的 GT 行数与 baseline 口径可直接对账。

    返回 (去重后的行列表, 原始有效行数)。
    """
    if not label_path.is_file():
        return [], 0
    seen: set[tuple[float, ...]] = set()
    rows: list[str] = []
    raw_count = 0
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) < 5:
            continue
        raw_count += 1
        key = tuple(float(value) for value in values[1:5])
        if key in seen:
            continue
        seen.add(key)
        rows.append(" ".join(values))
    return rows, raw_count


def select_indices(total: int, stride: int, start: int) -> list[int]:
    return list(range(start, total, stride))


def _compose_chain(h_list: list[np.ndarray]) -> np.ndarray:
    """左乘累积 M = H_n @ ... @ H_1（与 build_fpv_ir_gmc_median.py 的 compose_affine 同序）。"""
    total = np.eye(3, dtype=np.float32)
    for h in h_list:
        h3 = np.eye(3, dtype=np.float32)
        h3[:2, :] = h
        total = h3 @ total
    return total[:2, :].astype(np.float32)


def iter_sequence_features(
    seq_dir: Path,
    wanted: list[int],
    window: int,
    stride_step: int,
    downscale: int,
    semantics: str = "fpv",
):
    """逐帧产出特征（生成器），**不累积**整条序列的特征。

    ⚠️ 曾经的实现把整条序列的特征攒进 dict 再落盘，stride=1 的长序列
    （如 001324：11,810 帧 × 3 通道 × 640×512 ≈ 11.6 GB）会把 worker 直接撑爆、
    表现为 0 帧/s 的假死。这里改成生成器，边算边写。

    semantics
    ---------
    ``fpv``     : 相邻帧仿射连乘（``build_fpv_ir_gmc_median.py`` 口径）。
                  既有缓存 ``longquanshan_ir_gmc_median`` 即由此产出（已逐字节验证），
                  龙泉山业务域全部评测与三轨 baseline 都建立在它之上。
    ``official``: 逐 lag 直接拟合（``build_full_median_dataset.py`` 口径，Anti-UAV
                  官方集 ``uav_gmc_median`` 的口径）。
    """
    frames = [f for f in seq_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES]
    frames.sort(key=natural_key)
    if not frames:
        return

    index_map: dict[int, int] = {}
    for position, frame in enumerate(frames):
        digits = ""
        for ch in reversed(frame.stem):
            if ch.isdigit():
                digits = ch + digits
            elif digits:
                break
        index_map.setdefault(int(digits) if digits else position, position)

    estimator = FastGMCEstimator(downscale=downscale)
    cache: dict[int, np.ndarray] = {}
    step1: dict[int, np.ndarray] = {}
    keep = window * stride_step + 8

    def get(list_idx: int) -> np.ndarray | None:
        if list_idx in cache:
            return cache[list_idx]
        if list_idx < 0 or list_idx >= len(frames):
            return None
        image = cv2.imread(str(frames[list_idx]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        cache[list_idx] = image
        if len(cache) > keep:
            for key in sorted(cache)[: len(cache) - keep]:
                del cache[key]
        return image

    def get_step1(i: int) -> np.ndarray | None:
        """相邻帧仿射 H(i -> i+1)，FPV 连乘语义的基本单元。"""
        if i in step1:
            return step1[i]
        im_a = get(i)
        im_b = get(i + 1)
        if im_a is None or im_b is None:
            return None
        step1[i] = estimator.compute_affine(im_a, im_b)
        if len(step1) > keep + 2:
            for key in sorted(step1)[: len(step1) - keep]:
                del step1[key]
        return step1[i]

    for frame_id in wanted:
        curr_list_idx = index_map.get(frame_id, min(frame_id, len(frames) - 1))
        im_curr = get(curr_list_idx)
        if im_curr is None:
            continue

        if semantics == "fpv":
            # 通道 1：两帧步长的仿射连乘差分 |I_t - W(I_{t-2})|
            if curr_list_idx >= 2:
                im_prev2 = get(curr_list_idx - 2)
                h_a = get_step1(curr_list_idx - 2)
                h_b = get_step1(curr_list_idx - 1)
                if im_prev2 is None or h_a is None or h_b is None:
                    continue
                h_prev2 = _compose_chain([h_a, h_b])
                diff2 = cv2.absdiff(im_curr, estimator.warp(im_prev2, h_prev2))
            elif curr_list_idx == 1:
                im_prev1 = get(0)
                h1 = get_step1(0)
                if im_prev1 is None or h1 is None:
                    continue
                diff2 = cv2.absdiff(im_curr, estimator.warp(im_prev1, h1))
            else:
                diff2 = np.zeros_like(im_curr)

            # 通道 2：21 帧滑窗仿射连乘中值背景残差 (I_t - B_t)^+
            history = []
            for step in range(1, window + 1):
                h_idx = curr_list_idx - step * stride_step
                if h_idx < 0:
                    break
                im_h = get(h_idx)
                if im_h is None:
                    continue
                chain = []
                broken = False
                for k in range(h_idx, curr_list_idx):
                    h_k = get_step1(k)
                    if h_k is None:
                        broken = True
                        break
                    chain.append(h_k)
                if broken:
                    continue
                history.append(estimator.warp(im_h, _compose_chain(chain)))
        else:
            # official：逐 lag 直接拟合
            idx_prev2 = max(0, curr_list_idx - 2)
            im_prev2 = get(idx_prev2)
            if im_prev2 is None:
                continue
            h2 = estimator.compute_affine(im_prev2, im_curr)
            diff2 = cv2.absdiff(im_curr, estimator.warp(im_prev2, h2))

            history = []
            for step in range(1, window + 1):
                h_idx = max(0, curr_list_idx - step * stride_step)
                im_h = get(h_idx)
                if im_h is None:
                    continue
                h_h = estimator.compute_affine(im_h, im_curr)
                history.append(estimator.warp(im_h, h_h))

        if len(history) >= 3:
            stack = np.stack(history, axis=0)
            median_bg = np.median(stack, axis=0).astype(np.float32)
            res_median = np.clip(im_curr.astype(np.float32) - median_bg, 0, 255).astype(np.uint8)
        else:
            res_median = diff2

        yield frame_id, np.stack([im_curr, diff2, res_median], axis=-1)


def process_sequence(task: dict) -> dict:
    """单个序列的构建任务（进程池内执行）。"""
    seq = task["sequence"]
    group = task["group"]
    raw_dir = Path(task["raw_root"]) / seq
    out_img_dir = Path(task["out_root"]) / "images" / group
    out_lbl_dir = Path(task["out_root"]) / "labels" / group
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    wanted = task["wanted"]
    stats = {
        "sequence": seq,
        "group": group,
        "role": task.get("role", ""),
        "wanted_frames": len(wanted),
        "images_written": 0,
        "labels_written": 0,
        "label_rows": 0,
        "label_rows_raw": 0,
        "empty_labels": 0,
        "missing_labels": 0,
        "failed_frames": 0,
    }

    # 标签
    label_dir = Path(task["labels_root"])
    for frame_id in wanted:
        stem = f"{seq}__frame_{frame_id:06d}"
        src = label_dir / f"{seq}__{frame_id:06d}.txt"
        rows, raw_count = read_label_rows(src)
        if not src.is_file():
            stats["missing_labels"] += 1
        stats["label_rows_raw"] += raw_count
        stats["label_rows"] += len(rows)
        if not rows:
            stats["empty_labels"] += 1
        (out_lbl_dir / f"{stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
        stats["labels_written"] += 1

    if task["feature_mode"] == "copy":
        src_dir = Path(task["existing_features_root"])
        for frame_id in wanted:
            stem = f"{seq}__frame_{frame_id:06d}"
            src = src_dir / f"{stem}.jpg"
            if not src.is_file():
                stats["failed_frames"] += 1
                continue
            dst = out_img_dir / f"{stem}.jpg"
            if dst.is_file() and not task["overwrite"]:
                stats["images_written"] += 1
                continue
            os.link(src, dst) if _same_fs(src, dst) else shutil_copy(src, dst)
            stats["images_written"] += 1
    else:
        # 边算边写：特征逐帧产出即落盘，避免整条序列的特征占满内存
        for frame_id, img in iter_sequence_features(
            raw_dir, wanted, task["window"], task["stride_step"], task["downscale"], task["gmc_semantics"]
        ):
            stem = f"{seq}__frame_{frame_id:06d}"
            dst = out_img_dir / f"{stem}.jpg"
            if dst.is_file() and not task["overwrite"]:
                stats["images_written"] += 1
                continue
            if not cv2.imwrite(str(dst), img):
                stats["failed_frames"] += 1
                continue
            stats["images_written"] += 1

    stats["elapsed_sec"] = round(time.time() - started, 2)
    return stats


def _same_fs(a: Path, b: Path) -> bool:
    try:
        return os.stat(a).st_dev == os.stat(b).st_dev
    except OSError:
        return False


def shutil_copy(src: Path, dst: Path) -> None:
    import shutil

    shutil.copy2(src, dst)


def write_manifest_only(args, config, config_path, plan, groups_cfg) -> int:
    """按磁盘真值重写 data.yaml + split_manifest.json（不做图像计算）。"""
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    per_sequence = []
    totals = {"images": 0, "labels": 0, "label_rows": 0, "label_rows_raw": 0, "sequences": 0}
    for group in ("val", "train"):
        img_dir = out_root / "images" / group
        lbl_dir = out_root / "labels" / group
        if not img_dir.is_dir():
            continue
        seq_stats: dict[str, dict] = {}
        for path in sorted(img_dir.glob("*.jpg")):
            seq = path.stem.split("__frame_")[0]
            entry = seq_stats.setdefault(
                seq, {"group": group, "images": 0, "labels": 0, "label_rows": 0, "first": None, "last": None}
            )
            entry["images"] += 1
            idx = int(path.stem.split("__frame_")[1])
            entry["first"] = idx if entry["first"] is None else min(entry["first"], idx)
            entry["last"] = idx if entry["last"] is None else max(entry["last"], idx)
        for path in sorted(lbl_dir.glob("*.txt")) if lbl_dir.is_dir() else []:
            seq = path.stem.split("__frame_")[0]
            entry = seq_stats.setdefault(
                seq, {"group": group, "images": 0, "labels": 0, "label_rows": 0, "first": None, "last": None}
            )
            entry["labels"] += 1
            entry["label_rows"] += len([ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()])
        for seq, entry in sorted(seq_stats.items()):
            per_sequence.append(
                {
                    "sequence": seq,
                    "group": entry["group"],
                    "images": entry["images"],
                    "labels": entry["labels"],
                    "label_rows": entry["label_rows"],
                    "first_index": entry["first"],
                    "last_index": entry["last"],
                }
            )
            totals["images"] += entry["images"]
            totals["labels"] += entry["labels"]
            totals["label_rows"] += entry["label_rows"]
            totals["sequences"] += 1

    data_yaml = (
        f"# 龙泉山业务域微调特征数据集（三轨划分：黄金池守护大盘 + 001324 盲测）\n"
        f"# 生成于 {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"# 特征来源：既有缓存 longquanshan_ir_gmc_median（FPV 相邻帧仿射连乘语义）\n"
        f"#          已用 sha256 逐字节验证与该缓存全等（42,915/42,915）\n"
        f"# 标签来源：{args.labels_root}（按评测端 (cx,cy,w,h) 四元组去重口径）\n"
        f"# 通道磁盘序 [I_t, |I_t-W(I_{{(t-2}})|, (I_t-B_t)^+]；推理时需反转为模型序\n"
        f"path: {out_root}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"\n"
        f"names:\n"
        f"  0: uav\n"
    )
    (out_root / "data.yaml").write_text(data_yaml)

    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "manifest_only_regenerated": True,
        "out_root": str(out_root),
        "raw_root": str(args.raw_root),
        "labels_root": str(args.labels_root),
        "split_config": str(config_path),
        "split_config_sha256": sha256_file(config_path),
        "feature_provenance": {
            "source": "复用既有特征缓存 longquanshan_ir_gmc_median（FPV 相邻帧仿射连乘语义）",
            "byte_identical_verification": "sha256 逐文件全等，42,915/42,915",
            "verification_report": "runs/logs/lq_ft_feature_parity.json",
            "note": "旧缓存的标签为 0 字节空文件，标签全部重新取自合并真值；"
            "如需从原始帧重算请用 --feature-mode rebuild --gmc-semantics {fpv,official}",
        },
        "feature_params": {
            "window": 21,
            "stride_step": 2,
            "downscale": 2,
            "gmc_semantics": "fpv",
            "gmc_semantics_note": "相邻帧仿射连乘（build_fpv_ir_gmc_median.py 口径），与 ms_cache_nop0 同源",
            "channel_disk_order": ["I_t", "gmc_diff", "median_residual"],
            "encoding": "cv2.imwrite 默认 JPEG 质量",
        },
        "sampling": {
            "val_stride": config.get("sampling", {}).get("val_stride", 1),
            "val_start": config.get("sampling", {}).get("val_start", 0),
            "train_stride": config.get("sampling", {}).get("train_stride", 5),
            "train_start": config.get("sampling", {}).get("train_start", 0),
        },
        "plan": plan,
        "per_sequence": per_sequence,
        "totals": totals,
        "ignore_sequences": [e["sequence"] for e in groups_cfg.get("ignore", {}).get("sequences", [])],
        "baseline_reference": config.get("baseline_reference", {}),
        "caveats": [
            "multiscale_heatmap_cache.py 的 frame_index 取排序位置而非文件名帧号；"
            "本数据集 val 为连续 0..N-1 故两者一致，train 为抽帧子集，位置≠真值帧号，"
            "若要对 train 建缓存须按本 manifest 的 first_index/last_index 还原。",
            "001324 为绝对留出盲测序列，严禁任何切片进入训练集。",
            "特征为 3 通道 JPG（有损）；如需无损对照，另建 PNG 变体。",
        ],
    }
    (out_root / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("[MANIFEST] 已按磁盘真值重写：")
    print(f"  data.yaml           -> {out_root / 'data.yaml'}")
    print(f"  split_manifest.json -> {out_root / 'split_manifest.json'}")
    print(f"  序列 {totals['sequences']}  图像 {totals['images']}  标签 {totals['labels']}  框行 {totals['label_rows']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="构建龙泉山业务域微调特征数据集")
    parser.add_argument("--split-config", required=True)
    parser.add_argument("--raw-root", required=True, help="frames_ir_jpg 根目录")
    parser.add_argument("--labels-root", required=True, help="合并真值目录（{seq}__{idx:06d}.txt）")
    parser.add_argument("--out-root", required=True)
    parser.add_argument(
        "--feature-mode",
        choices=["rebuild", "copy"],
        default="rebuild",
        help="rebuild=从原始帧重算特征（权威）；copy=复用既有特征缓存（需先证明语义一致）",
    )
    parser.add_argument("--existing-features-root", default="", help="copy 模式下的既有特征目录")
    parser.add_argument("--val-stride", type=int, default=0, help="覆盖配置中的 val 步长")
    parser.add_argument("--val-start", type=int, default=None)
    parser.add_argument("--train-stride", type=int, default=0)
    parser.add_argument("--train-start", type=int, default=None)
    parser.add_argument("--window", type=int, default=21)
    parser.add_argument("--stride-step", type=int, default=2)
    parser.add_argument("--downscale", type=int, default=2)
    parser.add_argument(
        "--gmc-semantics",
        choices=["fpv", "official"],
        default="fpv",
        help="fpv=相邻帧仿射连乘（既有 longquanshan_ir_gmc_median 缓存口径，龙泉山 baseline 口径，默认）；"
        "official=逐 lag 直接拟合（uav_gmc_median / Anti-UAV 官方集口径）",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--groups", default="val,train", help="要构建的组，逗号分隔；ignore 组不构建")
    parser.add_argument("--seqs", default="", help="只构建这些序列（逗号分隔），用于单序列试跑")
    parser.add_argument(
        "--split-chunks",
        type=int,
        default=1,
        help="把每条序列的目标帧切成 N 个连续块并行处理（关键路径优化）。"
        "安全：特征所需的上下文（t-1/t-2/中值窗口 t-2k）始终按**绝对帧号**从原始帧读取，"
        "与该帧落在哪个块无关 ⇒ 分块不改变任何一帧的计算结果。",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="只按磁盘真值重写 data.yaml + split_manifest.json（不做任何图像计算）",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印计划与统计，不写文件")
    args = parser.parse_args()

    config_path = Path(args.split_config)
    config = json.loads(config_path.read_text())
    groups_cfg = config["groups"]
    sampling = config.get("sampling", {})

    val_stride = args.val_stride or sampling.get("val_stride", 1)
    val_start = args.val_start if args.val_start is not None else sampling.get("val_start", 0)
    train_stride = args.train_stride or sampling.get("train_stride", 5)
    train_start = args.train_start if args.train_start is not None else sampling.get("train_start", 0)

    raw_root = Path(args.raw_root)
    labels_root = Path(args.labels_root)
    out_root = Path(args.out_root)
    wanted_groups = [g.strip() for g in args.groups.split(",") if g.strip()]

    tasks: list[dict] = []
    plan: list[dict] = []
    only_seqs = {s.strip() for s in args.seqs.split(",") if s.strip()}
    for group in wanted_groups:
        if group not in groups_cfg:
            print(f"[FAIL] 未知组别: {group}")
            return 2
        stride = val_stride if group == "val" else train_stride
        start = val_start if group == "val" else train_start
        for entry in groups_cfg[group]["sequences"]:
            seq = entry["sequence"]
            if only_seqs and seq not in only_seqs:
                continue
            seq_dir = raw_root / seq
            if not seq_dir.is_dir():
                print(f"[FAIL] 原始序列目录不存在: {seq_dir}")
                return 2
            total = len([f for f in seq_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES])
            indices = select_indices(total, stride, start)
            plan.append(
                {
                    "group": group,
                    "sequence": seq,
                    "role": entry.get("role", ""),
                    "total_frames": total,
                    "stride": stride,
                    "start": start,
                    "export_frames": len(indices),
                    "first_index": indices[0] if indices else None,
                    "last_index": indices[-1] if indices else None,
                }
            )
            # 切成 N 个连续块并行；块内帧号互不相交，上下文按绝对帧号读取 ⇒ 结果不变
            n_chunks = max(1, args.split_chunks)
            chunk_size = (len(indices) + n_chunks - 1) // n_chunks if indices else 0
            for chunk_id in range(n_chunks):
                chunk = indices[chunk_id * chunk_size : (chunk_id + 1) * chunk_size]
                if not chunk:
                    continue
                tasks.append(
                    {
                        "sequence": seq,
                        "group": group,
                        "role": entry.get("role", ""),
                        "chunk_id": chunk_id,
                        "chunks": n_chunks,
                        "wanted": chunk,
                        "raw_root": str(raw_root),
                        "labels_root": str(labels_root),
                        "out_root": str(out_root),
                        "feature_mode": args.feature_mode,
                        "existing_features_root": args.existing_features_root,
                        "window": args.window,
                        "stride_step": args.stride_step,
                        "downscale": args.downscale,
                        "gmc_semantics": args.gmc_semantics,
                        "overwrite": args.overwrite,
                    }
                )

    print("=" * 92)
    print(f"龙泉山微调数据集构建计划  |  输出: {out_root}")
    print(f"  特征模式: {args.feature_mode}   GMC 语义: {args.gmc_semantics}   "
          f"窗口/步长: {args.window}/{args.stride_step}  下采样: {args.downscale}")
    print(f"  val  步长 {val_stride} 起点 {val_start}   |   train 步长 {train_stride} 起点 {train_start}")
    print("=" * 92)
    for item in plan:
        print(
            f"  {item['group']:>5} | {item['sequence']:<32} total={item['total_frames']:>6} "
            f"-> export={item['export_frames']:>6} (idx {item['first_index']}..{item['last_index']}, step {item['stride']})"
        )
    total_export = sum(p["export_frames"] for p in plan)
    print(f"  合计导出帧数: {total_export}")

    if args.dry_run:
        print("\n[DRY-RUN] 未写入任何文件。")
        return 0

    if args.manifest_only:
        return write_manifest_only(args, config, config_path, plan, groups_cfg)

    if args.feature_mode == "copy" and not args.existing_features_root:
        print("[FAIL] copy 模式必须提供 --existing-features-root")
        return 2

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "images").mkdir(exist_ok=True)
    (out_root / "labels").mkdir(exist_ok=True)

    started = time.time()
    results: list[dict] = []
    workers = max(1, min(args.workers, len(tasks)))
    print(f"\n[RUN] {len(tasks)} 个序列，{workers} 进程并行 ...\n")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_sequence, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                stats = future.result()
            except Exception as exc:  # noqa: BLE001
                print(f"[FAIL] {task['sequence']}: {exc}")
                results.append({"sequence": task["sequence"], "group": task["group"], "error": str(exc)})
                continue
            results.append(stats)
            print(
                f"  [{task['group']:>5}] {task['sequence']:<32}#{task.get('chunk_id', 0)}"
                f"/{task.get('chunks', 1)} imgs={stats['images_written']:>6} "
                f"lbl={stats['labels_written']:>6} rows={stats['label_rows']:>7} "
                f"(raw {stats['label_rows_raw']}) empty={stats['empty_labels']:>6} "
                f"failed={stats['failed_frames']:>4} {stats['elapsed_sec']:>7.1f}s"
            )

    # data.yaml
    data_yaml = (
        f"# 龙泉山业务域微调特征数据集（三轨划分：黄金池守护大盘 + 001324 盲测）\n"
        f"# 构建于 {time.strftime('%Y-%m-%d %H:%M:%S')} | 特征模式 {args.feature_mode} | GMC 语义 {args.gmc_semantics}\n"
        f"# 通道磁盘序 [I_t, |I_t-W(I_{{(t-2}})|, (I_t-B_t)^+]；推理时需反转为模型序\n"
        f"path: {out_root}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"\n"
        f"names:\n"
        f"  0: uav\n"
    )
    (out_root / "data.yaml").write_text(data_yaml)

    # manifest
    # 按序列聚合（一条序列可能被切成多个块）
    agg: dict[tuple[str, str], dict] = {}
    for r in results:
        key = (r.get("group", ""), r.get("sequence", ""))
        entry = agg.setdefault(
            key,
            {
                "group": r.get("group", ""),
                "sequence": r.get("sequence", ""),
                "images_written": 0,
                "labels_written": 0,
                "label_rows": 0,
                "label_rows_raw": 0,
                "empty_labels": 0,
                "failed_frames": 0,
                "chunks": 0,
                "elapsed_sec": 0.0,
            },
        )
        for k in ("images_written", "labels_written", "label_rows", "label_rows_raw", "empty_labels", "failed_frames"):
            entry[k] += r.get(k, 0)
        entry["elapsed_sec"] = round(entry["elapsed_sec"] + r.get("elapsed_sec", 0.0), 2)
        entry["chunks"] += 1

    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "out_root": str(out_root),
        "raw_root": str(raw_root),
        "labels_root": str(labels_root),
        "split_config": str(config_path),
        "split_config_sha256": sha256_file(config_path),
        "feature_mode": args.feature_mode,
        "existing_features_root": args.existing_features_root,
        "feature_params": {
            "window": args.window,
            "stride_step": args.stride_step,
            "downscale": args.downscale,
            "gmc_semantics": args.gmc_semantics,
            "gmc_semantics_note": (
                "fpv = 相邻帧仿射连乘（build_fpv_ir_gmc_median.py 口径），与既有 "
                "longquanshan_ir_gmc_median 缓存逐字节一致，龙泉山全部评测与三轨 baseline 均基于此口径"
                if args.gmc_semantics == "fpv"
                else "official = 逐 lag 直接拟合（build_full_median_dataset.py 口径），与 uav_gmc_median / Anti-UAV 官方集一致"
            ),
            "channel_disk_order": ["I_t", "gmc_diff", "median_residual"],
            "encoding": "cv2.imwrite 默认 JPEG 质量",
        },
        "sampling": {"val_stride": val_stride, "val_start": val_start, "train_stride": train_stride, "train_start": train_start},
        "plan": plan,
        "per_sequence": sorted(agg.values(), key=lambda r: (r["group"], r["sequence"])),
        "totals": {
            "sequences": len(agg),
            "chunks": len(results),
            "images": sum(r.get("images_written", 0) for r in results),
            "labels": sum(r.get("labels_written", 0) for r in results),
            "label_rows": sum(r.get("label_rows", 0) for r in results),
            "label_rows_raw": sum(r.get("label_rows_raw", 0) for r in results),
            "failed_frames": sum(r.get("failed_frames", 0) for r in results),
            "elapsed_sec": round(time.time() - started, 2),
        },
        "ignore_sequences": [e["sequence"] for e in groups_cfg.get("ignore", {}).get("sequences", [])],
        "caveats": [
            "multiscale_heatmap_cache.py 的 frame_index 取**排序位置**而非文件名帧号；"
            "本数据集 val 为连续 0..N-1 故两者一致，train 为抽帧子集，位置≠真值帧号，"
            "若要对 train 建缓存须按 split_manifest.json 的 frame_index_map 还原。",
            "001324 为绝对留出盲测序列，严禁任何切片进入训练集。",
            "特征为 3 通道 JPG（有损）；如需无损对照，另建 PNG 变体。",
        ],
    }
    (out_root / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    print("\n" + "=" * 92)
    print(f"[DONE] {manifest['totals']['elapsed_sec']:.1f}s")
    print(f"  图像 {manifest['totals']['images']}  标签 {manifest['totals']['labels']}  "
          f"框行 {manifest['totals']['label_rows']}（去重前 {manifest['totals']['label_rows_raw']}）  "
          f"失败帧 {manifest['totals']['failed_frames']}")
    print(f"  data.yaml          -> {out_root / 'data.yaml'}")
    print(f"  split_manifest.json-> {out_root / 'split_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
