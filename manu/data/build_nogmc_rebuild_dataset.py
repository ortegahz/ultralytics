#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
No-GMC Rebuild Dataset Builder —— 单变量消融：W(.) ≡ IDENTITY（去掉 GMC，其余逐字复用 rebuild 代码）

Input Composition: [I_t, |I_t - W(I_{t-2})|, (I_t - B_t)^+]
  - Channel 0: I_t（原始红外帧）
  - Channel 1: |I_t - W(I_{t-2})|（2-lag 差分）
  - Channel 2: (I_t - B_t)^+（时域滑动窗口中值背景残差）

本脚本的蓝本是 **已验证的 rebuild 代码** `manu/data/build_full_median_dataset.py`
（该脚本产出的特征与冻结集逐字节一致，见 memory §九.4 空对照臂：
 `200/200 抽样文件 md5 与冻结集完全相同`、`dF1 = +0.0000`）。

与 rebuild 代码**逐字复用**的项（不构成任何额外变量）：
  - 文件名清单与标签 1:1 镜像（0.0px 标签坐标漂移）
  - 帧索引解析（`parse_seq_and_frame` / `idx_map`）
  - 序列起点钳位（`max(0, ...)`，21 帧历史在序列开头被钳到 frame 0）
  - history lags（stride_step 递推：2, 4, ..., 42）
  - `np.median` 表达式与 `astype(np.float32)` / `np.clip(..., 0, 255)` / uint8 取整
  - `np.stack` 通道序 [Ch0, Ch1, Ch2]
  - `cv2.imwrite` JPEG 编码路径（同输入数组 ⇒ 逐字节相同输出）
  - 计数严格校验（success != 期望 ⇒ RuntimeError 中止）

**唯一变量**：`W(.) = I`。
  - identity 模式：完全跳过 GMC —— 0 次仿射拟合、0 次 warp 重采样
  - gmc 模式：逐字复刻 rebuild 代码的 GMC 路径，**仅用于空对照臂验证**
    （跑它重建冻结集，md5 必须与冻结集逐字节相同 ⇒ 证明本脚本除 GMC 外
      与 rebuild 代码二进制等价）

两套数据源共用同一脚本（official 语义在两边逐行一致）：
  - Anti-UAV: --ref-dataset <uav>            --raw-root <anti-uav>
  - 龙泉山:   --ref-dataset <longquan_..._official> --raw-root <龙泉山/frames_ir_jpg>

Usage on Server:
    # Anti-UAV（train + val 一次建完，data.yaml 不会被单 split 覆盖）
    python manu/data/build_nogmc_rebuild_dataset.py \
        --ref-dataset /mnt/data/siping/datasets/manu/uav \
        --raw-root /mnt/data/siping/datasets/manu/anti-uav \
        --output /mnt/data/siping/datasets/manu/uav_gmc_median_nogmc \
        --splits train,val --mode identity --window 21 --workers 24

    # 龙泉山
    python manu/data/build_nogmc_rebuild_dataset.py \
        --ref-dataset /mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_official \
        --raw-root "/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg" \
        --output /mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_nogmc \
        --splits train,val --mode identity --window 21 --workers 24
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import re
import shutil
import sys
import time

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Build No-GMC (Identity-Warp) Temporal Median Dataset")
    parser.add_argument(
        "--ref-dataset",
        type=str,
        default="/mnt/data/siping/datasets/manu/uav",
        help="Path to reference YOLO dataset (provides filename manifest + labels, mirrored 1:1)",
    )
    parser.add_argument(
        "--raw-root",
        type=str,
        default="/mnt/data/siping/datasets/manu/anti-uav",
        help="Root directory containing raw sequences",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/mnt/data/siping/datasets/manu/uav_gmc_median_nogmc",
        help="Output directory for the no-GMC median dataset",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val",
        help="Comma-separated splits to build, e.g. 'train,val' or 'val'",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="identity",
        choices=["identity", "gmc"],
        help=(
            "identity = W(.) is IDENTITY (no GMC at all, the actual deliverable); "
            "gmc = verbatim rebuild GMC path, used ONLY for the empty-control verification"
        ),
    )
    parser.add_argument("--window", type=int, default=21, help="Temporal sliding window size (default: 21)")
    parser.add_argument("--stride-step", type=int, default=2, help="Temporal sampling stride step (default: 2)")
    parser.add_argument("--workers", type=int, default=16, help="Parallel worker processes")
    parser.add_argument("--downscale", type=int, default=2, help="Downscale factor for GMC estimation (gmc mode only)")
    parser.add_argument(
        "--verify-ch0-sample",
        type=int,
        default=0,
        help="After build, sample N output files and report Ch0 array-level MAE vs the reference dataset",
    )
    return parser.parse_args()


def natural_key(path: Path | str):
    stem = Path(path).stem
    parts = re.split(r"(\d+)", stem)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def parse_seq_and_frame(im_name: str) -> tuple[str, int]:
    stem = Path(im_name).stem
    if "___" in stem:
        parts = stem.split("___")
        seq = parts[0]
        match = re.search(r"(\d+)$", parts[1])
        return seq, int(match.group(1)) if match else 0
    if "__" in stem:
        parts = stem.split("__")
        seq = parts[0]
        match = re.search(r"(\d+)$", parts[1])
        return seq, int(match.group(1)) if match else 0

    match = re.search(r"^(.*?)(?:[_-]+)?(\d+)$", stem)
    if match:
        return match.group(1).rstrip("_-"), int(match.group(2))
    raise ValueError(f"Cannot parse sequence and frame from {im_name}")


def find_sequence_folder(raw_root: Path, seq_name: str, cache: dict[str, Path]) -> Path | None:
    if seq_name in cache:
        return cache[seq_name]
    cand = raw_root / seq_name
    if cand.is_dir():
        cache[seq_name] = cand
        return cand
    for sub in [
        raw_root / "Data" / "val" / seq_name,
        raw_root / "val" / seq_name,
        raw_root / "Data" / "train" / seq_name,
        raw_root / "train" / seq_name,
    ]:
        if sub.is_dir():
            cache[seq_name] = sub
            return sub
    for p in raw_root.rglob(seq_name):
        if p.is_dir():
            cache[seq_name] = p
            return p
    return None


class FastGMCEstimator:
    """仅 --mode gmc（空对照臂验证）使用；identity 模式完全不实例化、不调用。"""

    def __init__(self, downscale: int = 2):
        self.downscale = downscale
        self.feature_params = {
            "maxCorners": 600,
            "qualityLevel": 0.01,
            "minDistance": 4,
            "blockSize": 3,
        }

    def compute_affine(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
        h, w = curr_gray.shape[:2]
        ds = self.downscale
        H = np.eye(2, 3, dtype=np.float32)

        if ds > 1:
            prev_small = cv2.resize(prev_gray, (w // ds, h // ds))
            curr_small = cv2.resize(curr_gray, (w // ds, h // ds))
        else:
            prev_small, curr_small = prev_gray, curr_gray

        pts_prev = cv2.goodFeaturesToTrack(prev_small, mask=None, **self.feature_params)
        if pts_prev is not None and len(pts_prev) >= 6:
            pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_small, curr_small, pts_prev, None, winSize=(15, 15), maxLevel=2
            )
            good = status.ravel() == 1
            p0 = pts_prev[good]
            p1 = pts_curr[good]
            if len(p0) >= 6:
                M, _ = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                if M is not None:
                    H = M.astype(np.float32)
                    if ds > 1:
                        H[0, 2] *= ds
                        H[1, 2] *= ds
        return H

    def warp(self, img: np.ndarray, H: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        return cv2.warpAffine(img, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def process_sequence_chunk(
    seq_name: str,
    img_names: list[str],
    ref_lbl_dir_str: str,
    raw_root_str: str,
    out_img_dir_str: str,
    out_lbl_dir_str: str,
    window: int,
    stride_step: int,
    downscale: int,
    mode: str,
) -> dict:
    raw_root = Path(raw_root_str)
    out_img_p = Path(out_img_dir_str)
    out_lbl_p = Path(out_lbl_dir_str)
    ref_lbl_p = Path(ref_lbl_dir_str)

    cache: dict[str, Path] = {}
    seq_dir = find_sequence_folder(raw_root, seq_name, cache)
    if seq_dir is None:
        return {"seq": seq_name, "success": 0, "fail": len(img_names), "status": "missing_seq"}

    frames = [f for f in seq_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES]
    frames.sort(key=natural_key)
    if not frames:
        return {"seq": seq_name, "success": 0, "fail": len(img_names), "status": "no_raw_frames"}

    idx_map = {}
    for list_i, f in enumerate(frames):
        match = re.search(r"(\d+)$", f.stem)
        idx_map[int(match.group(1)) if match else list_i] = list_i

    estimator = FastGMCEstimator(downscale=downscale) if mode == "gmc" else None
    success_cnt = 0
    fail_cnt = 0

    sorted_im_names = sorted(img_names, key=natural_key)
    frame_cache: dict[int, np.ndarray] = {}

    def get_frame(list_idx: int) -> np.ndarray:
        if list_idx not in frame_cache:
            frame_cache[list_idx] = cv2.imread(str(frames[list_idx]), cv2.IMREAD_GRAYSCALE)
        return frame_cache[list_idx]

    for im_name in sorted_im_names:
        # 1. 严格检查：源标签文件存在性检查（确保 1:1 镜像）
        src_lbl_file = ref_lbl_p / f"{Path(im_name).stem}.txt"
        dst_lbl_file = out_lbl_p / f"{Path(im_name).stem}.txt"

        _, frame_idx = parse_seq_and_frame(im_name)
        curr_list_idx = idx_map.get(frame_idx, min(frame_idx, len(frames) - 1))

        # 2. 读取当前帧
        im_curr = get_frame(curr_list_idx)
        if im_curr is None:
            fail_cnt += 1
            continue

        # 3. 2-lag 差分: |I_t - W(I_{t-2})|
        #    identity 模式：W(.) ≡ I ⇒ 直接用原帧，跳过 1 次拟合 + 1 次重采样
        idx_prev2 = max(0, curr_list_idx - 2)
        im_prev2 = get_frame(idx_prev2)
        if im_prev2 is None:
            raise RuntimeError(f"[FATAL] unreadable frame {frames[idx_prev2]} (seq={seq_name}, img={im_name})")
        if mode == "gmc":
            H2 = estimator.compute_affine(im_prev2, im_curr)
            diff2 = cv2.absdiff(im_curr, estimator.warp(im_prev2, H2))
        else:
            diff2 = cv2.absdiff(im_curr, im_prev2)

        # 4. 时域滑动窗口中值背景残差: (I_t - B_t)^+
        history = []
        for step in range(1, window + 1):
            h_idx = max(0, curr_list_idx - step * stride_step)
            im_h = get_frame(h_idx)
            if im_h is None:
                continue
            if mode == "gmc":
                H_h = estimator.compute_affine(im_h, im_curr)
                history.append(estimator.warp(im_h, H_h))
            else:
                history.append(im_h)

        if len(history) >= 3:
            stack = np.stack(history, axis=0)
            median_bg = np.median(stack, axis=0).astype(np.float32)
            res_median = np.clip(im_curr.astype(np.float32) - median_bg, 0, 255).astype(np.uint8)
        else:
            res_median = diff2

        # 5. 组合 3 通道: [I_t, diff2, res_median]
        merged = np.stack([im_curr, diff2, res_median], axis=-1)

        # 6. 保存图像与复制标签（文件名与内容 100% 相同）
        dst_img_file = out_img_p / im_name
        cv2.imwrite(str(dst_img_file), merged)

        if src_lbl_file.exists():
            shutil.copy(src_lbl_file, dst_lbl_file)
        else:
            dst_lbl_file.write_text("", encoding="utf-8")

        success_cnt += 1

        # 内存释放
        evict_idx = curr_list_idx - (window * stride_step + 10)
        if evict_idx in frame_cache:
            del frame_cache[evict_idx]

    return {"seq": seq_name, "success": success_cnt, "fail": fail_cnt, "status": "ok"}


def process_split(
    ref_dir: Path,
    split_name: str,
    raw_root: Path,
    out_dir: Path,
    window: int,
    stride_step: int,
    workers: int,
    downscale: int,
    mode: str,
):
    print(f"\n==================== Processing Split [{split_name}] (mode={mode}) ====================")
    ref_img_dir = ref_dir / "images" / split_name
    ref_lbl_dir = ref_dir / "labels" / split_name

    if not ref_img_dir.is_dir():
        print(f"[WARN] Split directory not found: {ref_img_dir}, skipping.")
        return

    out_img_dir = out_dir / "images" / split_name
    out_lbl_dir = out_dir / "labels" / split_name
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    ref_images = [p.name for p in ref_img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    ref_images.sort(key=natural_key)
    total_imgs = len(ref_images)
    print(f"[INFO] Found {total_imgs} images in reference split [{split_name}].")

    # Group by sequence
    seq_groups: dict[str, list[str]] = {}
    for im_name in ref_images:
        seq, _ = parse_seq_and_frame(im_name)
        seq_groups.setdefault(seq, []).append(im_name)

    print(f"[INFO] Discovered {len(seq_groups)} distinct video sequences in [{split_name}].")
    print(f"[INFO] Launching parallel pool with {workers} worker processes...")

    t0 = time.time()
    total_success = 0
    total_fail = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = []
        for seq_name, img_names in seq_groups.items():
            f = executor.submit(
                process_sequence_chunk,
                seq_name,
                img_names,
                str(ref_lbl_dir),
                str(raw_root),
                str(out_img_dir),
                str(out_lbl_dir),
                window,
                stride_step,
                downscale,
                mode,
            )
            futures.append(f)

        for f in futures:
            res = f.result()
            total_success += res["success"]
            total_fail += res["fail"]
            status_tag = f"Fail={res['fail']}" if res["status"] != "ok" else "OK"
            print(f"  --> Completed [{res['seq']:<28}] : {res['success']:>5} images ({status_tag})")

    elapsed = time.time() - t0
    fps = total_imgs / max(0.1, elapsed)
    print(f"\n[{split_name}] Completed in {elapsed:.1f}s ({fps:.1f} imgs/s) | Success: {total_success}, Fail: {total_fail}")

    # 严格校验：生成的图像数量与标签数量必须与参考数据集完全一致！
    if total_success != total_imgs or total_fail > 0:
        raise RuntimeError(
            f"[FATAL] Split [{split_name}] count mismatch! Expected {total_imgs}, successfully generated {total_success}, failed {total_fail}. Cannot guarantee fair comparison!"
        )
    print(f"[VERIFY PASSED] Split [{split_name}] exact count {total_success}/{total_imgs} verified with 100% parity!\n")


def write_data_yaml(out_dir: Path):
    yaml_content = f"""# UAV No-GMC Temporal Median Dataset [I_t, |I_t - I_{{t-2}}|, (I_t - B_t)^+]
# W(.) = IDENTITY（无 GMC：0 次拟合 / 0 次 warp）。其余语义与冻结 GMC 集逐字一致。
path: {out_dir.resolve()}
train: images/train
val: images/val

names:
  0: uav
"""
    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"[INFO] Successfully created data configuration: {yaml_path.resolve()}")


def verify_ch0_sample(ref_dir: Path, out_dir: Path, splits: list[str], n_sample: int) -> None:
    """Ch0 数组层自检：no-GMC 集的 Ch0 与参照集 Ch0 应一致（仅差 JPEG 4:2:0 色度底噪）。

    判据必须下到**数组层**（memory §七.2 教训）：3 通道 JPEG 的 4:2:0 色度下采样会让
    Ch1/Ch2 一变就带动 Ch0 解码值抖 1~8 灰阶 ⇒ 「Ch0 逐字节为 0」是错误判据。
    """
    print("\n==================== Ch0 array-level self-check ====================")
    for split in splits:
        ref_img_dir = ref_dir / "images" / split
        out_img_dir = out_dir / "images" / split
        if not ref_img_dir.is_dir() or not out_img_dir.is_dir():
            continue
        names = sorted(
            (p.name for p in ref_img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
            key=natural_key,
        )
        if not names:
            continue
        step = max(1, len(names) // max(1, n_sample))
        picked = names[::step][:n_sample]
        maes = []
        for name in picked:
            a = cv2.imread(str(ref_img_dir / name), cv2.IMREAD_COLOR)
            b = cv2.imread(str(out_img_dir / name), cv2.IMREAD_COLOR)
            if a is None or b is None:
                continue
            maes.append(float(np.abs(a[:, :, 0].astype(np.float32) - b[:, :, 0].astype(np.float32)).mean()))
        if maes:
            print(
                f"[VERIFY] split={split} sampled={len(maes)} | Ch0 MAE vs reference = "
                f"{np.mean(maes):.5f} 灰阶 (max {np.max(maes):.5f})"
            )


def main():
    args = parse_args()
    ref_dir = Path(args.ref_dataset).resolve()
    raw_root = Path(args.raw_root).resolve()
    out_dir = Path(args.output).resolve()

    if not ref_dir.is_dir():
        for cand in [Path("/mnt/data/siping/datasets/manu/uav"), Path("/home/manu/mnt/datasets/manu/uav")]:
            if cand.is_dir():
                ref_dir = cand
                break
    if not raw_root.is_dir():
        for cand in [Path("/mnt/data/siping/datasets/manu/anti-uav"), Path("/home/manu/mnt/datasets/manu/anti-uav")]:
            if cand.is_dir():
                raw_root = cand
                break

    print("=" * 90)
    print("   UAV Tiny Object Detection: No-GMC Rebuild Dataset Builder (W(.) = IDENTITY)")
    print(f"   Reference Dataset : {ref_dir}")
    print(f"   Raw Video Root    : {raw_root}")
    print(f"   Target Output     : {out_dir}")
    print(f"   Mode              : {args.mode}   ({'0 fit / 0 warp' if args.mode == 'identity' else 'verbatim rebuild GMC path'})")
    print(f"   Median Window     : {args.window} frames (step: {args.stride_step}) | Workers: {args.workers}")
    print("=" * 90)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for s in splits:
        process_split(
            ref_dir=ref_dir,
            split_name=s,
            raw_root=raw_root,
            out_dir=out_dir,
            window=args.window,
            stride_step=args.stride_step,
            workers=args.workers,
            downscale=args.downscale,
            mode=args.mode,
        )

    write_data_yaml(out_dir)

    if args.verify_ch0_sample > 0:
        verify_ch0_sample(ref_dir, out_dir, splits, args.verify_ch0_sample)

    print(f"\n[SUCCESS] Entire dataset successfully created at: {out_dir}\n")


if __name__ == "__main__":
    main()
