#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Anti-UAV **红外（IR）单目标**加载器 —— v2 Target-Centric 流水线的数据底座。

任务书约束（不可违反）
----------------------
1. **只处理红外模态**：可见光（``visible.mp4`` / ``visible.json`` / ``RGB_*``）一律**跳过**，
   连"看一眼都不行"——本模块不会打开任何非 IR 文件；
2. **原始数据只读**：绝不 rm / 改写数据集里的任何文件；剔除多目标序列与无效帧
   全部在「加载 / 清单生成」层完成；
3. **统一轨迹协议**：输出标准记录 ``(seq_id, frame_id, track_id, cx, cy, w, h, exist)``，
   原始左上角标注 ``[x, y, w, h]`` 在这里转换成中心坐标。

本机 Anti-UAV 副本的实际形态（实测，非假设）
--------------------------------------------
* 根目录 ``/mnt/data/siping/datasets/manu/anti-uav/train/<seq_id>/``，**223 条**红外序列；
* 每条序列 = 红外帧（``000001.jpg`` / ``000001.png``，640×512 或 512×512，8bit 灰度）
  + 一个 ``IR_label.json``；
* ``IR_label.json`` 结构::

      {"exist": [1,1,...], "gt_rect": [[x,y,w,h], ...], "LR": [...], "VE": [...], ...}

  其中 ``gt_rect[i]`` 是**该帧的扁平 ``[x, y, w, h]``**（不是框的列表）；
* 本副本**已经是单目标版本**：223 条序列里 **0 条**出现"单帧多框"，
  因此「多目标剔除名单」为空——这是事实，不是我漏检（见 ``dataset_manifest.json``）。

执行机：服务器 ``huangzhe@192.168.99.40:32222``（conda env ``uav``，需要 cv2 解帧）。
复现：

    python3 iqa_v2/dataset_anti_uav.py \
        --dataset-root /mnt/data/siping/datasets/manu/anti-uav \
        --output-dir iqa_v2/output --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- 常量
WINDOW_LEN = 5                 # 有效 5 帧短时程窗口
SMALL_TARGET_MAX_DIM = 20      # 小目标分层阈值：max(w, h) <= 20 px
SEED = 42                      # 固定随机种子（确定性采样）
MAX_WINDOWS_PER_SEQ_CAP = 30   # 每条序列的窗口数上限（任务书硬上限）
TOTAL_WINDOW_TARGET = 2600     # 全量窗口数目标（任务书口径 2,000~3,500）

#: 红外标注候选文件名（**只有**这些会被打开）
IR_LABEL_NAMES = ("IR_label.json", "infrared.json", "IR.json", "ir_label.json")
#: 可见光 / 配对模态的判别词——命中即**跳过**，绝不打开
VISIBLE_HINTS = ("visible", "rgb", "color", "optic")
FRAME_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# --------------------------------------------------------------------------- 数据结构
@dataclass
class FrameRecord:
    """统一轨迹协议的单条记录。"""

    seq_id: str
    frame_id: int
    track_id: int
    cx: float
    cy: float
    w: float
    h: float
    exist: int


@dataclass
class Sequence:
    """一条红外序列（已通过单目标过滤或已记录剔除原因）。"""

    seq_id: str
    root: Path
    label_path: Path | None = None
    frame_files: list[Path] = field(default_factory=list)
    records: list[FrameRecord] = field(default_factory=list)
    width: int = 0
    height: int = 0
    max_targets_per_frame: int = 1
    n_track_ids: int = 1
    drop_reason: str = ""

    @property
    def kept(self) -> bool:
        return not self.drop_reason

    def summary(self) -> dict:
        return {
            "seq_id": self.seq_id,
            "n_frames": len(self.frame_files),
            "n_annotated_frames": len(self.records),
            "width": self.width,
            "height": self.height,
            "max_targets_per_frame": self.max_targets_per_frame,
            "n_track_ids": self.n_track_ids,
            "n_exist0": sum(1 for r in self.records if r.exist == 0),
            "drop_reason": self.drop_reason,
        }


@dataclass
class Window:
    """一个有效的 5 帧评估窗口。"""

    seq_id: str
    frame_id: int          # 窗口首帧 t
    length: int            # 恒为 5
    gt: list[FrameRecord]  # t..t+4 的真值记录
    small_target: bool     # max(w, h) <= 20 px（按首帧 t 的目标尺寸）

    @property
    def w(self) -> float:
        return self.gt[0].w

    @property
    def h(self) -> float:
        return self.gt[0].h


# --------------------------------------------------------------------------- 标注解析
def _norm_bbox(row) -> list[float] | None:
    """把一行的框解析成 ``[x, y, w, h]``；不合法返回 ``None``。"""
    if row is None:
        return None
    if isinstance(row, dict):
        row = row.get("bbox") or row.get("box") or row.get("rect")
        if row is None:
            return None
    if not isinstance(row, (list, tuple)) or len(row) < 4:
        return None
    try:
        x, y, w, h = (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x, y, w, h)):
        return None
    return [x, y, w, h]


def _rows_from_label(label: dict) -> tuple[list[list[list[float]]], list[int], int]:
    """从标注 dict 抽出「每帧的框列表」与 ``exist`` 序列。

    支持三种官方/本地形态：

    * ``gt_rect: [[x,y,w,h], ...]``            → 每帧一个扁平框（本地副本形态）
    * ``gt_rect: [[[x,y,w,h], ...], ...]``      → 每帧一个框列表（多目标形态）
    * ``gt: [{"bbox": [x,y,w,h]}, ...]``        → 官方 ``infrared.json`` 形态

    返回 ``(rows, exist, n_track_ids)``；``rows[i]`` 是第 i 帧的**框列表**。
    """
    exist = [int(v) for v in label.get("exist", [])] if "exist" in label else []
    track_ids: set[int] = set()

    raw = None
    for key in ("gt_rect", "gt", "targets"):
        if key in label:
            raw = label[key]
            break
    if raw is None:
        return [], exist, 0

    rows: list[list[list[float]]] = []
    for index, item in enumerate(raw):
        boxes: list[list[float]] = []
        if isinstance(item, dict):
            box = _norm_bbox(item)
            if box is not None:
                boxes.append(box)
            tid = item.get("track_id", item.get("id"))
            if tid is not None:
                track_ids.add(int(tid))
        elif isinstance(item, (list, tuple)):
            if len(item) >= 4 and all(isinstance(v, (int, float)) for v in item[:4]):
                box = _norm_bbox(item)
                if box is not None:
                    boxes.append(box)
            else:
                for sub in item:
                    box = _norm_bbox(sub)
                    if box is not None:
                        boxes.append(box)
        rows.append(boxes)
        if not exist or index >= len(exist):
            pass
    if not exist:
        exist = [1 if boxes else 0 for boxes in rows]
    return rows, exist, len(track_ids)


def _is_visible_name(name: str) -> bool:
    low = name.lower()
    return any(hint in low for hint in VISIBLE_HINTS)


def discover_sequence_dirs(root: Path) -> list[Path]:
    """找出所有**含红外标注**的序列目录（可见光文件一律不看内容）。"""
    seq_dirs: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_dir():
            continue
        for name in IR_LABEL_NAMES:
            if _is_visible_name(name):
                continue
            if (path / name).is_file():
                seq_dirs.append(path)
                break
    # 去重（一个目录可能命中多个候选名）
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in seq_dirs:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def list_ir_frames(seq_dir: Path) -> list[Path]:
    """列出该序列下的**红外帧**文件（跳过任何可见光命名的文件）。"""
    frames = []
    for path in sorted(seq_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in FRAME_SUFFIXES:
            continue
        if _is_visible_name(path.stem):
            continue
        frames.append(path)
    return frames


def load_sequence(seq_dir: Path) -> Sequence:
    """加载一条红外序列；多目标 / 不可用的序列**只打标记**，不改任何原始文件。"""
    seq_id = seq_dir.name
    label_path = None
    for name in IR_LABEL_NAMES:
        candidate = seq_dir / name
        if candidate.is_file() and not _is_visible_name(name):
            label_path = candidate
            break
    seq = Sequence(seq_id=seq_id, root=seq_dir, label_path=label_path)
    if label_path is None:
        seq.drop_reason = "missing_ir_label"
        return seq

    label = json.loads(label_path.read_text(encoding="utf-8"))
    rows, exist, n_track_ids = _rows_from_label(label)
    seq.n_track_ids = max(n_track_ids, 1)
    seq.max_targets_per_frame = max((len(b) for b in rows), default=0)

    frames = list_ir_frames(seq_dir)
    seq.frame_files = frames
    if not frames:
        seq.drop_reason = "no_ir_frames"
        return seq

    # ---- 多目标剔除（任务书硬要求）：单帧出现多个框，或标注含多个 track_id
    if seq.max_targets_per_frame > 1:
        seq.drop_reason = "multi_target_in_frame"
        return seq
    if n_track_ids > 1:
        seq.drop_reason = "multiple_track_ids"
        return seq
    # ---- 帧数与标注数不一致 → 无法建立帧-标注一一对应，剔除
    if rows and len(rows) != len(frames):
        seq.drop_reason = "frame_annotation_count_mismatch(%d!=%d)" % (len(frames), len(rows))
        return seq

    for index in range(len(rows)):
        boxes = rows[index]
        exist_flag = int(exist[index]) if index < len(exist) else int(bool(boxes))
        if not boxes:
            seq.records.append(FrameRecord(seq_id, index, 0, float("nan"), float("nan"),
                                           0.0, 0.0, 0))
            continue
        x, y, w, h = boxes[0]
        seq.records.append(FrameRecord(seq_id, index, 0, x + w / 2.0, y + h / 2.0,
                                       float(w), float(h), exist_flag))
    return seq


def read_frame_size(path: Path) -> tuple[int, int]:
    """读一帧的 (width, height)（灰度）。用 cv2；不可用时回退到 PIL。"""
    try:
        import cv2

        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            return int(image.shape[1]), int(image.shape[0])
    except Exception:  # pragma: no cover - 仅在缺 cv2 时兜底
        pass
    from PIL import Image

    with Image.open(path) as im:
        return int(im.size[0]), int(im.size[1])


# --------------------------------------------------------------------------- 窗口协议
def valid_windows(seq: Sequence) -> list[Window]:
    """切出有效 5 帧窗口：5 帧全部 ``exist==1``、``w>0``、``h>0``、中心在画面内。"""
    records = seq.records
    out: list[Window] = []
    for start in range(0, len(records) - WINDOW_LEN + 1):
        chunk = records[start:start + WINDOW_LEN]
        ok = True
        for rec in chunk:
            if rec.exist != 1 or rec.w <= 0 or rec.h <= 0:
                ok = False
                break
            if not (math.isfinite(rec.cx) and math.isfinite(rec.cy)):
                ok = False
                break
            if not (0.0 <= rec.cx < seq.width and 0.0 <= rec.cy < seq.height):
                ok = False
                break
        if not ok:
            continue
        first = chunk[0]
        out.append(Window(seq_id=seq.seq_id, frame_id=start, length=WINDOW_LEN, gt=list(chunk),
                          small_target=max(first.w, first.h) <= SMALL_TARGET_MAX_DIM))
    return out


def sample_windows(all_windows: dict[str, list[Window]], seed: int = SEED,
                   max_per_seq: int = MAX_WINDOWS_PER_SEQ_CAP,
                   total_target: int = TOTAL_WINDOW_TARGET) -> tuple[dict[str, list[Window]], dict]:
    """确定性滑窗采样：每条序列按**固定步长**取最多 ``max_per_seq`` 个窗口。

    * 序列内：候选窗口按帧号排序后，用**等距步长**取点（``numpy.linspace``），
      ⇒ 同一条序列的窗口在时间轴上均匀铺开，不偏前不偏后；
    * 序列间：``quota = clip(total_target // n_seq, 1, max_per_seq)``，
      当序列数多到 ``quota * n_seq`` 会超出总预算时自动收紧；
    * 随机种子只影响「序列顺序打散」（``seed`` 固定 ⇒ 结果可复现）。
    """
    seq_ids = sorted(all_windows.keys())
    n_seq = len(seq_ids)
    quota = max(1, min(max_per_seq, total_target // max(n_seq, 1)))

    rng = random.Random(seed)
    order = list(seq_ids)
    rng.shuffle(order)          # 固定种子的打散（结果可复现）

    selected: dict[str, list[Window]] = {}
    for seq_id in order:
        candidates = all_windows[seq_id]
        if not candidates:
            continue
        if len(candidates) <= quota:
            picked = list(candidates)
        else:
            indices = np.linspace(0, len(candidates) - 1, quota).round().astype(int)
            picked = [candidates[i] for i in sorted(set(indices.tolist()))]
        selected[seq_id] = picked

    stats = {
        "seed": seed,
        "max_windows_per_seq_cap": max_per_seq,
        "quota_per_seq": quota,
        "total_windows": sum(len(v) for v in selected.values()),
        "sequences_with_windows": len(selected),
    }
    return selected, stats


# --------------------------------------------------------------------------- 清单输出
def build_manifest(dataset_root: Path, sequences: list[Sequence],
                    selected: dict[str, list[Window]], sampling: dict) -> dict:
    kept = [s for s in sequences if s.kept]
    dropped = [s for s in sequences if not s.kept]
    windows = [w for seq_windows in selected.values() for w in seq_windows]
    small = [w for w in windows if w.small_target]
    multi = [s for s in dropped if s.drop_reason.startswith("multi_target")
             or s.drop_reason.startswith("multiple_track_ids")]
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_root": str(dataset_root),
        "modality": "IR only (visible paired files are skipped unread)",
        "ir_label_names_scanned": list(IR_LABEL_NAMES),
        "visible_name_hints_skipped": list(VISIBLE_HINTS),
        "total_ir_sequences_found": len(sequences),
        "kept_single_target_sequences": len(kept),
        "dropped_sequences_count": len(dropped),
        "multi_target_dropped_count": len(multi),
        "dropped_sequences": [{"seq_id": s.seq_id, "reason": s.drop_reason} for s in dropped],
        "total_annotated_frames": sum(len(s.records) for s in sequences),
        "window_protocol": {
            "length": WINDOW_LEN,
            "requirements": ["exist==1", "w>0", "h>0", "center inside frame",
                             "5 consecutive frames"],
            "small_target_rule": "max(w,h) <= %d px (measured on the first frame t)" % SMALL_TARGET_MAX_DIM,
        },
        "sampling": sampling,
        "window_counts": {
            "total": len(windows),
            "small_target": len(small),
            "small_target_ratio": round(len(small) / len(windows), 6) if windows else 0.0,
            "large_target": len(windows) - len(small),
        },
        "sequences": [s.summary() for s in sequences],
    }


def write_windows_csv(path: Path, selected: dict[str, list[Window]]) -> None:
    """窗口清单 CSV（评估与测试都读它）。"""
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["seq_id", "frame_id", "length", "small_target", "w", "h"] + [
        "gt_%d_cx" % i for i in range(WINDOW_LEN)] + [
        "gt_%d_cy" % i for i in range(WINDOW_LEN)] + [
        "gt_%d_exist" % i for i in range(WINDOW_LEN)]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for seq_id in sorted(selected):
            for window in selected[seq_id]:
                row = {
                    "seq_id": seq_id,
                    "frame_id": window.frame_id,
                    "length": window.length,
                    "small_target": int(window.small_target),
                    "w": window.w,
                    "h": window.h,
                }
                for index, rec in enumerate(window.gt):
                    row["gt_%d_cx" % index] = rec.cx
                    row["gt_%d_cy" % index] = rec.cy
                    row["gt_%d_exist" % index] = rec.exist
                writer.writerow(row)


# --------------------------------------------------------------------------- 主流程
def main() -> None:
    parser = argparse.ArgumentParser(description="Anti-UAV 红外单目标加载器（v2）")
    parser.add_argument("--dataset-root",
                        default="/mnt/data/siping/datasets/manu/anti-uav",
                        help="Anti-UAV 根目录（只读）")
    parser.add_argument("--output-dir", default="iqa_v2/output", help="产物目录")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-windows-per-seq", type=int, default=MAX_WINDOWS_PER_SEQ_CAP)
    parser.add_argument("--total-target", type=int, default=TOTAL_WINDOW_TARGET)
    parser.add_argument("--smoke", action="store_true",
                        help="冒烟：3 条序列、每条 20 个窗口")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    seq_dirs = discover_sequence_dirs(dataset_root)
    print("[DATASET] 红外序列目录：%d" % len(seq_dirs))
    sequences: list[Sequence] = []
    for seq_dir in seq_dirs:
        seq = load_sequence(seq_dir)
        if seq.kept:
            seq.width, seq.height = read_frame_size(seq.frame_files[0])
        sequences.append(seq)
    kept = [s for s in sequences if s.kept]
    dropped = [s for s in sequences if not s.kept]
    print("[DATASET] 保留单目标序列 %d / 剔除 %d（多目标 %d）"
          % (len(kept), len(dropped),
             sum(1 for s in dropped if s.drop_reason.startswith(("multi_target", "multiple_track_ids")))))
    for seq in dropped[:10]:
        print("   [DROP] %s：%s" % (seq.seq_id, seq.drop_reason))

    all_windows: dict[str, list[Window]] = {}
    for seq in kept:
        windows = valid_windows(seq)
        if windows:
            all_windows[seq.seq_id] = windows
    print("[WINDOW] 有有效窗口的序列：%d，总候选窗口 %d"
          % (len(all_windows), sum(len(v) for v in all_windows.values())))

    if args.smoke:
        picked = {}
        for seq_id in sorted(all_windows)[:3]:
            picked[seq_id] = all_windows[seq_id][:20]
        sampling = {"seed": args.seed, "max_windows_per_seq_cap": 20, "quota_per_seq": 20,
                    "total_windows": sum(len(v) for v in picked.values()),
                    "sequences_with_windows": len(picked), "mode": "smoke"}
    else:
        picked, sampling = sample_windows(all_windows, seed=args.seed,
                                          max_per_seq=args.max_windows_per_seq,
                                          total_target=args.total_target)
    print("[WINDOW] 采样后窗口：%d（quota/序列 %d）"
          % (sampling["total_windows"], sampling["quota_per_seq"]))

    manifest = build_manifest(dataset_root, sequences, picked, sampling)
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_windows_csv(output_dir / "windows.csv", picked)
    print("[OUT] %s" % manifest_path)
    print("[OUT] %s" % (output_dir / "windows.csv"))


if __name__ == "__main__":
    main()
