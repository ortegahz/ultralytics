#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Build the 纵横AI数据采集 sequence partition from the RC2 centre-only OSD detection records.

Why this exists
---------------
The 「纵横AI数据采集」 domain has **58 raw mp4 and no ground truth**. The RC2 OSD run therefore
produces, for every frame, the detection centres and scores that a reviewer sees in the video. That
per-frame record is the only evidence the domain has, so it is also what the partition must be built
from: a sequence that the detector finds nothing in is not a "bad" sequence, it is a sequence whose
content has not been characterised yet, and it must not silently become training data.

The output mirrors ``manu/data/longquan_ft_split.json`` field for field. That is a hard requirement,
not a convenience: the domain-data-curator SOP says a new domain copies that file's structure and
**does not invent field names**, and ``manu/data/build_longquan_ft_dataset.py`` already consumes
``groups`` with ``val`` / ``train`` / ``ignore`` buckets and the ``role`` vocabulary below.

How a sequence is classified
-----------------------------
Nothing here is a quality judgement -- there is no GT to judge against yet. The classification only
records **what the detector found**, and leaves the quality call to the human review sheet:

* ``val``  -- sequences the detector fires on reliably (``core_positive_mining``): the reviewer
  checks them first, they become the golden pool of the new domain.
* ``train`` -- sequences the detector fires on intermittently (``core_positive_mining`` /
  ``negative_mining``): partial coverage, they feed the fine-tuning slices once reviewed.
* ``ignore`` -- sequences with no detection at all (``negative_mining``): nothing to review, and
  nothing to train on yet.

The split between ``val`` and ``train`` is the **intermittency** of the detection: a sequence that
fires on most of its frames is a stable positive candidate, one that fires on a scattered minority
is a weak one. This mirrors how 龙泉山 separated its golden pool from the mining set, except that
there the discriminator was F1 against GT and here it is the detection coverage itself.

Every sequence also carries the numbers the reviewer needs, so the human review sheet and the
partition never drift apart: the same detection coverage is written into both.

Usage
-----
    python manu/data/build_zongheng_ft_split.py \
        --frames-jsonl <out>/frames_shard*.jsonl \
        --out manu/data/zongheng_ft_split.json
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu.data.extract_video_frames import safe_name  # noqa: E402

# A sequence is a stable positive candidate when the detector fires on at least this share of its
# frames; below it the sequence is intermittent. Both thresholds are stated here so the partition is
# reproducible from the script alone.
STABLE_COVERAGE = 0.50
ANY_COVERAGE = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the 纵横AI sequence partition from RC2 OSD detections")
    parser.add_argument(
        "--frames-jsonl",
        action="append",
        default=[],
        help="Per-frame detection JSONL written by render_rc2_center_osd_video.py (repeatable)",
    )
    parser.add_argument(
        "--frames-dir",
        default="",
        help=(
            "Directory of per-video per-frame detection JSONL (<sequence>.jsonl). Preferred over "
            "--frames-jsonl because the renderer flushes one file per finished video, so the partition "
            "can be built from a partially finished batch."
        ),
    )
    parser.add_argument("--out", required=True, help="Where to write the split manifest")
    parser.add_argument(
        "--stable-coverage",
        type=float,
        default=STABLE_COVERAGE,
        help="Detection coverage at or above which a sequence is a stable positive (val)",
    )
    parser.add_argument(
        "--any-coverage",
        type=float,
        default=ANY_COVERAGE,
        help="Detection coverage below which a sequence is ignored",
    )
    parser.add_argument("--review-sheet", default="", help="Optional human review sheet to write alongside")
    return parser.parse_args()


def load_sequences(paths: list[str]) -> dict[str, dict]:
    """Fold the per-frame records into one summary per sequence."""
    sequences: dict[str, dict] = defaultdict(
        lambda: {"frames": 0, "det_frames": 0, "detections": 0, "scores": [], "first": None, "last": None}
    )
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                name = record["sequence"]
                bucket = sequences[name]
                bucket["frames"] += 1
                detections = record.get("detections") or []
                if detections:
                    bucket["det_frames"] += 1
                    bucket["detections"] += len(detections)
                    bucket["scores"].extend(float(item["score"]) for item in detections)
                    frame = int(record["frame"])
                    bucket["first"] = frame if bucket["first"] is None else min(bucket["first"], frame)
                    bucket["last"] = frame if bucket["last"] is None else max(bucket["last"], frame)
    for bucket in sequences.values():
        frames = max(bucket["frames"], 1)
        bucket["coverage"] = bucket["det_frames"] / frames
        bucket["det_per_frame"] = bucket["detections"] / frames
        bucket["mean_score"] = sum(bucket["scores"]) / len(bucket["scores"]) if bucket["scores"] else 0.0
        bucket.pop("scores")
    return dict(sequences)


def classify(bucket: dict, stable: float, any_coverage: float) -> tuple[str, str, str]:
    """Return (group, role, note) for one sequence."""
    coverage = bucket["coverage"]
    frames = bucket["frames"]
    detected = bucket["det_frames"]
    if coverage >= stable:
        return (
            "val",
            "golden_pool",
            f"检测覆盖 {coverage:.1%}（{detected}/{frames} 帧）且平均分 {bucket['mean_score']:.3f}；"
            f"稳定检出，作为本域黄金池候选，人工复核后定级",
        )
    if coverage >= any_coverage:
        return (
            "train",
            "core_positive_mining",
            f"检测覆盖 {coverage:.1%}（{detected}/{frames} 帧）且平均分 {bucket['mean_score']:.3f}；"
            f"间断检出，提供正切片与背景负切片养料",
        )
    return (
        "ignore",
        "",
        f"检测覆盖 {coverage:.1%}（{detected}/{frames} 帧）；模型几乎不响应，"
        f"本域尚无 GT，无法区分是空场景还是漏检，先不入训练/验证闭环",
    )


def build_manifest(
    sequences: dict[str, dict],
    args: argparse.Namespace,
    threshold: float = 0.44,
    merge_radius: float = 768.0,
) -> dict:
    groups: dict[str, list[dict]] = {"val": [], "train": [], "ignore": []}
    for name in sorted(sequences):
        bucket = sequences[name]
        group, role, note = classify(bucket, args.stable_coverage, args.any_coverage)
        if group == "ignore":
            groups["ignore"].append({"sequence": name, "reason": note})
            continue
        entry = {
            "sequence": name,
            "role": role,
            "note": note,
            # The template's per-sequence metric field. This domain has no GT yet, so it stays null
            # until the annotation pass fills it in -- same field name, no new name invented.
            "single_frame_f1": None,
            # The rendered review file for this sequence. It is not always <sequence>_osd.mp4: the
            # renderer names outputs with safe_name(), which folds non-ASCII characters to '_',
            # while the annotator's own seq_id keeps them. One sequence in this domain has a Chinese
            # name, so the reviewer is given the exact file name instead of a guess.
            "osd_file": f"{safe_name(Path(name))}_osd.mp4",
            "detection_coverage": round(bucket["coverage"], 4),
            "frames": bucket["frames"],
            "frames_with_detection": bucket["det_frames"],
            "detections": bucket["detections"],
            "mean_score": round(bucket["mean_score"], 4),
            "first_detected_frame": bucket["first"] if bucket["first"] is not None else -1,
            "last_detected_frame": bucket["last"] if bucket["last"] is not None else -1,
        }
        groups[group].append(entry)

    # The template keeps one absolute hold-out in val (role blind_hardcase, absolute_holdout=true) so the
    # blind set is never touched while the rest of the domain is being curated. The sequence to reserve
    # is a human decision -- it has to be one the annotator has never worked on -- so the slot is reserved
    # here and the choice is recorded as pending rather than invented here.
    if groups["val"]:
        reserve = groups["val"].pop()  # the last one by sort order, so the choice is explicit and stable
        reserve["role"] = "blind_hardcase"
        reserve["note"] = (
            "【待人工指定】本序列从 val 池中预留为**绝对留出盲测**（absolute_holdout=true）："
            "标注阶段不得先看它，训练/调参一律不碰；具体选哪一条由人工复核拍板后改写本条。"
            f"原始归类依据：{reserve['note']}"
        )
        reserve["absolute_holdout"] = True
        groups["val"].append(reserve)

    total_frames = sum(bucket["frames"] for bucket in sequences.values())
    total_det_frames = sum(bucket["det_frames"] for bucket in sequences.values())
    total_detections = sum(bucket["detections"] for bucket in sequences.values())

    return {
        "version": "1.0",
        "created": date.today().isoformat(),
        "author": "manu",
        "purpose": (
            "纵横AI数据采集域的序列划分（58 条 mp4，无 GT）：按 RC2 单帧多尺度检测的实际覆盖"
            "把序列分成 val / train / ignore 三桶，供人工复核、后续标注与训练使用"
        ),
        "rationale": (
            "本域**没有 GT**，因此划分**不是质量判定**，只记录「模型看到了什么」："
            "检测覆盖率高 ⇒ 稳定正样本候选（val/黄金池）；覆盖断续 ⇒ 正负切片养料（train）；"
            "几乎不响应 ⇒ 本域无法区分空场景与漏检，先隔离（ignore）。"
            "最终取舍由人工复核底稿 manu/docs/dataset/zongheng-tk.md 拍板，"
            "标注完成后回填真实指标（沿用龙泉山的 single_frame_f1 字段名，不另起字段）。"
        ),
        "sampling": {
            "val_stride": 1,
            "val_start": 0,
            "val_note": "本域当前无 GT，尚未抽帧；人工复核与标注后按步长 1 全帧进 val",
            "train_stride": 5,
            "train_start": 0,
            "train_note": "标注后每 5 帧抽 1 帧进 train（起点 0），与龙泉山同口径",
            "anti_uav_reference": {
                "dataset": "/mnt/data/siping/datasets/manu/uav",
                "train_sequences": 185,
                "train_step": 5,
                "train_first_index": 6,
                "val_sequences": 20,
                "val_step": 1,
                "val_first_index": 3,
                "data_yaml": "path: /mnt/data/siping/datasets/manu/uav | train: images/train | val: images/val | names: {0: uav}",
            },
        },
        "feature_spec": {
            "layout": "3 通道特征，磁盘通道序 [I_t, |I_t - W(I_{t-2})|, (I_t - B_t)^+]（内存现算，不落盘）",
            "model_channel_order": "推理时需反转为 [median_residual, gmc_diff, I_t]（见 manu/data/multiscale_heatmap_cache.py）",
            "window": 21,
            "stride_step": 2,
            "downscale": 2,
            "gmc_semantics": (
                "逐 lag 直接拟合（compute_affine(h, t)），与 manu/data/build_full_median_dataset.py 一致；"
                "禁用 FPV 仿射连乘变体"
            ),
            "naming": "<sequence>__frame_<真值帧号:06d>.jpg，与同名 .txt 1:1 镜像",
            "crop": {
                "source": "1920x1080 录制画幅，先按 (left=320, top=0, w=1280, h=1024) 裁回原生，再缩到 640x512",
                "native_size": [640, 512],
                "note": "与龙泉山原生画幅一致 ⇒ RC2 在龙泉山实测的工作点可原样搬过来",
            },
        },
        "groups": {
            "val": {
                "description": f"{len(groups['val'])} 条：RC2 稳定检出（覆盖 >= {args.stable_coverage:.0%}），本域黄金池候选",
                "sequences": groups["val"],
            },
            "train": {
                "description": f"{len(groups['train'])} 条：RC2 间断检出（{args.any_coverage:.0%} <= 覆盖 < {args.stable_coverage:.0%}），提供正负切片养料",
                "sequences": groups["train"],
            },
            "ignore": {
                "description": f"{len(groups['ignore'])} 条：RC2 几乎不响应（覆盖 < {args.any_coverage:.0%}），本域无 GT 无法区分空场景与漏检，先隔离",
                "sequences": groups["ignore"],
            },
        },
        "baseline_reference": {
            "note": (
                "本域无 GT，以下参照项全部来自**龙泉山业务域**的 RC2 交付工作点，"
                "用于说明「工作点从哪来」，不是本域的指标。"
            ),
            "model": {
                "name": "trial_0080 (RC2)",
                "weights": "runs/optuna_mix_ratio/trials/trial_0080/weights/epoch3.pt",
                "gate_t1": 0.9019097586568731,
                "gate_t2": 0.9989491698441769,
                "gate_t3": 0.8976937315523539,
                "use_p0_highway": False,
                "note": "RC2 门禁与龙泉山缓存均以 --ablate-p0 加载；本域 OSD 沿用同一加载配置",
            },
            "operating_point": {
                "threshold": threshold,
                "merge_radius_px": merge_radius,
                "region_threshold": 0.06,
                "scales": ["640", "160", "80"],
                "source": (
                    "manu/evaluation/probe_rc2_threshold_table.py 在真实龙泉山 RC2 缓存 "
                    "(runs/optuna_mix_ratio/trials/trial_0080/lq_eval_fpv8/ms_cache) 上实测"
                    "（merge_radius=768px）：th=0.44 ⇒ F1 94.07%/FAR 0.0176；"
                    "th=0.20 ⇒ Recall 93.91%/F1 91.98%/FAR 0.0830（虚警 4.7 倍）。"
                    "半径 128⇒F1 93.92%、64⇒93.86%，均不如 768。"
                ),
                "tracking": False,
                "note": (
                    "本域只用单帧检测（无 track），与用户要求一致。"
                    + (
                        ""
                        if abs(threshold - 0.44) < 1e-9
                        else f" ⚠️ 本档是 th={threshold} 的**敏感档**（召回优先、虚警更多），"
                        f"不是交付档（th=0.44）⇒ 交付结论仍以 th=0.44 档为准。"
                    )
                ),
            },
            "golden_pool_single_frame": {
                "gt": 0,
                "f1": None,
                "tp": None,
                "fp": None,
                "fn": None,
                "frames": None,
                "recall": None,
                "precision": None,
                "far_per_frame": None,
                "semantics": "official",
                "note": (
                    "本域黄金池的冻结单帧基线。**本域尚未标注 ⇒ gt=0、其余指标为 null**，"
                    "标注完成后回填真实数值（键名沿用龙泉山，见 domain-data-curator SKILL 步骤 2）。"
                    "注意：gt=0 时 GT 对账会**平凡通过**（数据集 0 行 == baseline 0 行）—— "
                    "这只是占位，真正的对账要等标注回填后才成立。"
                ),
            },
            "blind_001324_single_frame": {
                "gt": 0,
                "f1": None,
                "tp": None,
                "fp": None,
                "fn": None,
                "frames": None,
                "recall": None,
                "precision": None,
                "far_per_frame": None,
                "semantics": "official",
                "note": (
                    "本域独立难例盲测序列的冻结单帧基线。同样**待标注后回填**（gt=0）。"
                    "本域当前**没有**指定 blind_hardcase 序列（该 role 需人工复核后指定）。"
                ),
            },
            "longquan_reference": {
                "note": (
                    "以下是**龙泉山**业务域 RC2 的实测数值（不是本域指标），"
                    "放在这里只为说明「工作点从哪来」：同一模型在同一原生画幅上的表现。"
                ),
                "golden_pool_single_frame": {
                    "f1": 0.9884359729262149,
                    "tp": 16283,
                    "fp": 26,
                    "fn": 355,
                    "gt": 16638,
                    "frames": 17187,
                    "recall": 0.9786633008775093,
                    "precision": 0.998405788215096,
                    "far_per_frame": 0.0015127712806190725,
                    "semantics": "official",
                },
                "blind_001324_single_frame": {
                    "f1": 0.8373308251032032,
                    "tp": 8012,
                    "fp": 590,
                    "fn": 2523,
                    "gt": 10535,
                    "frames": 11810,
                    "recall": 0.7605125771238728,
                    "precision": 0.9314112996977447,
                    "far_per_frame": 0.04995766299745978,
                    "semantics": "official",
                },
            },
            "this_domain_detection_summary": {
                "sequences": len(sequences),
                "frames": total_frames,
                "frames_with_detection": total_det_frames,
                "detections": total_detections,
                "note": "本域 RC2 单帧多尺度检测的总体统计（无 GT，无 F1）",
            },
        },
    }


def write_review_sheet(path: Path, manifest: dict) -> None:
    """Write the human review sheet in the 龙泉山 format: '# pass' / '# unpass' + one line per sequence."""
    groups = manifest["groups"]
    point = manifest["baseline_reference"]["operating_point"]
    threshold = point.get("threshold")
    tier = "交付档（th=0.44，F1 最优、虚警最少）" if abs(float(threshold) - 0.44) < 1e-9 else (
        f"敏感档（th={threshold}，召回优先、虚警更多）—— **仅供人工复核/标注播种，交付结论仍以 th=0.44 档为准**"
    )
    lines = [
        "# pass",
        "# 人工复核底稿 —— 纵横AI数据采集（RC2 单帧多尺度，中心点 OSD）",
        f"# 工作点档位：{tier}",
        "# 复核对象（OSD 视频在 OSD_DIR 下）：",
        "#   /mnt/data/siping/datasets/manu/纵横AI数据采集_osd_rc2_<档位>/<下面给的 OSD 文件名>",
        "# 填法：[pass] / [ignore] <SEQ>_F1-<val>.mp4 - <原因>；不填 = 还没看",
        "# 判定依据：模型检出的中心点是否落在真实目标上（漏检 / 虚警 / 位置偏移）",
        "",
    ]
    for group in ("val", "train"):
        for entry in groups[group]["sequences"]:
            lines.append(
                f"[ ] {entry['sequence']}_F1-TBD.mp4 - 覆盖 {entry['detection_coverage']:.1%}"
                f"（{entry['frames_with_detection']}/{entry['frames']} 帧），平均分 {entry['mean_score']:.3f}"
                f" | OSD 文件：{entry['osd_file']}"
            )
    lines += ["", "# unpass", ""]
    for entry in groups["ignore"]["sequences"]:
        lines.append(f"[ignore] {entry['sequence']}_F1-TBD.mp4 - {entry['reason']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    sources = list(args.frames_jsonl)
    if args.frames_dir:
        sources.extend(str(path) for path in sorted(Path(args.frames_dir).glob("*.jsonl")))
    if not sources:
        raise SystemExit("no frames were read -- pass --frames-dir or --frames-jsonl")

    sequences = load_sequences(sources)
    if not sequences:
        raise SystemExit("no frames were read -- check --frames-dir / --frames-jsonl")

    # The operating point is read back from the render's own summary rather than hardcoded:
    # there are two working points (th=0.44 delivery / th=0.20 sensitive) and a manifest that
    # misreports its threshold would silently attribute one tier's numbers to the other.
    threshold, merge_radius = 0.44, 768.0
    for summary_path in sorted(Path(args.frames_dir).parent.glob("summary_shard*.json")) if args.frames_dir else []:
        config = json.loads(summary_path.read_text(encoding="utf-8")).get("config", {})
        threshold = float(config.get("threshold", threshold))
        merge_radius = float(config.get("merge_radius", merge_radius))
        break
    print(f"[INFO] operating point from render summary: th={threshold} r={merge_radius}px")

    manifest = build_manifest(sequences, args, threshold=threshold, merge_radius=merge_radius)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    groups = manifest["groups"]
    print(f"[OK] {out.resolve()}")
    for name in ("val", "train", "ignore"):
        print(f"  {name:6s} = {len(groups[name]['sequences'])} 条")
    total_frames = manifest["baseline_reference"]["this_domain_detection_summary"]["frames"]
    total_hits = manifest["baseline_reference"]["this_domain_detection_summary"]["frames_with_detection"]
    print(f"  frames={total_frames} frames_with_det={total_hits} ({total_hits / max(total_frames, 1):.1%})")

    if args.review_sheet:
        sheet = Path(args.review_sheet)
        write_review_sheet(sheet, manifest)
        print(f"[OK] {sheet.resolve()}")


if __name__ == "__main__":
    main()
