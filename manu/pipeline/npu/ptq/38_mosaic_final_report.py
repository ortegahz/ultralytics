#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""38_mosaic_final_report.py -- assemble the auditable comparison table.

It recomputes EVERY arm from its primary artefact with ONE scorer
(``manu/evaluation/system_sota_longquan.py --no-pipeline`` plus the gate's own
per-track params), so no number in the report is hand-copied.

GROUPS (tracks)
---------------
Each group is a named pool of Longquanshan val sequences, scored as ONE
aggregate (``--seqs`` keeps the caller's order and aggregates):

  track2_golden  the 7 golden-pool val sequences (T2)
  track3_blind   the blind hard-case hold-out sequence (T3)

ARMS (computed for every group)
--------------------------------
A   PC f32, per-scale (3 forwards)   -> the frozen gate cache
A0  PC f32, per-scale, native tile only
B   PC f32, mosaic (1 forward)       -> 37_mosaic_pc_control.py, INTER_AREA thumbs
B1  PC f32, mosaic, INTER_LINEAR thumbs (== the board's RGA semantics)
B0  PC f32, mosaic, native tile only
C   board f16, mosaic (1 forward)    -> the board TSVs
C0  board f16, mosaic, native tile only

Usage
-----
    python 38_mosaic_final_report.py \
        --group track2_golden=SEQ1,SEQ2,... \
        --group track3_blind=SEQ \
        --board-tsv-dir DIR --pc-area-tsv-dir DIR --pc-linear-tsv-dir DIR \
        --out DIR
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCORER = PROJECT_ROOT / "manu/evaluation/system_sota_longquan.py"
GATE_DIR = Path(
    "/tmp/pycharm_project_10ae9e2e/runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3"
)
LABELS = Path("/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010")


# --------------------------------------------------------------------------------------
# the board TSV format (one TAB-separated column per detection)
# --------------------------------------------------------------------------------------
def parse_mosaic_tsv(path: Path) -> dict[str, list[tuple]]:
    """-> {stem: [(tag, cx, cy, w, h, peak, sum), ...]}"""
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
            dets.append((int(v[0]), float(v[1]), float(v[2]), int(v[3]), int(v[4]),
                         float(v[6]), float(v[7])))
        if len(dets) != n:
            raise ValueError(f"{path}: {stem} declares {n} detections but carries {len(dets)}")
        out[stem] = dets
    return out


def build_cache(dets: dict[str, list[tuple]], meta: dict) -> dict:
    """ms_cache-shaped dict -- dtype layout copied from multiscale_heatmap_cache.py:437-476."""
    frame_index, points, sizes, scores, sums, areas, tags = [], [], [], [], [], [], []
    offset = [0]
    for stem in sorted(dets, key=lambda s: int(s.rsplit("__", 1)[1])):
        frame_index.append(int(stem.rsplit("__", 1)[1]))
        for tag, cx, cy, bw, bh, peak, total in dets[stem]:
            points.append((cx, cy))
            sizes.append((bw, bh))
            scores.append(peak)
            sums.append(total)
            areas.append(1)          # the scorer never reads `area` either
            tags.append(tag)
        offset.append(len(points))
    return {
        "meta": {**meta, "frames": len(frame_index)},
        "frame_index": np.asarray(frame_index, dtype=np.int32),
        "offset": np.asarray(offset, dtype=np.int64),
        "points": np.asarray(points, dtype=np.float32).reshape(-1, 2),
        "sizes": np.asarray(sizes, dtype=np.float32).reshape(-1, 2),
        "scores": np.asarray(scores, dtype=np.float16),
        "sums": np.asarray(sums, dtype=np.float32),
        "area": np.asarray(areas, dtype=np.int32),
        "tag": np.asarray(tags, dtype=np.uint8),
    }


def filter_tags(cache: dict, keep: set[int]) -> dict:
    tags, off = cache["tag"], cache["offset"]
    pts, szs, sco, sums, ar, tg, new_off = [], [], [], [], [], [], [0]
    for i in range(len(cache["frame_index"])):
        a, b = int(off[i]), int(off[i + 1])
        for j in range(a, b):
            if int(tags[j]) in keep:
                pts.append(cache["points"][j])
                szs.append(cache["sizes"][j])
                sco.append(cache["scores"][j])
                sums.append(cache["sums"][j])
                ar.append(cache["area"][j])
                tg.append(int(tags[j]))
        new_off.append(len(pts))
    out = dict(cache)
    out.update(
        points=np.asarray(pts, np.float32).reshape(-1, 2),
        sizes=np.asarray(szs, np.float32).reshape(-1, 2),
        scores=np.asarray(sco, np.float16),
        sums=np.asarray(sums, np.float32),
        area=np.asarray(ar, np.int32),
        tag=np.asarray(tg, np.uint8),
        offset=np.asarray(new_off, np.int64),
    )
    return out


def per_tag(cache: dict) -> dict:
    counts: dict[int, int] = {0: 0, 1: 0, 2: 0}
    for t in cache["tag"]:
        counts[int(t)] = counts.get(int(t), 0) + 1
    return counts


def score_group(cache_root: Path, sequences: list[str], params_json: Path, tag: str) -> dict:
    """Score a cache root holding one <sequence>.pkl per sequence."""
    out_json = cache_root / f"_scores_{tag}.json"
    cmd = [
        sys.executable, str(SCORER),
        "--cache-root", str(cache_root),
        "--labels-root", str(LABELS),
        "--seqs", ",".join(sequences),
        "--no-pipeline",
        "--params-json", str(params_json),
        "--out-json", str(out_json),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"scorer failed for {tag}: {result.stdout[-1500:]}\n{result.stderr[-1500:]}")
    report = json.loads(out_json.read_text(encoding="utf-8"))
    return {"overall": report["overall"], "per_sequence": report["per_sequence"]}


def write_per_seq(cache_root: Path, seq: str, cache: dict) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / f"{seq}.pkl").write_bytes(pickle.dumps(cache, protocol=pickle.HIGHEST_PROTOCOL))


def parse_timing(path: Path) -> dict:
    """Parse the board's timing.txt.

    The e2e library writes three columns per segment
    (``segment total_ms avg_ms/frame``); the older BMP-driven binary writes
    four (``segment total_ms avg_ms/frame frames``).  Both are accepted.
    The per-frame AVERAGE is the primary value; the total is kept under
    ``<segment>_total_ms``.
    """
    out: dict[str, float] = {}
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 3 and not parts[0].startswith("#") and parts[0] != "fps=":
            try:
                out[parts[0]] = float(parts[2])
                out[parts[0] + "_total_ms"] = float(parts[1])
            except ValueError:
                continue
    for line in text.splitlines():
        if line.startswith("fps="):
            out["fps"] = float(line.split("=")[1])
        elif "=" in line and not line.startswith("#"):
            # Any key=value line the board writes: the service configuration
            # (gmc_workers / npu_workers / jitter_depth / ...) and the
            # service-health counters (dropped / deadline_blows /
            # gmc_fit_max).
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if key in ("frames", "canvas_mode", "input_mode", "seq"):
                out[key] = value if not value.isdigit() else float(value)
            else:
                try:
                    out[key] = float(value)
                except ValueError:
                    pass
    return out


# --------------------------------------------------------------------------------------
ARM_LABELS = [
    ("A", "PC f32, per-scale, 3 forwards", "frozen gate cache"),
    ("A0", "PC f32, per-scale, native only", "gate cache filtered to tag 0"),
    ("B", "PC f32, mosaic, 1 forward", "37_mosaic_pc_control.py (INTER_AREA thumbs)"),
    ("B1", "PC f32, mosaic, INTER_LINEAR thumbs", "37_mosaic_pc_control.py --thumb-interp linear"),
    ("B0", "PC f32, mosaic, native only", "37 mosaic TSV filtered to tag 0"),
    ("C", "board f16, mosaic, 1 forward", "rknn_mosaic_heatmap (RGA thumbs)"),
    ("C0", "board f16, mosaic, native only", "board TSV filtered to tag 0"),
    ("E", "**e2e**: raw IR -> GMC(step10) -> RGA mosaic -> board f16",
     "rknn_mosaic_e2e (libmosaic_pipeline)"),
    ("E0", "e2e, native only", "e2e TSV filtered to tag 0"),
]


def render_markdown(payload: dict) -> str:
    out: list[str] = []
    out.append("# RK3588 板端 f16 multi-scale 拼图推理 —— 与 PC f32 multi-scale 的对照表（no track）\n")
    out.append(f"> 生成时间 {payload['generated']} · 生成脚本 `manu/pipeline/npu/ptq/38_mosaic_final_report.py`")
    out.append("> 打分器：`manu/evaluation/system_sota_longquan.py --no-pipeline`（单帧臂，**无 track**）\n")

    for group in payload["groups"]:
        arms = group["arms"]
        m = {k: v["metrics"] for k, v in arms.items()}
        base = m["A"]
        out.append(f"## 轨道 `{group['name']}` —— {group['description']}\n")
        out.append(f"> 序列 {len(group['sequences'])} 条 / {group['frames']} 帧 / GT {base['gt']} · "
                   f"口径参数来自 `{group['params_source']}`（th_base={group['params']['th_base']}, "
                   f"dedup_radius={group['params']['dedup_radius']}）\n")
        out.append("### 表 A：精度对照\n")
        out.append("| arm | 计算图 | TP | FP | FN | GT | F1 | Recall | Precision | det | ΔF1 vs A |")
        out.append("|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
        for key, title, _src in ARM_LABELS:
            if key not in m:
                continue
            mm = m[key]
            out.append(
                f"| **{key}** | {title} | {mm['tp']} | {mm['fp']} | {mm['fn']} | {mm['gt']} | "
                f"{mm['f1']:.6f} | {mm['recall']:.6f} | {mm['precision']:.6f} | {mm['det']} | "
                f"{mm['f1'] - base['f1']:+.6f} |"
            )
        out.append("")
        out.append("### 表 B：损失分解（每一段都有中间量支撑）\n")
        out.append("| 段 | 口径 | ΔF1 | ΔTP | ΔFP | 含义 |")
        out.append("|:--|:--|--:|--:|--:|:--|")

        def seg(a: str, b: str, meaning: str) -> None:
            ma, mb = m[a], m[b]
            out.append(
                f"| {a} → {b} | {'逐尺度' if a.startswith('A') else '拼图'} → "
                f"{'逐尺度' if b.startswith('A') else '拼图'} | {mb['f1'] - ma['f1']:+.6f} | "
                f"{mb['tp'] - ma['tp']:+d} | {mb['fp'] - ma['fp']:+d} | {meaning} |"
            )

        seg("A", "B", "**拼图打包损失**（f32，单次前向 vs 三次前向）")
        seg("B", "B1", "缩略图重采样语义影响（INTER_AREA → INTER_LINEAR≈RGA）")
        seg("B", "C", "**f16 + 板端量化损失**（拼图对拼图）")
        seg("A0", "A", "多尺度增益（逐尺度口径）")
        seg("B0", "B", "多尺度增益（拼图口径）")
        seg("A0", "B0", "拼图打包损失（仅 native tile）")
        seg("B0", "C0", "f16 + 板端量化损失（仅 native tile）")
        if "E" in m:
            seg("C", "E", "**GMC step10 语义代价**（同为板端 f16 拼图，只换特征语义）")
            seg("B", "E", "**总代价**（f32 逐尺度 → e2e：打包 + f16 + 板端 + step10 语义）")
            seg("E0", "E", "多尺度增益（e2e 口径）")
            seg("B0", "E0", "拼图打包损失（仅 native tile，e2e 口径）")
        out.append("")
        out.append("### 表 D：各 arm 的按 tag 检测数\n")
        out.append("| arm | tag0 native | tag1 thumb160 | tag2 thumb80 | 三尺度合计 | tag255 融合行（不参与评测） |")
        out.append("|:--|--:|--:|--:|--:|--:|")
        for key, title, _src in ARM_LABELS:
            if key not in arms:
                continue
            tags = arms[key]["tags"]
            g = lambda k: tags.get(k, tags.get(str(k), 0))          # int key OR str key
            out.append(f"| {key} | {g(0)} | {g(1)} | {g(2)} | {g(0)+g(1)+g(2)} | {g(255)} |")
        out.append("")
        out.append("### 表 E：逐序列（no track）\n")
        out.append("| 序列 | arm A (PC f32 逐尺度) | arm B (PC f32 拼图) | arm C (板端 f16 拼图) |")
        out.append("|:--|--:|--:|--:|")
        per_seq = arms["C"]["per_sequence"]
        for seq in group["sequences"]:
            row_c = per_seq.get(seq, {})
            row_b = arms["B"]["per_sequence"].get(seq, {})
            row_a = arms["A"]["per_sequence"].get(seq, {})
            out.append(
                f"| {seq} | {row_a.get('tp','-')}/{row_a.get('fp','-')}/{row_a.get('fn','-')} "
                f"F1={row_a.get('f1', float('nan')):.6f} | "
                f"{row_b.get('tp','-')}/{row_b.get('fp','-')}/{row_b.get('fn','-')} "
                f"F1={row_b.get('f1', float('nan')):.6f} | "
                f"{row_c.get('tp','-')}/{row_c.get('fp','-')}/{row_c.get('fn','-')} "
                f"F1={row_c.get('f1', float('nan')):.6f} |"
            )
        out.append("")

    t = payload.get("board_timing_ms") or {}
    e = payload.get("e2e_timing_ms") or {}
    if e:
        out.append("## 表 C：板端每帧耗时分解 —— **e2e 库驱动**（原始 IR → GMC → RGA → f16 → 反投影）\n")
        seg_map = [
            ("gmc_fit", "GMC 拟合（Shi-Tomasi + LK + RANSAC + 锚点链合成）"),
            ("gmc_total", "GMC 引擎合计（含 OpenCL fused tail 的 warp/median，计时在 GPU 侧）"),
            ("rga_import", "RGA importbuffer_virtualaddr"),
            ("rga_improcess", "RGA improcess ×3（多尺度拼图装配）"),
            ("upload", "uint8→fp16 + rknn_inputs_set（上传 NPU）"),
            ("rknn_run", "**rknn_run（单次拼图前向，NPU 瓶颈）**"),
            ("outputs_get", "rknn_outputs_get + dequant"),
            ("backproject_cc", "CPU float32 反投影 + 8 连通域"),
        ]
        out.append("| 段 | ms/帧 | 说明 |")
        out.append("|:--|--:|:--|")
        for key, desc in seg_map:
            if key in e:
                out.append(f"| {key} | {e[key]:.3f} | {desc} |")
        if "total_wall" in e:
            out.append(f"| **total_wall** | **{e['total_wall']:.3f}** | 墙钟合计 |")
        if "fps" in e:
            out.append(f"| **fps** | **{e['fps']:.3f}** | **整体帧率** |")
        out.append("")
        if "rknn_run" in e and "npu_workers" in e:
            w = e.get("npu_workers", 2)
            out.append(f"> **瓶颈**：NPU 前向 {e['rknn_run']:.2f} ms/帧；{int(w)} 个 worker 并行 ⇒ "
                       f"理论下限 {e['rknn_run']/max(1.0, float(w)):.2f} ms/帧 ≈ "
                       f"{1000.0*max(1.0, float(w))/max(e['rknn_run'], 1e-9):.1f} fps；"
                       f"实际 {e.get('fps', 0):.2f} fps，差额来自每个 worker 串行的 "
                       f"upload+run+backproj。板卡 RKNN 上下文硬上限为 2，无法再加 worker。\n")
        out.append("> 拼图装配（RGA）只占 ~1 ms，相对一次 NPU 前向可忽略 ⇒ **「拼图一次前向」与「单尺度一次前向」的 NPU 代价相同**，"
                   "差异全部来自输出侧的反投影与连通域。\n")

    if t:
        out.append("## 表 C2：板端每帧耗时分解 —— **C 臂**（BMP 特征 → 拼图 → f16 → 反投影，不含 GMC）\n")
        seg_map = [
            ("decode", "BMP 解码（含 NFS 读 983 KB/帧）"),
            ("rga_import", "RGA importbuffer_virtualaddr"),
            ("rga_improcess", "RGA improcess ×3（拼图装配）"),
            ("fp16_or_upload", "uint8→fp16 + rknn_inputs_set"),
            ("rknn_run", "**rknn_run（单次拼图前向）**"),
            ("outputs_get", "rknn_outputs_get + dequant"),
            ("backproject_cc", "CPU float32 反投影 + 8 连通域"),
            ("write", "TSV 写出"),
        ]
        out.append("| 段 | ms/帧 | 说明 |")
        out.append("|:--|--:|:--|")
        for key, desc in seg_map:
            if key in t:
                out.append(f"| {key} | {t[key]:.3f} | {desc} |")
        if "total_wall" in t:
            out.append(f"| **total_wall** | **{t['total_wall']:.3f}** | 墙钟合计 |")
        if "fps" in t:
            out.append(f"| **fps** | **{t['fps']:.3f}** | |")
        out.append("")

    out.append("## 口径声明（必读）\n")
    out.append("1. **打分器唯一**：`manu/evaluation/system_sota_longquan.py --no-pipeline`（**no track / 单帧臂**），"
               "`match_mode=inbox`、`tol=8.0`、`pipeline=false`，只用 tag 0/1/2。")
    out.append("2. **每条轨道用自己 gate 产物里的 `config.params`**（`track2_golden.json` / `track3_blind.json`），"
               "**不能手抄、不能用脚本默认值**（默认 `th_base=0.22` / `dedup_radius=32` 会给出完全不同的数）。")
    out.append("3. **A 与 B/C 不是同一个计算图**：A 是逐尺度 3 次前向，B/C 是单次拼图前向。"
               "⇒ ΔF1(A→B) 是打包损失，ΔF1(B→C) 才是 f16/量化损失，**不能把 A→C 整段记在量化账上**。")
    out.append("4. **画布几何差异**：PC native 用 letterbox（上下各 64 行 pad=114）；拼图里 native 直接贴 (0,0) 无 pad；"
               "PC 缩略图画布背景是模型序 (residual=0, diff=0, I_t=mean)，拼图统一背景 114。"
               "⇒ **已知口径断裂**，B1 臂量化了它的上限。")
    out.append("5. **反投影方向不同**：PC 是「320→640 先放大再裁」，板端是「先裁 320 格再放大」，"
               "亚像素相位差 ≈0.25 输入像素。**非逐比特等价**。")
    out.append("6. **评测本身接近「每帧一个候选」**：`merge_scales(dedup_radius=768)` 的半径大于帧内任意两点距离，"
               "所以每帧基本只保留最高分的那一个候选 ⇒ 指标主要由「最高分候选是否落进 GT 框」决定，"
               "**不是常规的多框平均精度**。")
    out.append("7. **特征溯源闭环**：板端 BMP = `cv2.imwrite(cv2.imread(official JPG))` 的无损重编码，"
               "与 gate 缓存所用的 official 特征逐像素一致"
               "（`manu/pipeline/npu/ptq/06_verify_reference_anchor.py` 的 pixel_proof：mismatches=0 / max_abs_diff=0）。"
               "⇒ 板端臂与 PC 臂吃的是同一批像素，**不是两个特征源**。")
    out.append("8. **gate/epoch3 与 lq_eval 不可混用**：gate 是 official 逐 lag 拟合特征、8 序列；"
               "lq_eval 是 FPV 仿射连乘特征、24 序列，逐像素差 26~70/255。本表只用 gate 口径。\n")
    return "\n".join(out)


# --------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", action="append", default=[], required=True,
                    help="NAME=SEQ1,SEQ2,... (repeatable)")
    ap.add_argument("--board-tsv-dir", required=True, help="dir with <SEQ>_mosaic.tsv")
    ap.add_argument("--e2e-tsv-dir", default="",
                    help="dir with <SEQ>_mosaic.tsv produced by the END-TO-END demo "
                         "(rknn_mosaic_e2e: raw infrared -> GMC/OpenCL step10 -> RGA mosaic -> "
                         "board f16 -> back-projection).  Enables the arm 'E'.")
    ap.add_argument("--pc-area-tsv-dir", default="", help="dir with <SEQ>_pc_mosaic.tsv (INTER_AREA)")
    ap.add_argument("--pc-linear-tsv-dir", default="", help="dir with <SEQ>_pc_mosaic.tsv (INTER_LINEAR)")
    ap.add_argument("--gate-cache", default=str(GATE_DIR / "ms_cache"))
    ap.add_argument("--gate-dir", default=str(GATE_DIR))
    ap.add_argument("--out", required=True)
    ap.add_argument("--board-timing", default="", help="a board timing.txt (for the timing table)")
    ap.add_argument("--e2e-timing", default="",
                    help="the e2e demo's timing.txt (library-driven: 10-segment breakdown). "
                         "This is the timing table the deliverable reports.")
    ap.add_argument("--board-timing-dir", default="", help="dir with <SEQ>/timing.txt (per-sequence)")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    groups_spec = []
    for spec in args.group:
        name, _, csv = spec.partition("=")
        groups_spec.append((name.strip(), [s.strip() for s in csv.split(",") if s.strip()]))

    gate_cache = Path(args.gate_cache)
    payload = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "groups": [], "board_timing_ms": {},
               "e2e_timing_ms": {}}

    for name, sequences in groups_spec:
        print(f"\n===== group {name}: {len(sequences)} sequences =====", flush=True)
        gdir = GATE_DIR
        params_src = gdir / f"{name}.json"
        if not params_src.is_file():
            params_src = gdir / "track3_blind.json"
        params = json.loads(params_src.read_text(encoding="utf-8"))["config"]["params"]
        params_json = out_root / f"gate_params_{name}.json"
        params_json.write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{name}] th_base={params['th_base']} dedup_radius={params['dedup_radius']} "
              f"(from {params_src.name})", flush=True)

        arms: dict[str, dict] = {}
        tags: dict[str, dict] = {}

        def meta_for(source: str, seq: str) -> dict:
            return {
                "sequence": seq, "frames": 0, "native_size": [640, 512], "imgsz": 640,
                "stride": 2, "scales": [0, 160, 80], "fused_tag": 255,
                "coordinate_space": "native_pixel", "region_threshold": 0.06,
                "main_threshold": 0.22, "fusion_dilate": 32, "min_area": 4,
                "use_p0_highway": False, "source": source,
                "features_root": f"/mnt/manu/ptq/bmp/{seq}",
                "hm_weights": "trial_0080/weights/epoch3.pt",
            }

        # ---- A / A0 : the frozen gate cache --------------------------------------
        root_a = out_root / f"cache_{name}_A"
        root_a0 = out_root / f"cache_{name}_A0"
        for seq in sequences:
            gate = pickle.loads((gate_cache / f"{seq}.pkl").read_bytes())
            write_per_seq(root_a, seq, gate)
            write_per_seq(root_a0, seq, filter_tags(gate, {0}))
        res = score_group(root_a, sequences, params_json, "A")
        arms["A"] = {"metrics": res["overall"], "per_sequence": res["per_sequence"]}
        tags["A"] = {}
        for seq in sequences:
            g = pickle.loads((gate_cache / f"{seq}.pkl").read_bytes())
            tags["A"].update(per_tag(g))
        res = score_group(root_a0, sequences, params_json, "A0")
        arms["A0"] = {"metrics": res["overall"], "per_sequence": res["per_sequence"]}
        tags["A0"] = {}
        for seq in sequences:
            g = pickle.loads((root_a0 / f"{seq}.pkl").read_bytes())
            tags["A0"].update(per_tag(g))
        print(f"[{name}] A  = {arms['A']['metrics']['f1']:.6f}  "
              f"A0 = {arms['A0']['metrics']['f1']:.6f}", flush=True)

        # ---- B / B1 / B0 : the PC-side mosaic control -----------------------------
        for key, tsv_dir, suffix in (("B", args.pc_area_tsv_dir, "B"),
                                     ("B1", args.pc_linear_tsv_dir, "B1")):
            if not tsv_dir:
                continue
            root = out_root / f"cache_{name}_{suffix}"
            root0 = out_root / f"cache_{name}_{suffix}0"
            for seq in sequences:
                tsv = Path(tsv_dir) / f"{seq}_pc_mosaic.tsv"
                cache = build_cache(parse_mosaic_tsv(tsv), meta_for(f"pc_mosaic_f32_{suffix}", seq))
                write_per_seq(root, seq, cache)
                write_per_seq(root0, seq, filter_tags(cache, {0}))
            res = score_group(root, sequences, params_json, suffix)
            arms[key] = {"metrics": res["overall"], "per_sequence": res["per_sequence"]}
            tags[key] = {}
            for seq in sequences:
                tags[key].update(per_tag(pickle.loads((root / f"{seq}.pkl").read_bytes())))
            if key == "B":
                res0 = score_group(root0, sequences, params_json, "B0")
                arms["B0"] = {"metrics": res0["overall"], "per_sequence": res0["per_sequence"]}
                tags["B0"] = {}
                for seq in sequences:
                    tags["B0"].update(per_tag(pickle.loads((root0 / f"{seq}.pkl").read_bytes())))
            print(f"[{name}] {key} = {arms[key]['metrics']['f1']:.6f}", flush=True)

        # ---- C / C0 : the board ---------------------------------------------------
        root_c = out_root / f"cache_{name}_C"
        root_c0 = out_root / f"cache_{name}_C0"
        frames = 0
        for seq in sequences:
            tsv = Path(args.board_tsv_dir) / f"{seq}_mosaic.tsv"
            cache = build_cache(parse_mosaic_tsv(tsv), meta_for("board_mosaic_f16", seq))
            frames += len(cache["frame_index"])
            write_per_seq(root_c, seq, cache)
            write_per_seq(root_c0, seq, filter_tags(cache, {0}))
        res = score_group(root_c, sequences, params_json, "C")
        arms["C"] = {"metrics": res["overall"], "per_sequence": res["per_sequence"]}
        tags["C"] = {}
        for seq in sequences:
            tags["C"].update(per_tag(pickle.loads((root_c / f"{seq}.pkl").read_bytes())))
        res = score_group(root_c0, sequences, params_json, "C0")
        arms["C0"] = {"metrics": res["overall"], "per_sequence": res["per_sequence"]}
        tags["C0"] = {}
        for seq in sequences:
            tags["C0"].update(per_tag(pickle.loads((root_c0 / f"{seq}.pkl").read_bytes())))
        print(f"[{name}] C  = {arms['C']['metrics']['f1']:.6f}  "
              f"C0 = {arms['C0']['metrics']['f1']:.6f}", flush=True)

        # ---- E / E0 : the END-TO-END demo (raw infrared -> GMC -> mosaic -> NPU) ----
        if args.e2e_tsv_dir:
            root_e = out_root / f"cache_{name}_E"
            root_e0 = out_root / f"cache_{name}_E0"
            for seq in sequences:
                tsv = Path(args.e2e_tsv_dir) / f"{seq}_mosaic.tsv"
                cache = build_cache(parse_mosaic_tsv(tsv), meta_for("e2e_gmc_step10_board_f16", seq))
                write_per_seq(root_e, seq, cache)
                write_per_seq(root_e0, seq, filter_tags(cache, {0}))
            res = score_group(root_e, sequences, params_json, "E")
            arms["E"] = {"metrics": res["overall"], "per_sequence": res["per_sequence"]}
            tags["E"] = {}
            for seq in sequences:
                tags["E"].update(per_tag(pickle.loads((root_e / f"{seq}.pkl").read_bytes())))
            res = score_group(root_e0, sequences, params_json, "E0")
            arms["E0"] = {"metrics": res["overall"], "per_sequence": res["per_sequence"]}
            tags["E0"] = {}
            for seq in sequences:
                tags["E0"].update(per_tag(pickle.loads((root_e0 / f"{seq}.pkl").read_bytes())))
            print(f"[{name}] E  = {arms['E']['metrics']['f1']:.6f}  "
                  f"E0 = {arms['E0']['metrics']['f1']:.6f}", flush=True)

        for key in arms:
            arms[key]["tags"] = tags[key]

        payload["groups"].append({
            "name": name, "sequences": sequences, "frames": frames,
            "description": f"{len(sequences)} sequences / {frames} frames",
            "params": params, "params_source": str(params_src), "arms": arms,
        })

    if args.e2e_timing:
        payload["e2e_timing_ms"] = parse_timing(Path(args.e2e_timing))
    if args.board_timing:
        payload["board_timing_ms"] = parse_timing(Path(args.board_timing))
    if args.board_timing_dir:
        merged: dict[str, float] = {}
        total_frames = 0.0
        for d in sorted(Path(args.board_timing_dir).iterdir()):
            if not d.is_dir():
                continue
            f = d / "timing.txt"
            if not f.is_file():
                continue
            t = parse_timing(f)
            n = t.get("frames", 0)
            if not isinstance(n, (int, float)):
                continue
            total_frames += n
            for key, value in t.items():
                if key.endswith("_total_ms") or key in ("frames", "seq", "canvas_mode", "input_mode"):
                    continue
                if not isinstance(value, (int, float)):
                    continue
                merged[key] = merged.get(key, 0.0) + value * n
        if total_frames:
            for key in list(merged):
                merged[key] /= total_frames
            merged["frames"] = total_frames
            payload["board_timing_ms"] = merged

    (out_root / "mosaic_vs_f32_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_root / "mosaic_vs_f32_report.md").write_text(render_markdown(payload), encoding="utf-8")
    for g in payload["groups"]:
        print(f"\n### {g['name']}: " + "  ".join(
            f"{k}={v['metrics']['f1']:.6f}" for k, v in g["arms"].items()), flush=True)
    print(f"\n[ok] -> {out_root/'mosaic_vs_f32_report.md'}")


if __name__ == "__main__":
    main()
