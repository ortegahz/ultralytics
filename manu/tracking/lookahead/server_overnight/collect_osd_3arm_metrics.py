#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""P4 —— 汇总 OSD 三臂对照（stale / replay / original）的序列级数字与视频清单。

口径（与仓库既有报告一致）：
* **交付基线 = ``stale`` 臂**；``original`` 臂是**零延迟 oracle 上界，不是基线**。
* **总体 CLE 用 ``cle_pooled_*``（全帧池化）**，不用逐序列最差值（``cle_p95``）。
* 8 序 gate 集只覆盖 24 序官方口径的 **20.39% GT** ⇒ 只能做机理筛选 / A-B，
  **不能给交付档排序**。
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
ARMS = ("stale", "replay", "original")


def mtime_str(path: Path) -> str:
    """产物修改时间（可读格式）。"""
    return datetime.datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds")


def ffprobe(path: Path) -> dict:
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-count_frames",
                "-show_entries", "stream=nb_read_frames,width,height,r_frame_rate,codec_name",
                "-show_entries", "format=duration",
                "-of", "json", str(path),
            ],
            capture_output=True, text=True, timeout=120,
        )
        data = json.loads(out.stdout)
        stream = (data.get("streams") or [{}])[0]
        num, den = (stream.get("r_frame_rate") or "0/1").split("/")
        fps = float(num) / float(den) if float(den) else 0.0
        duration = float((data.get("format") or {}).get("duration") or 0.0)
        return {
            "codec": stream.get("codec_name"),
            "width": stream.get("width"),
            "height": stream.get("height"),
            "fps": fps,
            "duration_s": duration,
            "frames": int(stream["nb_read_frames"]) if stream.get("nb_read_frames") else None,
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def main() -> None:
    ap = argparse.ArgumentParser(description="汇总 OSD 三臂对照数字与视频清单")
    ap.add_argument("--arms-json", default=str(REPO / "runs/lookahead/osd/nogmc_cs_t0043_d3/arms_replay.json"))
    ap.add_argument("--video-dir", default=str(REPO / "runs/lookahead/osd/videos"))
    ap.add_argument("--tag", default="cs_t0043_d3")
    ap.add_argument("--seqs", default="VIDEO00007_19700101_001324,VIDEO00033_19700101_015000")
    ap.add_argument("--out-md", default="")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    arms_json = Path(args.arms_json).resolve()
    data = json.loads(arms_json.read_text(encoding="utf-8"))
    arms = data["arms"]
    seqs = [s for s in args.seqs.split(",") if s]

    per_seq = []
    for seq in seqs:
        row = {"sequence": seq}
        for arm in ARMS:
            cf = arms[arm]["current_frame"]["per_sequence"][seq]
            sv = arms[arm]["servo"]["per_sequence"][seq]
            row[arm] = {
                "f1": cf["f1"], "recall": cf["recall"], "precision": cf["precision"],
                "far_per_frame": cf["far_per_frame"], "tp": cf["tp"], "fp": cf["fp"], "fn": cf["fn"],
                "gt": cf["gt"], "det": cf["det"],
                "hit_rate_5px": sv["hit_rate_5px"], "hit_rate_inbox": sv["hit_rate_inbox"],
                "coverage": sv["coverage"],
                "cle_p95_worst_seq": sv["cle_p95"], "cle_mean": sv["cle_mean"], "cle_max": sv["cle_max"],
                "fragmentation": sv["fragmentation"], "lost_frames": sv["lost_frames"],
            }
        per_seq.append(row)

    overall = {"sequence": "OVERALL(8序池化)"}
    for arm in ARMS:
        cf = arms[arm]["current_frame"]["overall"]
        sv = arms[arm]["servo"]["overall"]
        overall[arm] = {
            "f1": cf["f1"], "recall": cf["recall"], "precision": cf["precision"],
            "far_per_frame": cf["far_per_frame"], "tp": cf["tp"], "fp": cf["fp"], "fn": cf["fn"],
            "gt": cf["gt"], "det": cf["det"],
            "hit_rate_5px": sv["hit_rate_5px"], "hit_rate_inbox": sv["hit_rate_inbox"],
            "cle_pooled_p50": sv["cle_pooled_p50"], "cle_pooled_p90": sv["cle_pooled_p90"],
            "cle_pooled_p95": sv["cle_pooled_p95"], "cle_pooled_p99": sv["cle_pooled_p99"],
            "cle_pooled_max": sv["cle_pooled_max"], "cle_pooled_mean": sv["cle_pooled_mean"],
            "cle_pooled_samples": sv["cle_pooled_samples"],
            "cle_p95_worst_seq": sv["cle_p95"],
        }

    # ---- 视频清单 ----
    video_dir = Path(args.video_dir)
    videos = []
    for seq in seqs:
        for arm in ARMS:
            mp4 = video_dir / f"{seq}_3arm_{arm}_{args.tag}.mp4"
            if not mp4.exists():
                continue
            videos.append({
                "sequence": seq, "arm": arm, "path": str(mp4),
                "size_bytes": mp4.stat().st_size,
                "mtime": mtime_str(mp4),
                **ffprobe(mp4),
            })

    windows = {}
    for seq in seqs:
        w = video_dir.parent / "windows" / f"{seq}_3arm.json"
        if w.exists():
            wd = json.loads(w.read_text(encoding="utf-8"))
            windows[seq] = {"start": wd["start"], "end": wd["end"], "limit": wd["limit"],
                            "criterion": wd.get("criterion"), "separation_score": wd.get("separation_score"),
                            "picked_on": "rc2 (trial_0080ep3 / GMC 语义) 模型"}

    result = {
        "task": "P4 —— no-GMC (cs_t0043) OSD 三臂对照视频 + 序列级数字",
        "host": "test-Super-Server",
        "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": args.tag,
        "delta_records": data.get("lookahead", {}).get("delta"),
        "cache_root": data.get("config", {}).get("cache_root"),
        "arms_json": str(arms_json),
        "per_sequence": per_seq,
        "overall": overall,
        "windows": windows,
        "videos": videos,
        "caveats": [
            "交付基线 = stale 臂；original 臂 = 零延迟 oracle 上界，不是基线。",
            "总体 CLE 用 cle_pooled_*（池化）；cle_p95（逐序列最差值）仅作参考，不跨序列合并。",
            "8 序 gate 集只覆盖 24 序官方口径的 20.39% GT ⇒ 只能做机理筛选 / A-B，不能给交付档排序。",
            "窗口 start/limit 沿用 rc2 模型上按「stale vs replay 分离度最大」挑的窗口，保证与历史视频可比；cs_t0043 上的分离度可能不同。",
        ],
    }

    out_json = Path(args.out_json) if args.out_json else video_dir.parent / f"osd_3arm_{args.tag}.json"
    out_md = Path(args.out_md) if args.out_md else video_dir.parent / f"osd_3arm_{args.tag}.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    L = []
    a = L.append
    a(f"# P4 —— OSD 三臂对照（{args.tag}）")
    a("")
    a(f"- 执行机：**服务器 test-Super-Server**（无 GPU；本任务纯 CPU）")
    a(f"- 时间戳：**{result['timestamp']}**")
    a(f"- 模型包：`{data.get('config', {}).get('cache_root')}`")
    a(f"- Δ = **{result['delta_records']} record**（120 ms @25 Hz）")
    a(f"- 插件产物：`{arms_json}`")
    a("")
    a("## 1. 序列级三臂数字（视频 ↔ 数字对得上）")
    a("")
    for row in per_seq:
        a(f"### {row['sequence']}")
        a("")
        a("| 臂 | F1 | R | P | FAR | TP | FP | FN | hit@5px | hit@8px(inbox) | CLE mean | CLE p95(本序列) | 碎片 |")
        a("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for arm in ARMS:
            m = row[arm]
            a(
                f"| {arm} | {m['f1']:.6f} | {m['recall']*100:.2f}% | {m['precision']*100:.2f}% | "
                f"{m['far_per_frame']:.4f} | {m['tp']} | {m['fp']} | {m['fn']} | "
                f"{m['hit_rate_5px']*100:.2f}% | {m['hit_rate_inbox']*100:.2f}% | "
                f"{m['cle_mean']:.3f} | {m['cle_p95_worst_seq']:.3f} | {m['fragmentation']} |"
            )
        a("")
    a("### 汇总（8 序 gate 集池化，`cle_pooled_*`）")
    a("")
    a("| 臂 | F1 | R | P | FAR | TP | FP | FN | hit@5px | CLE pooled p50 | CLE pooled p95 | CLE pooled max |")
    a("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for arm in ARMS:
        m = overall[arm]
        a(
            f"| {arm} | {m['f1']:.6f} | {m['recall']*100:.2f}% | {m['precision']*100:.2f}% | "
            f"{m['far_per_frame']:.4f} | {m['tp']} | {m['fp']} | {m['fn']} | "
            f"{m['hit_rate_5px']*100:.2f}% | {m['cle_pooled_p50']:.3f} | {m['cle_pooled_p95']:.3f} | {m['cle_pooled_max']:.3f} |"
        )
    a("")
    a("## 2. 视频产物清单")
    a("")
    a("| 序列 | 臂 | 路径 | 大小 | 时长 | 帧数 | 分辨率 | 编码 | fps |")
    a("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for v in videos:
        a(
            f"| {v['sequence']} | {v['arm']} | `{v['path']}` | {v['size_bytes']/1e6:.2f} MB | "
            f"{v.get('duration_s', 0):.2f} s | {v.get('frames')} | {v.get('width')}x{v.get('height')} | "
            f"{v.get('codec')} | {v.get('fps', 0):.2f} |"
        )
    a("")
    a("## 3. 渲染窗口（沿用历史，保证与 rc2 视频逐帧可比）")
    a("")
    a("| 序列 | start | end | limit | 挑选准则 | 分离度 | 挑选所用模型 |")
    a("| --- | --- | --- | --- | --- | --- | --- |")
    for seq, w in windows.items():
        a(
            f"| {seq} | {w['start']} | {w['end']} | {w['limit']} | {w['criterion']} | {w['separation_score']:.4f} | {w['picked_on']} |"
        )
    a("")
    a("## 4. 口径与限制")
    a("")
    for c in result["caveats"]:
        a(f"- {c}")
    a("")
    out_md.write_text("\n".join(L), encoding="utf-8")
    print(f"[SUCCESS] {out_json}")
    print(f"[SUCCESS] {out_md}")
    for row in per_seq:
        print(
            f"[SEQ] {row['sequence']} "
            + " | ".join(
                f"{arm}: F1={row[arm]['f1']:.6f} "
                f"hit@5px={row[arm]['hit_rate_5px']*100:.2f}% FAR={row[arm]['far_per_frame']:.4f}"
                for arm in ARMS
            )
        )


if __name__ == "__main__":
    main()
