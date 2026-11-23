#!/usr/bin/env python3
"""第 1~2 步：抽取「好片母本片段」并生成三组梯度退化视频。

动机：上一版画质合格线是**在 4 条真实坏片上硬切阈值**得到的，样本太少、无法回答
「画面差到什么程度跟不住」这个连续问题。本脚本造一批**受控退化**的好片：
从好片抽母本 → 按真实病因施加梯度退化 → 得到一条「画质 ↔ 跟踪成功率」的衰减曲线，
拐点（物理崩溃点）就是新的合格线。

母本选取（Step 1）
------------------
用已有的 ``report/windows.csv``（每条片逐 1 秒窗口的画质指标）挑**质量最接近该片
中位数**的那 1 秒窗口的起点，再取连续 12 秒。⇒ 不需要解码就能选到「典型片段」，
且母本代表该片的中位水平而不是最好/最坏的一段。

⚠️ ``windows.csv`` 的 ``file`` 列**带扩展名**（如 ``VIDEO00007_..._012038IR.h264``），
匹配必须用 ``Path(path).name`` 而不是 ``.stem``；裸 ``.h264`` 用 ``ffprobe`` 取时长
会得到 ``N/A``，时长一律从 ``windows.csv`` 的 ``max(window)+1`` 推。

三组退化（Step 2）
------------------
  A 失焦/模糊（病因：糊）→ 影响 lap_var / edge_density
      高斯模糊 sigma = 0.5 / 1.0 / 1.5 / 2.0 / 2.5 / 3.0 / 3.5 / 4.0（8 档）
  B 反差坍缩/发灰（病因：灰）→ 影响 global_contrast / entropy
      向中间灰 128 线性收拢，压缩比 = 0.80 / 0.65 / 0.50 / 0.35 / 0.25 / 0.15（6 档）
  C 编码码率压制（病因：压）→ 影响 blockiness
      FFmpeg H.264 真实重编码 CRF = 23 / 26 / 29 / 32 / 36 / 40 / 44（7 档）

🔴 **编码口径（与目标书 §13.2 字面写法的偏离，务必知悉）**
目标书写「所有退化都在 CRF 18 母本这一层之上施加」。本脚本采用**单次有损编码**口径：
每个变体都**直接从原始源**解码 →（可选）加退化滤镜 → **只编码一次**。
  * blur / contrast 变体 = 源 + 滤镜 + CRF 18（与母本同 CRF，**与母本只差滤镜**）
  * crf 变体 = 源 + 无滤镜 + CRF N（与母本只差 CRF）
理由：若改成「母本(CRF18) → 滤镜 → 再编码」，每个变体会多一代压缩，
CRF 梯度上就会混入「二次压缩」这一额外变量，膝点的物理含义被污染；
而单次编码口径下，**同一梯度族内只有一个自变量**，曲线可解释。
代价是「退化施加在母本之上」的**字面**写法被打破，故在此显式记录，供报告 §9.5 引用。

⚠️ **裸 ``.h264`` 的 seek**：`ffmpeg -ss X -i src`（输入侧快进）对裸 h264 **无效**
（实测只出 262 字节空文件）；必须用**输出侧** ``-i src -ss X``（精确 seek，
逐帧丢弃到目标位置）。已实测 25 s 起点落在源第 750 帧（25 s × 30 fps）。

⚠️ **必须用 ``-frames:v`` 而不是 ``-t``**：ffmpeg 4.2 在**一路输入多路输出**时，
输出侧 ``-t`` 会把后面的输出越截越短（实测 22 路输出时只有第 1 路拿满 12 s，
其余依次变成 8 s / 4 s / 0 s）；``-frames:v`` 给精确帧数，逐路一致（22 路实测全部 360 帧）。

产物
----
  <out-dir>/videos/<src>__<gradient>__<level>.mp4      退化视频
  <out-dir>/variants.csv                              变体清单（供指标/SOT 两路对齐）
  <out-dir>/jobs.tsv                                  SOT 探针作业表（path\tgroup\tkind\tmodality）

用法（服务器 192.168.99.40:32222，工作目录 /tmp/pycharm_project_a7d966e6）：
    source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav
    python manu/video_quality/vq_degrade_prepare.py \\
        --jobs /tmp/sot_jobs.tsv \\
        --windows manu/video_quality/report/windows.csv \\
        --out-dir /tmp/vq_degrade --workers 3 --enc-threads 2
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
for _path in (str(PROJECT_ROOT), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# 退化梯度定义：gradient -> [(level 标签, 数值)]
BLUR_SIGMAS = [("s0.5", 0.5), ("s1.0", 1.0), ("s1.5", 1.5), ("s2.0", 2.0),
               ("s2.5", 2.5), ("s3.0", 3.0), ("s3.5", 3.5), ("s4.0", 4.0)]
CONTRAST_RATIOS = [("c0.80", 0.80), ("c0.65", 0.65), ("c0.50", 0.50),
                   ("c0.35", 0.35), ("c0.25", 0.25), ("c0.15", 0.15)]
CRF_LEVELS = [("q23", 23), ("q26", 26), ("q29", 29), ("q32", 32),
              ("q36", 36), ("q40", 40), ("q44", 44)]

MOTHER_CRF = 18
# 选母本窗口用的代表指标（与 §271/§9 同一组有效指标）
REPRESENTATIVE_METRICS = ("lap_var", "edge_density", "global_contrast", "entropy")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def probe_fps(path: str) -> float:
    """裸 .h264 的 duration 是 N/A，但 r_frame_rate 读得到。"""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    text = result.stdout.decode().strip()
    try:
        numerator, denominator = text.split("/")
        return float(numerator) / float(denominator)
    except Exception:
        return 30.0


def run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError("command failed: %s\n%s"
                           % (" ".join(cmd), result.stderr.decode("utf-8", "replace")[-2000:]))


def pick_mother_start(rows: list[dict[str, str]], seconds: float) -> tuple[float, str, float]:
    """挑质量最接近该片中位水平的 1 秒窗口作为母本起点。

    每个指标先按该片自身的「中位数 + 极差」归一（指标量纲差一个数量级，直接比会失真），
    再取 L1 距离最小的窗口。返回 (起点秒, 理由, 该片时长秒)。
    """
    usable: list[tuple[int, list[float]]] = []
    for row in rows:
        try:
            window = int(row["window"])
            vector = [float(row[m]) for m in REPRESENTATIVE_METRICS]
        except (TypeError, ValueError, KeyError):
            continue
        usable.append((window, vector))
    duration = float(max((w for w, _ in usable), default=0) + 1)
    if not usable:
        return 0.0, "no-windows (fallback: 起点 0)", duration
    columns = list(zip(*[vector for _, vector in usable]))
    medians = [sorted(column)[len(column) // 2] for column in columns]
    spans = [(max(column) - min(column)) or 1.0 for column in columns]
    best_window, best_distance = usable[0][0], float("inf")
    for window, vector in usable:
        distance = sum(abs(v - m) / s for v, m, s in zip(vector, medians, spans))
        if distance < best_distance:
            best_window, best_distance = window, distance
    start = float(best_window)
    if duration and start + seconds > duration:
        start = max(0.0, duration - seconds)
    reason = "median-quality window #%d (normalized L1 dist=%.4f)" % (best_window, best_distance)
    if duration and duration < seconds:
        reason += " | 片长 %.0fs < %.0fs，取全长" % (duration, seconds)
    return start, reason, duration


def build_variants(stem: str, source: str, modality: str, start: float,
                   seconds: float, videos_dir: Path) -> list[dict]:
    """列出母本 + 三组退化的全部变体（含目标文件名与 ffmpeg 参数）。"""
    specs: list[tuple[str, str, str, str, int]] = []
    specs.append(("base", "base", "", "", MOTHER_CRF))
    for label, sigma in BLUR_SIGMAS:
        specs.append(("blur", label, "%.2f" % sigma, "gblur=sigma=%.3f" % sigma, MOTHER_CRF))
    for label, ratio in CONTRAST_RATIOS:
        expr = "128+(val-128)*%.4f" % ratio
        specs.append(("contrast", label, "%.2f" % ratio,
                      "lutrgb=r='%s':g='%s':b='%s'" % (expr, expr, expr), MOTHER_CRF))
    for label, crf in CRF_LEVELS:
        specs.append(("crf", label, str(crf), "", crf))

    variants = []
    for gradient, level, level_value, vf, crf in specs:
        if gradient == "base":
            name = "%s__mother.mp4" % stem
        else:
            name = "%s__%s__%s.mp4" % (stem, gradient, level)
        variants.append({
            "variant_id": name[:-4],
            "source_clip": stem,
            "source_path": source,
            "modality": modality,
            "gradient": gradient,
            "level": level,
            "level_value": level_value,
            "video_path": str(videos_dir / name),
            "vf": vf,
            "crf": crf,
        })
    return variants


def encode_clip(job: dict) -> dict:
    """一条源片 → 母本 + 全部退化变体（一次解码、多路输出）。"""
    source = job["source_path"]
    start = job["start"]
    n_frames = int(round(job["seconds"] * job["fps"]))
    variants = job["variants"]
    enc_threads = job["enc_threads"]

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", source]
    for variant in variants:
        cmd += ["-map", "0:v", "-ss", "%.3f" % start, "-frames:v", str(n_frames),
                "-an", "-sn"]
        if variant["vf"]:
            cmd += ["-vf", variant["vf"]]
        cmd += ["-c:v", "libx264", "-crf", str(variant["crf"]), "-preset", "medium",
                "-pix_fmt", "yuv420p", "-threads", str(enc_threads), variant["video_path"]]
    started = time.perf_counter()
    run(cmd)
    return {"clip": job["stem"], "n_variants": len(variants),
            "seconds": time.perf_counter() - started}


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract mother clips and build the degradation ladder")
    parser.add_argument("--jobs", required=True, help="TSV: path\\tgroup\\tkind\\tmodality (取 group=good)")
    parser.add_argument("--windows", required=True, help="report/windows.csv，用于选代表性母本窗口")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seconds", type=float, default=12.0, help="母本片段长度（秒）")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 条（冒烟测试）")
    parser.add_argument("--workers", type=int, default=3, help="并发源片数")
    parser.add_argument("--enc-threads", type=int, default=2, help="每个编码器线程数")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不调用 ffmpeg")
    cli = parser.parse_args()

    out_dir = Path(cli.out_dir)
    videos_dir = out_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    windows_rows = read_csv(Path(cli.windows))
    by_file: dict[str, list[dict[str, str]]] = {}
    for row in windows_rows:
        by_file.setdefault(row["file"], []).append(row)

    clips = []
    for line in Path(cli.jobs).read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) < 4 or parts[1] != "good":
            continue
        clips.append((parts[0], parts[3]))
    if cli.limit:
        clips = clips[: cli.limit]
    print("[PREP] %d good source clips" % len(clips), flush=True)

    all_variants: list[dict] = []
    jobs_for_ffmpeg = []
    for path, modality in clips:
        stem = Path(path).stem
        rows = by_file.get(Path(path).name, [])
        start, reason, duration = pick_mother_start(rows, cli.seconds)
        seconds = min(cli.seconds, duration) if duration else cli.seconds
        fps = probe_fps(path)
        n_frames = int(round(seconds * fps))
        variants = build_variants(stem, path, modality, start, seconds, videos_dir)
        for variant in variants:
            variant["mother_start_s"] = "%.3f" % start
            variant["mother_seconds"] = "%.3f" % seconds
            variant["mother_reason"] = reason
            variant["clip_duration_s"] = "%.0f" % duration
            variant["fps"] = "%.6f" % fps
            variant["n_frames"] = str(n_frames)
        all_variants.extend(variants)
        jobs_for_ffmpeg.append({
            "stem": stem, "source_path": path, "start": start, "seconds": seconds,
            "fps": fps, "variants": variants, "enc_threads": cli.enc_threads,
        })
        print("[PLAN] %-30s start=%6.1fs len=%.0fs frames=%d %s"
              % (stem, start, seconds, n_frames, reason), flush=True)

    if cli.dry_run:
        print("[PREP] dry-run: %d variants would be produced" % len(all_variants))
        return

    print("[PREP] encoding %d variants (%d clips) with %d workers, %d threads/encoder"
          % (len(all_variants), len(jobs_for_ffmpeg), cli.workers, cli.enc_threads), flush=True)
    started = time.perf_counter()
    done = 0
    with ProcessPoolExecutor(max_workers=cli.workers) as pool:
        futures = [pool.submit(encode_clip, job) for job in jobs_for_ffmpeg]
        for future in as_completed(futures):
            done += 1
            result = future.result()
            print("[ENCODE] %-30s %d variants in %.1fs  [%d/%d, %.0fs]"
                  % (result["clip"], result["n_variants"], result["seconds"],
                     done, len(jobs_for_ffmpeg), time.perf_counter() - started), flush=True)

    variants_path = out_dir / "variants.csv"
    fields = ["variant_id", "source_clip", "source_path", "modality", "gradient", "level",
              "level_value", "video_path", "mother_start_s", "mother_seconds",
              "clip_duration_s", "mother_reason", "fps", "n_frames"]
    with variants_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for variant in all_variants:
            writer.writerow({key: variant[key] for key in fields})

    jobs_path = out_dir / "jobs.tsv"
    with jobs_path.open("w", encoding="utf-8") as handle:
        for variant in all_variants:
            handle.write("%s\t%s\t%s\t%s\n" % (
                variant["video_path"], "degraded", variant["gradient"], variant["modality"]))

    print("[PREP] %d variants -> %s" % (len(all_variants), variants_path))
    print("[PREP] SOT job table -> %s" % jobs_path)


if __name__ == "__main__":
    main()
