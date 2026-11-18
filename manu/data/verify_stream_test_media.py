#!/usr/bin/env python3
"""校验嵌入式组推流测试素材：原始 mp4 与 640x512 视频是否严格复刻 BMP 处理链。

校验项：
  A. 元数据（分辨率/帧率/帧数/像素格式）
  B. 原始 mp4 是否像素级无损于源 .h264（framemd5 全量比对）
  C. 640x512 无损灰度视频 vs BMP 全帧逐像素比对（期望 maxdiff=0）
  D. 640x512 推流版(yuv420p) vs BMP 抽样比对（给出 maxdiff / mean / PSNR）
  E. 独立复算：原始 mp4 --(OpenCV BMP 链)--> BMP 抽样比对
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

import cv2
import numpy as np

W, H = 640, 512
FS = W * H


def sh(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, **kw)


def probe(path):
    out = sh(["ffprobe", "-v", "error", "-select_streams", "v:0",
              "-show_entries", "stream=width,height,r_frame_rate,nb_frames,pix_fmt,codec_name",
              "-show_entries", "format=format_name,duration,size",
              "-of", "json", path]).stdout
    d = json.loads(out)
    st = d["streams"][0]
    fmt = d.get("format", {})
    n = st.get("nb_frames")
    if not n or n == "N/A":
        cnt = sh(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                  "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path]).stdout
        n = cnt.stdout.decode().strip().splitlines()[-1]
    return {
        "codec": st.get("codec_name"),
        "size": f"{st['width']}x{st['height']}",
        "fps": st.get("r_frame_rate"),
        "frames": int(n),
        "pix_fmt": st.get("pix_fmt"),
        "duration_s": round(float(fmt.get("duration", 0)), 3),
        "bytes": int(fmt.get("size", os.path.getsize(path))),
    }


def framemd5(path, raw=False):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if raw:
        cmd += ["-f", "h264"]
    cmd += ["-i", path, "-f", "framemd5", "-"]
    return sh(cmd).stdout


def stream_gray(mp4):
    """按帧顺序流出 gray 原始像素（精确，无 seek 误差）"""
    p = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", mp4,
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        stdout=subprocess.PIPE, bufsize=FS * 4)
    try:
        while True:
            buf = p.stdout.read(FS)
            if len(buf) < FS:
                break
            yield np.frombuffer(buf, np.uint8).reshape(H, W)
    finally:
        p.stdout.close()
        p.wait()


def cmp_against_bmp(mp4, bmp_tpl, step):
    maxdiff = 0
    sum_abs = 0.0
    sum_sq = 0.0
    n = 0
    worst = None
    for idx, frame in enumerate(stream_gray(mp4)):
        if idx % step:
            continue
        bmp = cv2.imread(bmp_tpl % idx, cv2.IMREAD_GRAYSCALE)
        if bmp is None:
            continue
        d = cv2.absdiff(frame, bmp)
        m = int(d.max())
        if m > maxdiff:
            maxdiff = m
            worst = idx
        sum_abs += float(d.sum())
        sum_sq += float((d.astype(np.float64) ** 2).sum())
        n += 1
    if n == 0:
        return None
    mse = sum_sq / (n * FS)
    psnr = float("inf") if mse == 0 else 10 * np.log10(255.0 * 255.0 / mse)
    return {"frames_compared": n, "maxdiff": maxdiff, "worst_frame": worst,
            "mean_abs_diff": round(sum_abs / (n * FS), 4),
            "psnr_db": None if psnr == float("inf") else round(psnr, 2),
            "lossless": maxdiff == 0}


def rederive_check(original_mp4, bmp_tpl, indices):
    """独立复算：从原始 mp4 用 OpenCV 复刻 BMP 链，和现成 BMP 比对"""
    cap = cv2.VideoCapture(original_mp4)
    results = {}
    wanted = set(indices)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in wanted:
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            roi = g[28:28 + 1024, 320:320 + 1280]
            small = cv2.resize(roi, (W, H), interpolation=cv2.INTER_AREA)
            bmp = cv2.imread(bmp_tpl % idx, cv2.IMREAD_GRAYSCALE)
            d = cv2.absdiff(small, bmp)
            results[idx] = {"maxdiff_vs_bmp": int(d.max()),
                            "mean_abs_diff": round(float(d.sum()) / d.size, 4)}
        idx += 1
        if idx > max(wanted):
            break
    cap.release()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-h264", required=True)
    ap.add_argument("--original-mp4", required=True)
    ap.add_argument("--video-640", required=True, help="推流版 yuv420p")
    ap.add_argument("--video-640-lossless", required=True, help="无损灰度版")
    ap.add_argument("--bmp-tpl", required=True, help="BMP 路径模板, 含 %%06d")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    report = {}
    print("=" * 68)
    print("A. 元数据")
    for key, path in (("src_h264", args.src_h264), ("original_mp4", args.original_mp4),
                      ("video_640_stream", args.video_640),
                      ("video_640_lossless", args.video_640_lossless)):
        try:
            info = probe(path)
        except Exception as exc:  # 裸流没有容器, ffprobe 可能报错
            info = {"error": str(exc)[:120]}
            if key == "src_h264":
                out = sh(["ffprobe", "-v", "error", "-f", "h264", "-select_streams", "v:0",
                          "-show_entries", "stream=width,height,r_frame_rate,codec_name,pix_fmt",
                          "-show_entries", "format=duration", "-of", "json", path]).stdout
                d = json.loads(out)["streams"][0]
                info = {"codec": d.get("codec_name"), "size": f"{d['width']}x{d['height']}",
                        "fps": d.get("r_frame_rate"), "pix_fmt": d.get("pix_fmt"),
                        "bytes": os.path.getsize(path)}
        report[key] = info
        print(f"  {key:22s} {info}")

    print("=" * 68)
    print("B. 原始 mp4 无损性（framemd5 全量比对 源.h264 vs 封装后.mp4）")
    a = hashlib.md5(framemd5(args.src_h264, raw=True)).hexdigest()
    b = hashlib.md5(framemd5(args.original_mp4)).hexdigest()
    report["framemd5_match"] = (a == b)
    print(f"  src.h264     framemd5-md5 = {a}")
    print(f"  original.mp4 framemd5-md5 = {b}")
    print(f"  -> {'一致（像素级无损）' if a == b else '不一致！'}")

    print("=" * 68)
    print("C. 640x512 无损灰度视频 vs BMP（全部帧逐像素）")
    r = cmp_against_bmp(args.video_640_lossless, args.bmp_tpl, step=1)
    report["lossless_vs_bmp"] = r
    print(f"  {r}")

    print("=" * 68)
    print("D. 640x512 推流版(yuv420p crf18) vs BMP（每 50 帧抽样）")
    r2 = cmp_against_bmp(args.video_640, args.bmp_tpl, step=50)
    report["stream_vs_bmp"] = r2
    print(f"  {r2}")

    print("=" * 68)
    print("E. 独立复算：原始 mp4 --(OpenCV BMP 链)--> BMP（抽样）")
    idxs = [0, 1, 100, 1000, 2955, 5905, 8855, 11809]
    r3 = rederive_check(args.original_mp4, args.bmp_tpl, idxs)
    report["rederive_vs_bmp"] = r3
    for k, v in r3.items():
        print(f"  frame {k:6d}: {v}")
    worst = max((v["maxdiff_vs_bmp"] for v in r3.values()), default=0)
    report["rederive_worst_maxdiff"] = worst
    print(f"  -> 抽样最大 maxdiff = {worst}")

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        print(f"\n[REPORT] {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
