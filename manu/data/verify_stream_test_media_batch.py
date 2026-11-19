#!/usr/bin/env python3
"""批量校验嵌入式组推流测试素材（龙泉山 pass 序列集）。

对每个序列复用 verify_stream_test_media.py 的判定逻辑，逐序列产出 JSON，
再汇总成一份 index（verification_report.json）。

校验项（与单序列版一致）：
  A. 元数据（分辨率 / 帧率 / 帧数 / 像素格式）
  B. 原始 mp4 是否像素级无损于源 .h264（framemd5 全量比对）
  C. 640x512 无损灰度版 vs 参照帧（全部帧逐像素）
  D. 640x512 推流版 vs 参照帧（每 50 帧抽样）
  E. 独立复算：原始 mp4 --(OpenCV BMP 链)--> vs 参照帧

参照帧（ref）解析顺序：
  1) --bmp-root/<SEQ> 存在 ⇒ 用真实 8bit BMP（与板端输入同一份像素）
  2) 否则 --ref-root/<SEQ> ⇒ 用 BMP 链产出的 JPG
     （JPG 与 BMP 逐像素相同已实测：imread(jpg,GRAYSCALE) -> imwrite(bmp) -> imread maxdiff=0）

判定（🔴 不可写死：像素级不合格必须让退出码非零）：
  · framemd5_match 必须为 True                      ⇒ 否则 FAIL
  · 无损灰度版 maxdiff <= --max-lossless-diff        ⇒ 否则 FAIL
  · 独立复算 worst maxdiff <= --max-rederive-diff     ⇒ 否则 FAIL
  · 推流版 maxdiff > --max-stream-diff               ⇒ 只告警（CRF 压缩噪声随内容变化）

--classify-only：不重跑像素比对，直接读已存在的 per-seq JSON 重新判定。
"""
import argparse
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("vm", HERE / "verify_stream_test_media.py")
vm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vm)


def resolve_ref(seq, bmp_root, ref_root):
    """返回 (模板, ref_kind)；模板含 %06d"""
    if bmp_root:
        d = Path(bmp_root) / seq
        if d.is_dir():
            return str(d / "frame_%06d.bmp"), "bmp"
    if ref_root:
        d = Path(ref_root) / seq
        if d.is_dir():
            return str(d / "frame_%06d.jpg"), "jpg"
    return None, None


def judge(rep, args):
    """按阈值判定 ⇒ (status, reasons, warnings)"""
    reasons, warnings = [], []
    if rep.get("pixel_hashes_match", rep.get("framemd5_match")) is not True:
        reasons.append("原始 mp4 与源 .h264 的像素 hash 序列不一致（像素级有损）")
    loss = rep.get("lossless_vs_bmp") or {}
    stream = rep.get("stream_vs_bmp") or {}
    if loss.get("maxdiff") is None or loss.get("maxdiff") > args.max_lossless_diff:
        reasons.append(f"无损灰度版 maxdiff={loss.get('maxdiff')} > 阈值 {args.max_lossless_diff}")
    worst = rep.get("rederive_worst_maxdiff")
    if worst is None or worst > args.max_rederive_diff:
        reasons.append(f"独立复算 worst maxdiff={worst} > 阈值 {args.max_rederive_diff}")
    if stream.get("maxdiff") is not None and stream.get("maxdiff") > args.max_stream_diff:
        warnings.append(f"推流版 maxdiff={stream.get('maxdiff')} > 告警阈值 {args.max_stream_diff}")
    return ("FAIL" if reasons else "OK"), reasons, warnings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-list", required=True)
    ap.add_argument("--out", required=True, help="素材目录")
    ap.add_argument("--ref-root", default="", help="参照帧根（JPG）")
    ap.add_argument("--bmp-root", default="", help="真实 BMP 根（优先）")
    ap.add_argument("--src-root", default="/mnt/data/xxxu/龙泉山/ir")
    ap.add_argument("--report-dir", default="")
    ap.add_argument("--index-json", default="")
    ap.add_argument("--classify-only", action="store_true",
                    help="不重跑像素比对，直接读已存在的 per-seq JSON 重新判定")
    ap.add_argument("--max-lossless-diff", type=int, default=1,
                    help="无损灰度版 vs 参照帧 允许的最大逐像素差（参考交付实测 1）")
    ap.add_argument("--max-stream-diff", type=int, default=64,
                    help="推流版 vs 参照帧 超过此值只告警，不阻断")
    ap.add_argument("--max-rederive-diff", type=int, default=12,
                    help="独立复算 vs 参照帧 允许的最大差（JPEG q95 链实测 7~9）")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report_dir = Path(args.report_dir) if args.report_dir else out / "verification"
    report_dir.mkdir(parents=True, exist_ok=True)
    index_json = Path(args.index_json) if args.index_json else out / "verification_report.json"

    seqs = [ln.strip() for ln in open(args.seq_list, encoding="utf-8")
            if ln.strip() and not ln.strip().startswith("#")]
    print(f"[BATCH-VERIFY] 序列数={len(seqs)}  report_dir={report_dir}  "
          f"classify_only={args.classify_only}", flush=True)

    index = {"sequences": [], "ok": 0, "failed": 0, "total": len(seqs),
             "thresholds": {"max_lossless_diff": args.max_lossless_diff,
                            "max_stream_diff": args.max_stream_diff,
                            "max_rederive_diff": args.max_rederive_diff}}
    for seq in seqs:
        entry = {"sequence": seq}
        rep = None
        try:
            ref_tpl, ref_kind = resolve_ref(seq, args.bmp_root, args.ref_root)
            entry["ref_kind"] = ref_kind
            entry["ref_template"] = ref_tpl
            if ref_tpl is None:
                raise FileNotFoundError(
                    f"找不到参照帧: bmp_root={args.bmp_root} ref_root={args.ref_root}")

            src_h264 = str(Path(args.src_root) / f"{seq}.h264")
            orig = str(out / f"{seq}_original_1920x1080.mp4")
            v640 = str(out / f"{seq}_640x512.mp4")
            vloss = str(out / f"{seq}_640x512_lossless_gray.mp4")
            rep_path = report_dir / f"{seq}.json"

            if args.classify_only:
                if not rep_path.is_file():
                    raise FileNotFoundError(f"缺 {rep_path}")
                rep = json.load(open(rep_path, encoding="utf-8"))
            else:
                missing = [p for p in (src_h264, orig, v640, vloss)
                           if not os.path.isfile(p)]
                if missing:
                    raise FileNotFoundError(f"缺文件 {missing}")
                argv = ["--src-h264", src_h264,
                        "--original-mp4", orig,
                        "--video-640", v640,
                        "--video-640-lossless", vloss,
                        "--bmp-tpl", ref_tpl,
                        "--out-json", str(rep_path)]
                saved = sys.argv
                sys.argv = ["verify_stream_test_media.py"] + argv
                try:
                    vm.main()
                finally:
                    sys.argv = saved
                rep = json.load(open(rep_path, encoding="utf-8"))

            status, reasons, warnings = judge(rep, args)
            rep["sequence"] = seq
            rep["ref_kind"] = ref_kind
            rep["ref_template"] = ref_tpl
            rep["status"] = status
            rep["fail_reasons"] = reasons
            rep["warnings"] = warnings
            json.dump(rep, open(rep_path, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)

            entry["status"] = status
            entry["fail_reasons"] = reasons
            entry["warnings"] = warnings
            entry["framemd5_match"] = rep.get("framemd5_match")
            entry["lossless_vs_ref"] = rep.get("lossless_vs_bmp")
            entry["stream_vs_ref"] = rep.get("stream_vs_bmp")
            entry["rederive_worst_maxdiff"] = rep.get("rederive_worst_maxdiff")
            if status == "FAIL":
                index["failed"] += 1
                print(f"[FAIL] {seq}  ref={ref_kind}  reasons={reasons}", flush=True)
            else:
                index["ok"] += 1
                print(f"[OK] {seq}  ref={ref_kind}  framemd5_match={rep.get('framemd5_match')} "
                      f"maxdiff(lossless)={(rep.get('lossless_vs_bmp') or {}).get('maxdiff')} "
                      f"maxdiff(stream)={(rep.get('stream_vs_bmp') or {}).get('maxdiff')} "
                      f"rederive_worst={rep.get('rederive_worst_maxdiff')}"
                      + (f"  WARN={warnings}" if warnings else ""), flush=True)
        except Exception as exc:
            traceback.print_exc()
            entry["status"] = "FAIL_EXCEPTION"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            index["failed"] += 1
            print(f"[FAIL] {seq}: {exc}", flush=True)
        finally:
            index["sequences"].append(entry)

    json.dump(index, open(index_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[BATCH-VERIFY] OK={index['ok']} FAIL={index['failed']} / {len(seqs)}  -> {index_json}",
          flush=True)
    return 0 if index["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
