"""验收 PointAnnotator 的三份交付：结构、配对、逐字节一致、等量、无重复无遗漏。

用法
----
    python3 verify_split.py --master /path/to/PointAnnotator_Windows \\
        --parts-dir /path/to/PointAnnotator_Windows_三份交付

退出码 0 = 全部通过；1 = 有 FAIL。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

DRAFT_FACTOR = 0.35  # 与 split_delivery.py 保持一致


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frames_of(video: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=True).stdout.strip()
    try:
        return int(out)
    except ValueError:
        return 0


class Checker:
    def __init__(self) -> None:
        self.fails: list[str] = []
        self.passes = 0

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        if ok:
            self.passes += 1
            print(f"  [PASS] {label}" + (f"  ({detail})" if detail else ""))
        else:
            self.fails.append(label)
            print(f"  [FAIL] {label}" + (f"  ({detail})" if detail else ""))
        return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="验收 PointAnnotator 三份交付")
    ap.add_argument("--master", required=True)
    ap.add_argument("--parts-dir", required=True)
    ap.add_argument("--prefix", default="PointAnnotator_Windows_标注员")
    ap.add_argument("--skip-hash", action="store_true", help="跳过逐文件 md5 比对（快，只查结构）")
    args = ap.parse_args(argv)

    master = Path(args.master).resolve()
    parts_dir = Path(args.parts_dir).resolve()
    ck = Checker()

    parts = sorted(p for p in parts_dir.iterdir() if p.is_dir() and p.name.startswith(args.prefix))
    print(f"母包：{master}\n交付目录：{parts_dir}\n")
    ck.check(len(parts) == 3, "恰好 3 份", f"实际 {len(parts)} 份")

    master_vids = {v.stem: v for v in (master / "videos").glob("*.mp4")}
    owner: dict[str, list[str]] = {}
    stats = []

    for part in parts:
        print(f"\n=== {part.name} ===")
        vids = {v.stem: v for v in (part / "videos").glob("*.mp4")}
        pres = {d.name: d for d in (part / "preannot_rc2_tracked").iterdir() if d.is_dir()}

        ck.check(bool(vids), "有视频")
        ck.check(set(vids) == set(pres), "视频与预标注一一对应",
                 f"视频 {len(vids)} / 预标注 {len(pres)}")

        # 骨架文件
        for name in ["1_自检.bat", "2_开始标注.bat", "3_打包exe给标注员.bat",
                     "msg_cn.txt", "msg_build_cn.txt", "README_先看这个.md",
                     "python-3.10.0-amd64.exe"]:
            ck.check((part / name).is_file(), f"骨架文件在位：{name}")
        for name in ["tool/point_annotator.py", "tool/build_exe.bat", "tool/run.bat",
                     "tool/README.md", "tool/dist/PointAnnotator/PointAnnotator.exe",
                     "docs/标注培训PPT_分页内容.md"]:
            ck.check((part / name).is_file(), f"工具/课件在位：{name}")

        # 干净度
        ck.check((part / "videos" / "annotations").is_dir(), "annotations 目录存在")
        ck.check(not list((part / "videos" / "annotations").iterdir()), "annotations 是空的")
        junk = [str(x.relative_to(part)) for x in part.rglob("*.log")]
        junk += [str(x.relative_to(part)) for x in part.rglob("build") if x.is_dir()]
        ck.check(not junk, "没有旧日志 / 没有 PyInstaller build 中间态", str(junk[:3]))

        # 逐字节
        if not args.skip_hash:
            bad = []
            for stem, v in vids.items():
                if md5(v) != md5(master_vids.get(stem, v)):
                    bad.append(stem)
                mdir = master / "preannot_rc2_tracked" / stem
                for f in (part / "preannot_rc2_tracked" / stem).iterdir():
                    mf = mdir / f.name
                    if not mf.is_file() or md5(f) != md5(mf):
                        bad.append(f"{stem}/{f.name}")
            ck.check(not bad, "视频与预标注逐字节等于母包", f"{len(bad)} 处不一致")
            ck.check(md5(part / "tool" / "point_annotator.py")
                     == md5(master / "tool" / "point_annotator.py"), "工具源码与母包一致")

        # 清单自校验
        manifest = part / "_split_manifest.md5"
        if manifest.is_file():
            proc = subprocess.run(["md5sum", "-c", "_split_manifest.md5"], cwd=part,
                                  capture_output=True, text=True)
            ok = proc.returncode == 0
            ok_count = proc.stdout.count(": OK")
            ck.check(ok, "本份 _split_manifest.md5 全量自校验通过", f"{ok_count} 个文件")

        # 统计
        total = manual = draft = 0
        size = 0
        for stem, v in vids.items():
            fr = frames_of(v)
            nt = len(list(pres[stem].glob("*.txt")))
            total += fr
            draft += nt
            manual += fr - nt
            size += v.stat().st_size
            owner.setdefault(stem, []).append(part.name)
        stats.append({"name": part.name, "n": len(vids), "frames": total,
                      "manual": manual, "draft": draft,
                      "effort": manual + DRAFT_FACTOR * draft, "mb": size / 1024 / 1024})

    # 全局
    print("\n=== 全局 ===")
    dup = {s: p for s, p in owner.items() if len(p) > 1}
    missing = set(master_vids) - set(owner)
    ck.check(not dup, "三份互不重叠", f"重复 {len(dup)} 条")
    ck.check(not missing, "母包视频全部覆盖", f"覆盖 {len(owner)}/{len(master_vids)}，缺 {len(missing)}")

    print("\n=== 等量 ===")
    for key, label in [("n", "视频条数"), ("frames", "总帧数"), ("manual", "手工帧数"),
                       ("effort", "工作量"), ("mb", "视频体积")]:
        vals = [s[key] for s in stats]
        mean = sum(vals) / len(vals)
        dev = max(abs(v - mean) for v in vals) / mean if mean else 0
        print(f"  {label:8s}: " + " / ".join(f"{v:,.0f}" if key != "mb" else f"{v:.1f}MB" for v in vals)
              + f"   最大偏差 {dev * 100:.1f}%")
    worst = max(
        max(abs(s[key] - sum(x[key] for x in stats) / len(stats))
            / (sum(x[key] for x in stats) / len(stats))
            for s in stats)
        for key in ("frames", "manual", "effort"))
    ck.check(worst <= 0.03, "三份工作量偏差 ≤ 3%", f"实际 {worst * 100:.1f}%")

    print(f"\n{'=' * 56}")
    if ck.fails:
        print(f"结论：**FAIL** —— {len(ck.fails)} 项不通过 / 共 {ck.passes + len(ck.fails)} 项")
        for f in ck.fails:
            print(f"  - {f}")
        return 1
    print(f"结论：**全绿** —— {ck.passes} 项全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
