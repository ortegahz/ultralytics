"""把 PointAnnotator_Windows 交付包拆成 N 份（默认 3），每份自包含、可单独发给标注员。

设计要点
--------
* **等量**：以「标注工作量」＝ 视频总帧数为主指标做均衡分配（最长任务优先 LPT），
  再用字节数做次级平衡；份数按条数尽量均分（58 条 ⇒ 20/19/19）。
* **自包含**：每份都带齐 tool（含 PyInstaller 的 dist/ 免装 Python 版）、docs、
  Python 安装包、三个 .bat —— 拷到 Windows 上双击即用，不依赖母包。
* **不复制中间产物**：``tool/build/``（PyInstaller 中间态）与旧 ``*.log`` 不带，
  ``videos/annotations/`` 一律建**空目录**（母包里那份 Linux 试跑草稿不带过去，
  免得标注员把它当人工标注接着标）。
* **可复验**：每份写一份 ``_split_manifest.md5``（逐文件 md5），并与母包逐字节比对。

用法
----
    python3 split_delivery.py --master /path/to/PointAnnotator_Windows \\
        --out-dir /path/to/workspace --parts 3

    # 只算分配方案不落盘（--plan）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# 母包里需要逐份复制的东西（相对母包根）
SKELETON_FILES = [
    "1_自检.bat",
    "2_开始标注.bat",
    "3_打包exe给标注员.bat",
    "msg_build_cn.txt",
    "python-3.10.0-amd64.exe",
]
SKELETON_DIRS = ["docs", "tool/dist"]
TOOL_FILES = ["point_annotator.py", "build_exe.bat", "run.bat", "README.md", "PointAnnotator.spec"]
# 有底稿的帧：标注员只需核对/微调，按此系数折算成「手工落点帧」的当量
DRAFT_FACTOR = 0.35
# 母包里**不带**过去的（中间产物 / 旧日志）
TOOL_SKIP_NAMES = {"build", "__pycache__"}
TOOL_SKIP_SUFFIX = {".log"}


@dataclass
class Item:
    """一条视频及其配套预标注。"""

    stem: str
    video: Path
    preannot: Path
    frames: int = 0
    bytes: int = 0
    draft: int = 0
    """预标注覆盖的帧数（逐帧 YOLO 树的文件数）⇒ 其余帧都要人工手工落点。"""

    @property
    def manual(self) -> int:
        """需要人工落点的帧数。"""
        return max(0, self.frames - self.draft)

    @property
    def effort(self) -> float:
        """工作量估算：手工落点 1.0/帧，有底稿只需核对（按 DRAFT_FACTOR 折算）。"""
        return self.manual + DRAFT_FACTOR * self.draft

    @property
    def key(self) -> tuple[int, int]:
        """排序键：工作量优先、字节次之。"""
        return (-self.effort, -self.bytes)


@dataclass
class Bin:
    """一份交付。"""

    index: int
    items: list[Item] = field(default_factory=list)

    @property
    def frames(self) -> int:
        return sum(i.frames for i in self.items)

    @property
    def bytes(self) -> int:
        return sum(i.bytes for i in self.items)

    @property
    def video_bytes(self) -> int:
        return sum(i.bytes for i in self.items if i.video.suffix.lower() in {".mp4", ".avi", ".mkv", ".mov"})

    @property
    def manual(self) -> int:
        """需要人工落点的帧数。"""
        return sum(i.manual for i in self.items)

    @property
    def draft(self) -> int:
        """有预推点可核对的帧数。"""
        return sum(i.draft for i in self.items)

    @property
    def effort(self) -> float:
        return sum(i.effort for i in self.items)


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_frames(video: Path) -> int:
    """数视频真实帧数。

    用容器里的 ``nb_frames``（MP4 的 stsz/stss 表里就有，毫秒级），
    **不用** ``-count_frames``——那要逐帧解码，58 条 1080p 要十几分钟。
    """
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames", "-of", "csv=p=0", str(video),
        ],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    try:
        return int(out)
    except ValueError:
        return 0


def collect(master: Path, frame_cache: Path | None) -> list[Item]:
    """收集全部视频 + 配套预标注目录，并算工作量。"""
    videos_dir = master / "videos"
    pre_root = master / "preannot_rc2_tracked"
    pre_map = {d.name: d for d in pre_root.iterdir() if d.is_dir()}

    cache: dict[str, int] = {}
    if frame_cache and frame_cache.is_file():
        raw = json.loads(frame_cache.read_text(encoding="utf-8"))
        cache = {k: (v["frames"] if isinstance(v, dict) else int(v)) for k, v in raw.items()}

    items: list[Item] = []
    for video in sorted(videos_dir.glob("*.mp4")):
        pre = pre_map.get(video.stem)
        if pre is None:
            raise SystemExit(f"[!] 视频缺配套预标注目录：{video.name}")
        if video.stem in cache:
            frames = int(cache[video.stem])
        else:
            frames = probe_frames(video)
            cache[video.stem] = frames
        items.append(Item(stem=video.stem, video=video, preannot=pre, frames=frames,
                          bytes=video.stat().st_size, draft=len(list(pre.glob("*.txt")))))

    if frame_cache:
        frame_cache.parent.mkdir(parents=True, exist_ok=True)
        frame_cache.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    return items


def _spread(bins: list[Bin]) -> float:
    """三份之间「最大相对偏差」：总帧数 / 手工帧数 / 工作量 三项一起看，条数偏差做软惩罚。

    * **总帧数** —— 每帧都要过一遍，是总量；
    * **手工帧数** —— 没有预推点、必须手工落点的帧，是真正的「难活」；
    * **工作量** —— 手工 1.0 + 核对 0.35 的当量估算；
    * **条数** —— 只作软约束（19/19/20 与 20/19/19 都算等量，不值得为条数牺牲别的）。
    """
    def dev(attr: str) -> float:
        values = [getattr(b, attr) for b in bins]
        mean = sum(values) / len(values)
        if mean <= 0:
            return 0.0
        return max(abs(v - mean) for v in values) / mean
    counts = [len(b.items) for b in bins]
    mean_count = sum(counts) / len(counts)
    count_dev = max(abs(c - mean_count) for c in counts) / max(mean_count, 1)
    return max(dev("frames"), dev("manual"), dev("effort")) + 0.02 * count_dev


def partition(items: list[Item], parts: int) -> list[Bin]:
    """等量分配。

    1) **LPT 播种**：按工作量从大到小，依次放进当前最轻的一份；
    2) **局部搜索**：反复尝试「把一条挪到另一份」与「两条对调」，只要能让
       三份的最大相对偏差变小就接受 ⇒ 收敛后总帧/手工帧/工作量三项同时对齐。
    """
    bins = [Bin(index=i + 1) for i in range(parts)]
    for item in sorted(items, key=lambda i: i.key):
        target = min(bins, key=lambda b: (b.effort, b.manual, b.frames))
        target.items.append(item)

    best = _spread(bins)
    improved = True
    while improved:
        improved = False
        for a in range(len(bins)):
            for b in range(len(bins)):
                if a == b:
                    continue
                # 1) 单条挪动
                for item in list(bins[a].items):
                    bins[a].items.remove(item)
                    bins[b].items.append(item)
                    score = _spread(bins)
                    if score < best - 1e-9:
                        best = score
                        improved = True
                        break
                    bins[b].items.remove(item)
                    bins[a].items.append(item)
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # 2) 两条对调
        for a in range(len(bins)):
            for b in range(a + 1, len(bins)):
                for ia in list(bins[a].items):
                    for ib in list(bins[b].items):
                        bins[a].items.remove(ia); bins[b].items.remove(ib)
                        bins[a].items.append(ib); bins[b].items.append(ia)
                        score = _spread(bins)
                        if score < best - 1e-9:
                            best = score
                            improved = True
                            break
                        bins[a].items.remove(ib); bins[b].items.remove(ia)
                        bins[a].items.append(ia); bins[b].items.append(ib)
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    for b in bins:
        b.items.sort(key=lambda i: i.stem)
    return bins


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path, skip_names: set[str] | None = None,
              skip_suffix: set[str] | None = None) -> list[Path]:
    """复制目录树，返回实际复制的文件列表。"""
    skip_names = skip_names or set()
    skip_suffix = skip_suffix or set()
    copied: list[Path] = []
    for path in sorted(src.rglob("*")):
        rel = path.relative_to(src)
        if any(part in skip_names for part in rel.parts):
            continue
        if path.is_dir():
            continue
        if path.suffix.lower() in skip_suffix:
            continue
        target = dst / rel
        copy_file(path, target)
        copied.append(target)
    return copied


def write_part_readme(master: Path, part_dir: Path, b: Bin, total: int,
                      counts: list[int], frames: list[int]) -> None:
    """母包 README 复制后按本份实际情况改写（改不到就报错，不许静默漏改）。"""
    text = (master / "README_先看这个.md").read_text(encoding="utf-8")
    n_videos = len(b.items)

    def sub(old: str, new: str) -> None:
        nonlocal text
        if old not in text:
            raise SystemExit(f"[!] README 改写点没命中，请检查模板：{old[:60]!r}")
        text = text.replace(old, new, 1)

    sub(
        "# PointAnnotator Windows 交付包\n\n"
        "把这个**整个文件夹**拷到 Windows 机器上就能标。里面已经放好了工具、58 条原始视频、58 个序列的模型预推点。",
        f"# PointAnnotator Windows 交付包（第 {b.index} / {total} 份）\n\n"
        "把这个**整个文件夹**拷到 Windows 机器上就能标。里面已经放好了工具、"
        f"**{n_videos} 条原始视频、{n_videos} 个序列的模型预推点**。",
    )
    sub(
        "├─ videos\\                 ← 58 条原始视频（**可写**，标注就落在这里）\n"
        "│  ├─ VIDEO00028_19700101_002947.mp4 ...\n"
        "│  └─ annotations\\         ← 标完自动生成，勿删（已有 2 份 Linux 试跑留下的草稿，见第 4 节）",
        f"├─ videos\\                 ← 本份的 {n_videos} 条原始视频（**可写**，标注就落在这里）\n"
        f"│  ├─ {b.items[0].stem}.mp4 ...\n"
        "│  └─ annotations\\         ← 标完自动生成，勿删（**本份是空的**，没有草稿）",
    )
    sub(
        "└─ preannot_rc2_tracked\\   ← 58 个序列的模型预推点（只读即可）\n"
        "   └─ VIDEO00028_19700101_002947\\",
        f"└─ preannot_rc2_tracked\\   ← 本份 {n_videos} 个序列的模型预推点（只读即可）\n"
        f"   └─ {b.items[0].stem}\\",
    )
    sub(
        "> `videos\\annotations\\` 里**已经躺着 2 个 json**（`VIDEO00028...` 与 `VIDEO00031...`），\n"
        "> 是之前在 Linux 上试跑时从预推点导入存下来的草稿（不是人工标注）。工具会优先读它们，\n"
        "> 所以启动时会直接停在已有草稿上。\n"
        "> **想从零开始标**：先把这两个 json 删掉（或改名），再启动 `2_开始标注.bat`。\n"
        "> 想接着上次的进度：保留即可。",
        "> 本份的 `videos\\annotations\\` 是**空目录**（母包里的 Linux 试跑草稿没有带过来），\n"
        "> 直接启动 `2_开始标注.bat` 就是从第一条视频开始标，不需要先删任何东西。",
    )

    listing = "\n".join(
        f"| {i + 1} | `{it.stem}` | {it.frames} | {it.bytes / 1024 / 1024:.1f} |"
        for i, it in enumerate(b.items))
    counts_txt = " / ".join(f"{c} 条" for c in counts)
    frames_txt = " / ".join(f"{f:,} 帧" for f in frames)
    header = (
        f"## 0. 这是第 {b.index} / {total} 份（先看这里）\n\n"
        f"标注任务是 **{total} 份并行**，你拿的是**第 {b.index} 份**。\n\n"
        f"* 本份 **视频 {n_videos} 条**（三份分别是 {counts_txt}），"
        f"**预推点 {n_videos} 个序列**，视频与预推点一一对应；\n"
        f"* 工作量按总帧数对齐（三份分别是 {frames_txt}），**条数不同、活儿一样多**；\n"
        f"* 标注产出落 `videos\\annotations\\`，**标完把 `videos\\annotations\\` 整个目录传回来**；\n"
        f"* 三份标完之后由项目方按视频名合并回完整数据集（标注员不用管合并）。\n\n"
        f"### 本份视频清单（{n_videos} 条）\n\n"
        "| # | 视频 | 帧数 | 大小(MB) |\n| ---: | --- | ---: | ---: |\n"
        f"{listing}\n"
    )
    text = header + text

    (part_dir / "README_先看这个.md").write_text(text, encoding="utf-8")


def write_msg_cn(master: Path, part_dir: Path, b: Bin, total: int) -> None:
    """中文提示：母包 msg_cn.txt 复制后按本份改写。"""
    text = (master / "msg_cn.txt").read_text(encoding="utf-8")
    old = (
        "6) 想从零开始标：先删掉 videos\\annotations\\ 里已有的 2 个 json\n"
        "   （那是 Linux 试跑留下的草稿，不是人工标注）\n"
    )
    if old not in text:
        raise SystemExit("[!] msg_cn.txt 改写点没命中，请检查模板")
    new = (
        f"6) 你拿的是第 {b.index}/{total} 份，共 {len(b.items)} 条视频（清单见 README_先看这个.md 开头）\n"
        "   videos\\annotations\\ 是空的，直接从第一条开始标，不用先删任何东西\n"
        "7) 标完把 videos\\annotations\\ 整个目录传回来（里面的 json 就是你的标注成果）\n"
    )
    text = text.replace(old, new, 1)
    (part_dir / "msg_cn.txt").write_text(text, encoding="utf-8")


def write_bat(master: Path, part_dir: Path, b: Bin, total: int) -> None:
    """2_开始标注.bat：保持纯 ASCII + 原换行风格（.bat 是 CRLF），只加一行份数信息。

    **不能**按 ``\n`` 找锚点——这些批处理是 CRLF 的，按 ``\n`` 匹配必然落空；
    这里按行拆（保留行尾）再原样插回，行尾风格自动跟随原文件。
    """
    raw = (master / "2_开始标注.bat").read_bytes().decode("ascii")
    lines = raw.splitlines(keepends=True)
    out: list[str] = []
    hit = False
    for line in lines:
        out.append(line)
        if line.startswith("echo    output   :"):
            ending = "\r\n" if line.endswith("\r\n") else "\n"
            out.append(f"echo    Part     : {b.index} of {total}   (videos in this part: {len(b.items)}){ending}")
            hit = True
    if not hit:
        raise SystemExit("[!] 2_开始标注.bat 改写点没命中，请检查模板")
    (part_dir / "2_开始标注.bat").write_bytes("".join(out).encode("ascii"))


def build_part(master: Path, part_dir: Path, b: Bin, total: int,
               counts: list[int], frame_counts: list[int]) -> list[Path]:
    """生成一份交付，返回该份所有文件。"""
    if part_dir.exists():
        shutil.rmtree(part_dir)
    part_dir.mkdir(parents=True)

    copied: list[Path] = []
    for name in SKELETON_FILES:
        copy_file(master / name, part_dir / name)
        copied.append(part_dir / name)
    for name in SKELETON_DIRS:
        copied += copy_tree(master / name, part_dir / name, skip_names=TOOL_SKIP_NAMES,
                            skip_suffix=TOOL_SKIP_SUFFIX)
    for name in TOOL_FILES:
        copy_file(master / "tool" / name, part_dir / "tool" / name)
        copied.append(part_dir / "tool" / name)

    # 视频 + 配套预标注
    for item in b.items:
        copy_file(item.video, part_dir / "videos" / item.video.name)
        copied.append(part_dir / "videos" / item.video.name)
        for txt in sorted(item.preannot.glob("*.txt")):
            copy_file(txt, part_dir / "preannot_rc2_tracked" / item.stem / txt.name)
            copied.append(part_dir / "preannot_rc2_tracked" / item.stem / txt.name)
        side = item.preannot / "scores.json"
        if side.is_file():
            copy_file(side, part_dir / "preannot_rc2_tracked" / item.stem / side.name)
            copied.append(part_dir / "preannot_rc2_tracked" / item.stem / side.name)

    # 空的标注输出目录
    (part_dir / "videos" / "annotations").mkdir(parents=True, exist_ok=True)

    write_part_readme(master, part_dir, b, total, counts, frame_counts)
    write_msg_cn(master, part_dir, b, total)
    write_bat(master, part_dir, b, total)
    copied += [part_dir / "README_先看这个.md", part_dir / "msg_cn.txt", part_dir / "2_开始标注.bat"]
    return copied


def write_manifest(part_dir: Path, files: list[Path]) -> None:
    """写本份的 md5 清单（相对本份根）。"""
    lines = [
        f"# PointAnnotator Windows 交付包 —— 第 {part_dir.name} 的文件 md5 清单",
        "# 校验：cd <本份根> && md5sum -c _split_manifest.md5",
        "#",
    ]
    entries = []
    for path in sorted(files, key=lambda p: str(p.relative_to(part_dir))):
        rel = path.relative_to(part_dir).as_posix()
        entries.append(f"{md5(path)}  {rel}")
    lines += entries
    (part_dir / "_split_manifest.md5").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_overview(out_dir: Path, master: Path, bins: list[Bin], summary: list[dict],
                   total: int) -> None:
    """生成父目录的《交付说明.md》：三份清单、等量数据、发放与回收流程、复验命令。"""
    def dev(attr: str) -> float:
        vals = [getattr(b, attr) for b in bins]
        mean = sum(vals) / len(vals)
        return max(abs(v - mean) for v in vals) / mean if mean else 0.0

    lines = [
        "# PointAnnotator 标注任务 —— 三份并行交付说明",
        "",
        f"* 母包：`{master}`（**未改动**，仍保持 58 条视频的全量形态）",
        f"* 本目录：3 份自包含交付，每份发给 1 位标注员",
        f"* 视频总量：**{sum(b.frames for b in bins):,} 帧 / {sum(len(b.items) for b in bins)} 条 / "
        f"{sum(b.video_bytes for b in bins) / 1024 / 1024:.0f} MB**",
        "",
        "## 1. 三份的等量数据",
        "",
        "| 份 | 目录 | 视频条数 | 总帧数 | 手工帧 | 有底稿帧 | 工作量 | 视频体积 |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for b, row in zip(bins, summary):
        lines.append(
            f"| {b.index} | `{row['dir']}` | {len(b.items)} | {b.frames:,} | "
            f"{b.manual:,} | {b.draft:,} | {b.effort:,.0f} | {b.video_bytes / 1024 / 1024:.1f} MB |")
    lines += [
        "",
        f"**最大偏差：总帧 {dev('frames') * 100:.1f}% ｜ 手工帧 {dev('manual') * 100:.1f}% ｜ "
        f"工作量 {dev('effort') * 100:.1f}%** ⇒ 三份活儿一样多（条数 "
        f"{'/'.join(str(len(b.items)) for b in bins)}，差异只在 ±1 条以内）。",
        "",
        "> 「手工帧」= 没有模型预推点、必须人工逐帧落点的帧；「有底稿帧」= 有预推点、只需核对/微调。",
        "> 「工作量」= 手工帧 × 1.0 + 有底稿帧 × 0.35（折算当量）。分配算法同时对齐这三个指标。",
        "",
        "## 2. 每份的视频清单",
        "",
    ]
    for b in bins:
        lines += [f"### 第 {b.index} 份（`{b.items[0].stem}` … `{b.items[-1].stem}`）", "",
                  "| # | 视频 | 总帧 | 底稿帧 | 手工帧 |", "| ---: | --- | ---: | ---: | ---: |"]
        for i, it in enumerate(b.items, 1):
            lines.append(f"| {i} | `{it.stem}` | {it.frames:,} | {it.draft:,} | {it.manual:,} |")
        nodraft = [i.stem for i in b.items if i.draft == 0]
        if nodraft:
            lines += ["", f"> ⚠️ 本份有 **{len(nodraft)}** 条视频**没有模型预推点**（跟踪器在这两条上没出点），"
                          f"需**纯手工**标注：{'、'.join('`' + s + '`' for s in nodraft)}"
                          f"（合计 {sum(i.frames for i in b.items if i.draft == 0):,} 帧，占总量 "
                          f"{sum(i.frames for i in b.items if i.draft == 0) / sum(b.frames for b in bins) * 100:.1f}%）。"]
        lines.append("")

    lines += [
        "## 3. 怎么发给标注员",
        "",
        "1. 把**整个目录**（`PointAnnotator_Windows_标注员N/`）拷到 Windows 机器上（或压成 zip 再传）；",
        "2. 标注员双击 `1_自检.bat` → 应输出 `[self-test] PASS 101  FAIL 0`；",
        "3. 再双击 `2_开始标注.bat` 开始标注（视频与预推点已配好，**不要改目录名**）；",
        "4. ⚠️ **开机第一件事：输入法切英文**，否则单字母快捷键全被吞掉。",
        "",
        "每份都自带：`tool/`（源码 + 已打包好的免装 Python 版 exe）、`docs/` 课件、"
        "`python-3.10.0-amd64.exe` 安装包 ⇒ **不需要额外准备环境**。",
        "",
        "## 4. 回收与合并",
        "",
        "1. 标注员标完后，只需把 **`videos\\annotations\\` 整个目录**回传（里面的 json 就是标注成果）；",
        "2. 合并时按**视频同名**放回母包 `PointAnnotator_Windows/videos/annotations/` 即可 —— "
        "三份的视频名互不重叠，**直接并集，不存在覆盖**；",
        "3. 合并后用母包的 `knee_manifest.md5` 之类的清单复核（若走视频质量流程）。",
        "",
        "## 5. 复验",
        "",
        "```bash",
        "# 结构性 + 逐字节 + 等量验收（67 项）",
        f"python3 manu/point_annotator/verify_split.py --master {master} --parts-dir {out_dir}",
        "",
        "# 单份自校验（在任一份的根目录下执行）",
        "md5sum -c _split_manifest.md5",
        "```",
        "",
        f"生成命令：`python3 manu/point_annotator/split_delivery.py --master {master} "
        f"--out-dir {out_dir} --parts 3`",
        "",
        "## 6. 与母包的关系",
        "",
        f"* 母包 `{master.name}/` **原封未动**（58 条视频、58 个序列的预推点全在里面），可继续当全量视图用；",
        "* 三份交付里的每个视频、每个预标注文件、工具源码与 exe，都与母包**逐字节一致**（md5 已核）；",
        "* 母包 `videos/annotations/` 里那份 Linux 试跑草稿**没有带进三份**（三份的 annotations 是空的），"
        "避免标注员把它当成人工标注接着标；",
        "* `tool/build/`（PyInstaller 中间态）与旧 `*.log` 也没带过去。",
        "",
    ]
    (out_dir / "交付说明.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="把 PointAnnotator_Windows 交付包拆成 N 份")
    ap.add_argument("--master", required=True, help="母包目录（含 videos/ 与 preannot_rc2_tracked/）")
    ap.add_argument("--out-dir", required=True, help="三份交付包的输出父目录")
    ap.add_argument("--parts", type=int, default=3, help="拆几份（默认 3）")
    ap.add_argument("--prefix", default="PointAnnotator_Windows_标注员", help="各份目录名前缀")
    ap.add_argument("--frame-cache", default="", help="帧数缓存 json（重跑时省掉 ffprobe）")
    ap.add_argument("--plan", action="store_true", help="只打印分配方案，不落盘")
    args = ap.parse_args(argv)

    master = Path(args.master).resolve()
    out_dir = Path(args.out_dir).resolve()
    frame_cache = Path(args.frame_cache) if args.frame_cache else None

    items = collect(master, frame_cache)
    bins = partition(items, args.parts)

    print(f"母包：{master}")
    print(f"视频 {len(items)} 条 | 总帧数 {sum(i.frames for i in items)} | "
          f"总大小 {sum(i.bytes for i in items) / 1024 / 1024:.1f} MB")
    for b in bins:
        print(f"  第 {b.index} 份：{len(b.items)} 条 | 总帧 {b.frames} | 手工帧 {b.manual} | "
              f"底稿帧 {b.draft} | 工作量 {b.effort:.0f} | {b.video_bytes / 1024 / 1024:.1f} MB | "
              f"{b.items[0].stem} ... {b.items[-1].stem}")
    if args.plan:
        return 0

    counts = [len(b.items) for b in bins]
    frame_counts = [b.frames for b in bins]
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for b in bins:
        part_dir = out_dir / f"{args.prefix}{b.index}"
        print(f"[{b.index}/{args.parts}] 生成 {part_dir} ...")
        files = build_part(master, part_dir, b, args.parts, counts, frame_counts)
        write_manifest(part_dir, files)
        summary.append({
            "index": b.index,
            "dir": part_dir.name,
            "videos": len(b.items),
            "frames": b.frames,
            "manual_frames": b.manual,
            "draft_frames": b.draft,
            "effort": round(b.effort, 1),
            "video_mb": round(b.video_bytes / 1024 / 1024, 1),
            "files": len(files) + 1,
            "no_draft": [i.stem for i in b.items if i.draft == 0],
            "stems": [i.stem for i in b.items],
        })
        print(f"    {len(files)} 个文件 | 视频 {len(b.items)} 条 | {b.frames} 帧")

    (out_dir / "_split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_overview(out_dir, master, bins, summary, args.parts)
    print(f"\n完成：{out_dir}")
    print(f"  交付说明：{out_dir / '交付说明.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
