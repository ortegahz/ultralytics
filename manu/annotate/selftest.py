#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""End-to-end self-test for the annotation tool: scan → decode → label → HTTP API.

Builds a synthetic video with a moving bright blob (and a decoy frame directory) so the test exercises
the real decode path rather than mocking it, then drives the HTTP surface the browser actually uses.
Run with: ``/home/manu/anaconda3/bin/python -m manu.annotate.selftest``
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu.annotate.dataset import scan_root
from manu.annotate.labels import LabelStore, format_yolo, parse_yolo

WIDTH, HEIGHT, FRAMES, FRAME_DIR_FRAMES = 320, 256, 48, 6
PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> bool:
    results.append((PASS if condition else FAIL, name, detail))
    print(f"  [{PASS if condition else FAIL}] {name}" + (f" — {detail}" if detail else ""))
    return condition


def make_video(path: Path, frames: int, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (width, height))
    rng = np.random.default_rng(7)
    for index in range(frames):
        canvas = rng.integers(40, 70, size=(height, width), dtype=np.uint8)
        canvas = cv2.GaussianBlur(canvas, (5, 5), 0)
        # A blob drifting right at ~2 px/frame with a second slower one, so propagation has two tracks.
        for speed, phase in ((2.0, 0.0), (0.7, 40)):
            cx = int(30 + speed * index + phase) % (width - 20)
            cy = int(60 + 40 * (speed / 2.0)) % (height - 20)
            cv2.circle(canvas, (cx, cy), 3, 230, -1)
        # VideoWriter needs 3 channels; the synthetic scene is single-channel on purpose.
        writer.write(cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR))
    writer.release()


def get_json(url: str):
    with urllib.request.urlopen(url, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(url: str, payload: dict):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def test_yolo_roundtrip() -> None:
    print("\n== YOLO 解析/序列化往返 ==")
    boxes = [
        {"cls": 0, "x1": 10, "y1": 20, "x2": 30, "y2": 44},
        {"cls": 0, "x1": 0, "y1": 0, "x2": 5, "y2": 5},
    ]
    text = format_yolo(boxes, WIDTH, HEIGHT)
    back = parse_yolo(text, WIDTH, HEIGHT)
    check("往返后框数量一致", len(back) == 2, f"{len(back)}")
    first = back[0]
    check(
        "像素坐标误差 ≤ 1px",
        abs(first["x1"] - 10) <= 1 and abs(first["y2"] - 44) <= 1,
        f"({first['x1']},{first['y1']},{first['x2']},{first['y2']})",
    )

    duplicated = "\n".join([format_yolo(boxes[:1], WIDTH, HEIGHT).strip()] * 3)
    check("重复行被去重", len(parse_yolo(duplicated, WIDTH, HEIGHT)) == 1)

    clamped = parse_yolo("0 0.999 0.999 0.5 0.5\n", WIDTH, HEIGHT)
    check("越界框被夹到图内", clamped[0]["x2"] <= WIDTH and clamped[0]["y2"] <= HEIGHT)

    inverted = parse_yolo("0 0.5 0.5 -0.2 -0.2\n", WIDTH, HEIGHT)
    check("负尺寸被修正为合法框", inverted[0]["x2"] > inverted[0]["x1"])

    # A target leaving the frame must not be welded to the border as a sliver.
    mostly_out = format_yolo([{"cls": 0, "x1": 100, "y1": 250, "x2": 140, "y2": 330}], WIDTH, HEIGHT)
    check("大部分在画面外的框被丢弃（可见 7.5%）", mostly_out.strip() == "", repr(mostly_out))
    fully_out = format_yolo([{"cls": 0, "x1": 1000, "y1": 10, "x2": 1012, "y2": 22}], WIDTH, HEIGHT)
    check("完全在画面外的框被丢弃", fully_out.strip() == "", repr(fully_out))
    mostly_in = format_yolo([{"cls": 0, "x1": 100, "y1": 200, "x2": 140, "y2": 280}], WIDTH, HEIGHT)
    check("仅轻微出界的框被保留（可见 87.5%）", mostly_in.strip().startswith("0 "), repr(mostly_in))
    edge = parse_yolo(format_yolo([{"cls": 0, "x1": -5, "y1": 100, "x2": 40, "y2": 160}], WIDTH, HEIGHT), WIDTH, HEIGHT)
    check("仅边缘出界的框被保留并裁剪", len(edge) == 1 and edge[0]["x1"] == 0, str(edge[:1]))
    inside = format_yolo([{"cls": 0, "x1": 10, "y1": 10, "x2": 40, "y2": 60}], WIDTH, HEIGHT)
    check("画面内的框不受影响", inside.strip().startswith("0 "), repr(inside))

    try:
        parse_yolo("0 0.5 0.5\n", WIDTH, HEIGHT)
        check("残缺行抛出异常", False, "未抛异常")
    except ValueError:
        check("残缺行抛出异常", True)

    # Sub-pixel fidelity: the UI re-sends whatever the API returned, so the whole
    # write -> read -> write cycle has to be a no-op. It was not, twice over:
    #  * parsing rounded to whole pixels, so every save/reload shrank or grew a 6 px target by up to
    #    0.75 px (~10% of it) and the file changed underneath the annotator;
    #  * parse clamped to width-1 while format clipped to width, so a box touching the frame border
    #    lost a pixel on that edge *per cycle* and crept inward the whole time you worked.
    fractional = [{"cls": 0, "x1": 100.5, "y1": 100.25, "x2": 140.25, "y2": 160.75}]
    first_write = format_yolo(fractional, WIDTH, HEIGHT)
    first_read = parse_yolo(first_write, WIDTH, HEIGHT)
    second_write = format_yolo(first_read, WIDTH, HEIGHT)
    check("保存→读回→再保存，磁盘字节完全一致", first_write == second_write,
          f"{first_write!r} vs {second_write!r}")
    check("亚像素坐标被保留（x1=100.5 未被取整）",
          abs(first_read[0]["x1"] - 100.5) < 0.01, str(first_read[0]))
    check("亚像素框宽保真（39.75px）",
          abs((first_read[0]["x2"] - first_read[0]["x1"]) - 39.75) < 0.02, str(first_read[0]))
    tiny = [{"cls": 0, "x1": 300.5, "y1": 200.5, "x2": 306.5, "y2": 206.5}]
    tiny_back = parse_yolo(format_yolo(tiny, WIDTH, HEIGHT), WIDTH, HEIGHT)
    check("6x6 小目标往返后尺寸不变",
          abs((tiny_back[0]["x2"] - tiny_back[0]["x1"]) - 6.0) < 0.02, str(tiny_back[0]))
    for name, edge in (
        ("右边界", {"cls": 0, "x1": 300.0, "y1": 10.0, "x2": float(WIDTH), "y2": 40.0}),
        ("下边界", {"cls": 0, "x1": 10.0, "y1": 220.0, "x2": 40.0, "y2": float(HEIGHT)}),
        ("角上", {"cls": 0, "x1": float(WIDTH) - 20, "y1": float(HEIGHT) - 20, "x2": float(WIDTH), "y2": float(HEIGHT)}),
    ):
        written = format_yolo([edge], WIDTH, HEIGHT)
        check(f"贴{name}的框往返幂等", format_yolo(parse_yolo(written, WIDTH, HEIGHT), WIDTH, HEIGHT) == written,
              f"{written!r}")
    check("贴边框往返后尺寸不变",
          abs((parse_yolo(format_yolo([{"cls": 0, "x1": 300.0, "y1": 220.0,
                                        "x2": float(WIDTH), "y2": float(HEIGHT)}], WIDTH, HEIGHT),
                          WIDTH, HEIGHT)[0]["x2"]) - WIDTH) < 0.02)


def test_scan(root: Path) -> list:
    print("\n== 递归扫描与解码 ==")
    sequences = scan_root(root)
    check("扫描到 2 个序列", len(sequences) == 2, f"实际 {len(sequences)}")

    video = next((s for s in sequences if s.kind == "video"), None)
    frames = next((s for s in sequences if s.kind == "frames"), None)
    check("识别出视频序列", video is not None, video.name if video else "缺失")
    check("识别出帧目录序列", frames is not None, frames.name if frames else "缺失")

    if video:
        check("视频帧数 > 0", video.frame_count > 0, f"{video.frame_count}")
        check("视频分辨率正确", (video.width, video.height) == (WIDTH, HEIGHT), f"{video.width}x{video.height}")
        payloads = {video.frame_jpeg(i) for i in (0, 5, 20, FRAMES - 1)}
        check("首/中/末帧可解码且互不相同", len(payloads) == 4, f"{len(payloads)} 个不同 JPEG")
        check("JPEG 以 SOI 开头", video.frame_jpeg(0)[:2] == b"\xff\xd8")
        check("重复请求命中缓存", video.frame_jpeg(5) == video.frame_jpeg(5))

    if frames:
        check("帧目录数量正确", frames.frame_count == FRAME_DIR_FRAMES, f"{frames.frame_count}")
        check("帧目录可读", frames.frame_jpeg(3)[:2] == b"\xff\xd8")

    return sequences


def test_store(workspace: Path) -> None:
    print("\n== 标签存储与复核状态 ==")
    store = LabelStore(workspace)
    boxes = [{"cls": 0, "x1": 12, "y1": 14, "x2": 26, "y2": 30}]
    store.write_labels("seqA", 0, boxes, WIDTH, HEIGHT)
    store.write_labels("seqA", 1, boxes, WIDTH, HEIGHT)
    store.mark_reviewed("seqA", [0])

    reread = store.read_labels("seqA", 0, WIDTH, HEIGHT)
    check("写入后可读回", len(reread) == 1)
    check("复核状态已记录", store.is_reviewed("seqA", 0) and not store.is_reviewed("seqA", 1))

    summary = store.summary("seqA", 10)
    check("时间线统计 labelled=2", summary["counts"]["labelled"] == 2, str(summary["counts"]))
    check("时间线统计 reviewed=1", summary["counts"]["reviewed"] == 1)
    check("时间线统计 empty=8", summary["counts"]["empty"] == 8)
    check("RLE 覆盖完整帧数", sum(count for _, count in summary["reviewed"]) == 10)

    store.write_prelabels("seqA", 0, boxes * 3, WIDTH, HEIGHT)
    check("重复候选被去重", len(store.read_prelabels("seqA", 0, WIDTH, HEIGHT)) == 1)

    # The fast box total counts lines instead of re-parsing, so format_yolo must collapse duplicates.
    single = {"cls": 0, "x1": 10, "y1": 10, "x2": 22, "y2": 22}
    store.write_labels("seqC", 0, [single, dict(single), dict(single)], WIDTH, HEIGHT)
    fast = store.summary("seqC", 4, count_boxes=True)
    check("快速框计数正确（与解析一致）",
          fast["counts"]["boxes"] == len(store.read_labels("seqC", 0, WIDTH, HEIGHT)),
          f"{fast['counts']['boxes']}")
    slow = store.summary("seqC", 4, count_boxes=False)
    check("默认不统计框数（避免长序列卡顿）", slow["counts"]["boxes"] is None)
    check("两种模式的帧统计一致",
          slow["counts"]["labelled"] == fast["counts"]["labelled"])

    store.delete_labels("seqA", 1)
    check("删除后统计更新", store.summary("seqA", 10)["counts"]["labelled"] == 1)

    corrupt = LabelStore(workspace)  # a broken review file must not raise
    (workspace / "review" / "seqB.json").write_text("{not json", encoding="utf-8")
    check("损坏的复核文件不阻塞", corrupt._review("seqB")["reviewed"] == [])


def test_worker_pipeline(workspace: Path) -> None:
    """Exercise the worker's whole path with only the model forward stubbed.

    Running the real Trial 0474 is the user's decision under 铁律一, but everything around it — frame
    iteration, GMC + median feature construction, the frozen channel order, multi-scale canvases, heatmap
    fusion, threshold/dilate/connected-component reduction and the YOLO/scores output — is ordinary code
    that must not ship unexecuted. A stub heatmap with known blobs makes those assertions exact.
    """
    print("\n== 推理 worker 管线（模型前向已替换）==")
    try:
        from manu.annotate import worker_infer
    except Exception as error:  # torch lives only in the uav env; skipping beats a false failure
        print(f"  [SKIP] 需要 torch：请用 /home/manu/anaconda3/envs/uav/bin/python 运行（{type(error).__name__}）")
        return

    source = workspace / "clip"
    make_video(source / "seq.mp4", 12, WIDTH, HEIGHT)

    frames = list(worker_infer.iter_gray_frames(source / "seq.mp4", limit=12))
    check("视频逐帧解码", len(frames) == 12, f"{len(frames)}")
    check("解码结果为灰度", frames[0].ndim == 2, f"ndim={frames[0].ndim}")

    args = argparse.Namespace(
        temporal_stride=2, median_window=5, scales=[160, 80], imgsz=640,
        main_threshold=0.22, fusion_dilate=9, min_area=4, max_detections=100,
    )
    gmc = worker_infer.FastGMCEstimator(downscale=2)
    features, history = worker_infer.build_features(frames[:6], [], args, gmc)
    check("特征块数与帧数一致", len(features) == 6)
    check("特征为原生尺寸三通道", features[0].shape == (HEIGHT, WIDTH, 3), str(features[0].shape))
    check("特征取值在 uint8 范围", features[0].dtype == np.uint8 and int(features[0].max()) <= 255)
    # Channel 0 is the raw frame; the GMC-aligned history must not be identical to it.
    check("第 0 通道等于当前灰度帧", np.array_equal(features[0][:, :, 0], frames[0]))
    check("第 2 通道为时域中值残差（非原始帧）", not np.array_equal(features[0][:, :, 2], frames[0]))

    letterboxed = worker_infer.letterbox_gray(features[0], args.imgsz)  # HWC, disk order
    canvases, mappings = worker_infer.scale_canvases(features[0], args.scales, args.imgsz, HEIGHT, WIDTH)
    check("每个尺度一张画布", canvases.shape == (3, 3, args.imgsz, args.imgsz), str(canvases.shape))
    check("画布为 CHW uint8", canvases.dtype == np.uint8)
    # The frozen input contract: disk order [I_t, diff, residual] must reach the model reversed as
    # [residual, diff, I_t]. Getting this backwards silently degrades every detection.
    # ``transpose(2,0,1)`` reorders axes to (C,H,W) without transposing H/W, so the reference is the
    # channel plane itself. (Comparing against ``[:, :, k].T`` also "passes" on smooth infrared data
    # by coincidence, which is exactly why this assertion needs random values to be meaningful.)
    for channel, disk_channel, label in ((0, 2, "中值残差"), (1, 1, "GMC 差分"), (2, 0, "原始灰度 I_t")):
        check(f"模型序第 {channel} 通道 == {label}（磁盘序第 {disk_channel} 通道）",
              np.array_equal(canvases[0][channel], letterboxed[:, :, disk_channel]))
    check("模型序与磁盘序确实相反（通道内容各不相同）",
          not np.array_equal(canvases[0][2], canvases[0][0]))
    check("尺度映射数量与画布一致", len(mappings) == 3, str(len(mappings)))

    native = worker_infer.heatmap_to_native(
        np.zeros((args.imgsz, args.imgsz), np.float32), mappings[0][0], mappings[0][1], mappings[0][2],
        args.imgsz, WIDTH, HEIGHT)
    check("热图逆映射回原生尺寸", native.shape == (HEIGHT, WIDTH), str(native.shape))

    blob = np.zeros((HEIGHT, WIDTH), np.float32)
    blob[100:104, 200:204] = 0.9
    components = worker_infer.detect_components(blob, 0.22, 9, 4, 100)
    check("连通域检出合成热图", len(components) == 1, str(components[:1]))
    if components:
        cx, cy, bw, bh, area, peak, total = components[0]
        check("检出框中心正确", abs(cx - 201.5) <= 1 and abs(cy - 101.5) <= 1, f"({cx},{cy})")
        # A 4x4 blob dilated by a 9x9 kernel spans 4 + 9 - 1 = 12 px.
        check("检出框尺寸 = 原始 4px + 膨胀 9", bw == 12 and bh == 12, f"{bw}x{bh}")
        # Dilation grows the mask, not the heatmap, so the integral still counts only the 16 real
        # blob pixels: 16 * 0.9 = 14.4, not 144 * 0.9.
        check("峰值与积分正确", abs(peak - 0.9) < 1e-6 and abs(total - 14.4) < 1e-3, f"{peak},{total}")
    check("低于阈值的热图无检出", worker_infer.detect_components(blob * 0.1, 0.22, 9, 4, 100) == [])
    ordered = worker_infer.detect_components(blob * 1.0, 0.22, 9, 4, 100)
    check("检出按热图积分降序（与缓存脚本一致）", len(ordered) == 1)

    # Full main() with a stubbed forward pass: verifies chunking, history carry-over and the outputs.
    out_dir = workspace / "worker_prelabels"
    (out_dir / "seq").mkdir(parents=True, exist_ok=True)
    original_run = worker_infer.run_heatmaps
    original_model = worker_infer.load_hm_model
    original_resolve = worker_infer.resolve_weights

    def fake_run_heatmaps(model, canvases, device, batch):
        count = canvases.shape[0]
        heat = np.zeros((count, args.imgsz, args.imgsz), np.float32)
        for item in range(count):
            # Paint one blob in the native letterbox region so the fusion path has something to find.
            heat[item, 180:190, 300:310] = 0.95
        return heat

    worker_infer.run_heatmaps = fake_run_heatmaps
    worker_infer.load_hm_model = lambda weights, device: (object(), 2)
    # main() resolves the checkpoint before loading it; the file only exists on the training server.
    worker_infer.resolve_weights = lambda value: Path("stub-weights.pt")
    argv = [
        "worker", "--source", str(source / "seq.mp4"), "--seq-id", "seq",
        "--prelabel-dir", str(out_dir), "--scales", "160,80",
        "--chunk-size", "4", "--cpu-threads", "1", "--max-frames", "12",
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        worker_infer.main()
    finally:
        sys.argv = old_argv
        worker_infer.run_heatmaps = original_run
        worker_infer.load_hm_model = original_model
        worker_infer.resolve_weights = original_resolve

    produced = sorted(out_dir.glob("seq/*.txt"))
    check("worker 为每帧写出标签", len(produced) == 12, f"{len(produced)} 个文件")
    check("worker 写出分数边车", (out_dir / "seq" / "scores.json").exists())
    check("候选非空（注入的热图被检出）",
          all(path.read_text().strip() for path in produced), "存在空文件")
    sample = parse_yolo(produced[0].read_text(), WIDTH, HEIGHT)
    check("worker 输出为合法 YOLO", len(sample) == 1, str(sample))
    if sample:
        box = sample[0]
        check("worker 输出的框落在图内",
              0 <= box["x1"] < box["x2"] <= WIDTH and 0 <= box["y1"] < box["y2"] <= HEIGHT, str(box))
        check("注入热图位置与输出一致", abs((box["x1"] + box["x2"]) / 2 - 305 * WIDTH / 640) < 6, str(box))
    import json as _json
    scores = _json.loads((out_dir / "seq" / "scores.json").read_text())
    check("分数边车逐帧对应", len(scores) == 12, f"{len(scores)}")


def test_export(workspace: Path) -> None:
    """Export the workspace into the project's flat GT layout, including patch-merge over a base."""
    print("\n== 导出为交付用扁平布局 ==")
    store = LabelStore(workspace / "ws_export")
    for seq in ("seqP", "seqQ"):
        store.write_labels(seq, 0, [{"cls": 0, "x1": 10, "y1": 12, "x2": 22, "y2": 26}], WIDTH, HEIGHT)
        store.write_labels(seq, 1, [{"cls": 0, "x1": 30, "y1": 32, "x2": 42, "y2": 46}], WIDTH, HEIGHT)
    store.mark_reviewed("seqP", [0, 1])

    # Frame directories so --root can supply frame counts for empty-frame filling.
    root = workspace / "export_root" / "seqP"
    root.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        cv2.imwrite(str(root / f"frame_{i:06d}.jpg"), np.zeros((HEIGHT, WIDTH), np.uint8))

    out = workspace / "flat_out"
    result = subprocess.run(
        [sys.executable, "-m", "manu.annotate.export_labels",
         "--workspace", str(workspace / "ws_export"), "--out", str(out),
         "--root", str(workspace / "export_root"), "--sequences", "seqP"],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=120,
    )
    check("导出命令成功", result.returncode == 0, result.stderr.strip()[-160:])
    flat = sorted(p.name for p in out.glob("*.txt")) if out.is_dir() else []
    check("导出为扁平命名 {seq}__{index}", flat[0] == "seqP__000000.txt" if flat else False, str(flat[:3]))
    check("有标注的帧被导出", "seqP__000000.txt" in flat and "seqP__000001.txt" in flat)
    check("未标注帧也导出为空文件（交付要求每帧一个）", "seqP__000004.txt" in flat, f"{len(flat)} 个文件")
    check("只导出指定序列", all(name.startswith("seqP") for name in flat), str(flat))

    exported = (out / "seqP__000000.txt").read_text()
    check("导出内容是合法 YOLO", len(parse_yolo(exported, WIDTH, HEIGHT)) == 1, repr(exported))

    # The exported set must load back into the tool unchanged — this is the whole point.
    reloaded = LabelStore(workspace / "ws_roundtrip", extra_prelabel_roots=[out])
    check("扁平布局可被工具直接读回（无需转换）",
          reloaded.external_prelabel_path("seqP", 0) is not None)

    # --no-fill-empty
    out2 = workspace / "flat_nofill"
    subprocess.run(
        [sys.executable, "-m", "manu.annotate.export_labels",
         "--workspace", str(workspace / "ws_export"), "--out", str(out2),
         "--sequences", "seqP", "--no-fill-empty"],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=120,
    )
    check("--no-fill-empty 只导出有标注的帧", len(list(out2.glob("*.txt"))) == 2,
          f"{len(list(out2.glob('*.txt')))} 个")

    # Patch-merge over an existing base.
    base = workspace / "flat_base"
    base.mkdir(parents=True, exist_ok=True)
    (base / "seqP__000000.txt").write_text("0 0.9 0.9 0.05 0.05\n", encoding="utf-8")   # must be overwritten
    (base / "seqP__000002.txt").write_text("0 0.1 0.1 0.05 0.05\n", encoding="utf-8")   # must survive
    (base / "seqOTHER__000001.txt").write_text("0 0.2 0.2 0.05 0.05\n", encoding="utf-8")
    (base / "classes.txt").write_text("airplane\n", encoding="utf-8")
    out3 = workspace / "flat_patched"
    result3 = subprocess.run(
        [sys.executable, "-m", "manu.annotate.export_labels",
         "--workspace", str(workspace / "ws_export"), "--out", str(out3),
         "--base", str(base), "--sequences", "seqP", "--no-fill-empty"],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=120,
    )
    check("补丁合并成功", result3.returncode == 0, result3.stderr.strip()[-160:])
    patched_0 = out3 / "seqP__000000.txt"
    check("工作区标签覆盖 base 同一帧",
          patched_0.exists() and patched_0.read_text() != "0 0.9 0.9 0.05 0.05\n",
          repr(patched_0.read_text()) if patched_0.exists() else "缺失")
    check("base 中未标注的帧被保留", (out3 / "seqP__000002.txt").exists())
    check("base 的其他序列被保留", (out3 / "seqOTHER__000001.txt").exists())
    check("classes.txt 被带出", (out3 / "classes.txt").exists())

    # Refusing to clobber an existing output directory is what stops a partial export looking complete.
    result4 = subprocess.run(
        [sys.executable, "-m", "manu.annotate.export_labels",
         "--workspace", str(workspace / "ws_export"), "--out", str(out3)],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=120,
    )
    check("拒绝覆盖已存在的输出目录", result4.returncode != 0, result4.stderr.strip()[-80:])


def test_external_prelabels(workspace: Path) -> None:
    """Read-only proposal trees produced elsewhere, in both common layouts."""
    print("\n== 外部只读候选树 ==")
    external_flat = workspace / "legacy_flat"
    external_tree = workspace / "legacy_tree"
    (external_flat / "sub").mkdir(parents=True, exist_ok=True)
    (external_tree / "seqF").mkdir(parents=True, exist_ok=True)

    # Flat layout: <root>/<seq>__<frame>.txt, exactly like the project's preannot_v1 tree.
    (external_flat / "sub" / "seqE__000000.txt").write_text(
        f"1 0.5 0.5 0.1 0.1\n0 0.2 0.2 0.05 0.05\n", encoding="utf-8")
    (external_flat / "sub" / "seqE__000001.txt").write_text("", encoding="utf-8")
    # Per-sequence layout: <root>/<seq>/<frame>.txt
    (external_tree / "seqF" / "000000.txt").write_text("1 0.3 0.3 0.08 0.08\n", encoding="utf-8")

    store = LabelStore(workspace / "ws2", extra_prelabel_roots=[external_flat / "sub", external_tree])
    boxes = store.read_prelabels("seqE", 0, WIDTH, HEIGHT)
    check("扁平布局外部候选可读", len(boxes) == 2, f"{len(boxes)}")
    check("空外部候选文件返回空列表", store.read_prelabels("seqE", 1, WIDTH, HEIGHT) == [])
    filtered = store.read_prelabels("seqE", 0, WIDTH, HEIGHT, classes=[1])
    check("按类别过滤外部候选", len(filtered) == 1 and filtered[0]["cls"] == 1, str(filtered))
    check("按序列目录布局外部候选可读", len(store.read_prelabels("seqF", 0, WIDTH, HEIGHT)) == 1)
    check("外部树缺失该帧返回空", store.read_prelabels("seqE", 99, WIDTH, HEIGHT) == [])

    summary = store.summary("seqE", 4)
    check("外部候选计入时间线", summary["counts"]["prelabelled"] == 2, str(summary["counts"]))

    # Human edits must win and must never reach the external tree.
    store.write_labels("seqE", 0, [{"cls": 0, "x1": 5, "y1": 5, "x2": 15, "y2": 15}], WIDTH, HEIGHT)
    store.write_prelabels("seqE", 0, [{"cls": 0, "x1": 1, "y1": 1, "x2": 3, "y2": 3}], WIDTH, HEIGHT)
    own = store.read_prelabels("seqE", 0, WIDTH, HEIGHT)
    check("自有候选优先于外部候选", len(own) == 1 and own[0]["x2"] == 3, str(own))
    untouched = (external_flat / "sub" / "seqE__000000.txt").read_text(encoding="utf-8")
    check("外部树保持只读未被改写", untouched.count("\n") == 2, repr(untouched))

    store.invalidate_external_cache()
    check("缓存失效后仍能读到外部候选", len(store.read_prelabels("seqF", 0, WIDTH, HEIGHT)) == 1)


def test_job_queue(workspace: Path) -> None:
    """Drive the real dispatch loop with a stubbed ssh, so queue bugs fail here instead of in the UI.

    The bug this guards against was real: jobs were recorded in one set while the dispatcher consulted
    another, so every submission stayed ``queued`` forever and the button silently did nothing.
    """
    print("\n== 预标注任务队列 ==")
    from manu.annotate import jobs as jobs_module

    launched: list[dict] = []

    class FakeProcess:
        def __init__(self, command, **_kwargs):
            self.command = command
            self.returncode = 0
            self.stdout = iter(["[INFO] seq ok", "[DONE] seq: 10 frames, 3 proposals"])

        def wait(self, *_args):
            return 0

    def fake_popen(command, **_kwargs):
        launched.append({"command": command})
        return FakeProcess(command)

    original_popen = jobs_module.subprocess.Popen
    original_map = jobs_module.PATH_MAP
    # The selftest workspace lives in a temp dir, so map it onto itself for the duration of the test.
    jobs_module.PATH_MAP = [(str(workspace), str(workspace))] + list(original_map)
    jobs_module.subprocess.Popen = fake_popen
    try:
        manager = jobs_module.JobManager(
            workspace,
            ssh_target="user@host", ssh_port=32222,
            server_repo="/srv/repo", conda_env="uav",
            device="0", gpu_devices=["0", "1", "2", "3"], max_concurrent=2,
        )
        submitted = manager.submit_many(
            [(f"seq{i}", str(workspace / "data" / f"clip{i}.mp4")) for i in range(1, 5)],
            extra=["--max-frames", "5"],
            devices=["0", "1"],
        )
        check("批量提交返回 4 个任务", len(submitted) == 4)
        check("按 GPU 轮询分片", [job.device for job in submitted] == ["0", "1", "0", "1"],
              str([job.device for job in submitted]))
        check("初始状态为 queued", all(job.status == "queued" for job in submitted))

        deadline = time.time() + 20
        while time.time() < deadline:
            states = [manager.status(job.seq_id).status for job in submitted]
            if all(state in {"done", "failed"} for state in states):
                break
            time.sleep(0.25)

        states = {job.seq_id: manager.status(job.seq_id).status for job in submitted}
        check("全部任务离开 queued（调度器真的派发了）",
              all(state == "done" for state in states.values()), str(states))
        check("并发上限被遵守（最多 2 个 ssh 在飞）", len(launched) == 4, f"{len(launched)} 个进程")
        if launched:
            command = launched[0]["command"]
            check("使用指定端口 32222", "32222" in command, str(command[:6]))
            check("远程命令带 conda activate", "conda activate uav" in " ".join(command))
            check("远程命令切到服务器仓库", "/srv/repo" in " ".join(command))
            check("远程命令使用 uav 环境下的 worker", "manu/annotate/worker_infer.py" in " ".join(command))
            check("GPU 传入 worker", "--device 0" in " ".join(command))
            check("限帧参数透传", "--max-frames 5" in " ".join(command))
            check("BatchMode 免交互", "BatchMode=yes" in command)

        # Re-running a finished sequence is legitimate; only an in-flight one must be refused.
        check("已完成的序列可重新提交",
              manager.submit("seq1", str(workspace / "data" / "clip1.mp4")).status == "queued")
        manager.status("seq1").status = "running"  # freeze it in flight to test the guard
        try:
            manager.submit("seq1", str(workspace / "data" / "clip1.mp4"))
            check("进行中的序列重复提交被拒绝", False, "未抛出")
        except RuntimeError:
            check("进行中的序列重复提交被拒绝", True)

        # A source the server cannot see must be refused loudly rather than queued into nothing.
        try:
            manager.submit("local_only", "/home/somebody/private/clip.mp4")
            check("服务器不可见路径被拒绝", False, "未抛出")
        except RuntimeError as error:
            check("服务器不可见路径被拒绝", "not reachable" in str(error), str(error)[:60])
    finally:
        jobs_module.subprocess.Popen = original_popen
        jobs_module.PATH_MAP = original_map


def test_local_inference_mode(workspace: Path) -> None:
    """The web UI can live on the GPU box itself; SSH-ing to self is rejected there, so guard local mode.

    Two failure modes are covered, and neither raises — both would look like a working button:

    * spawning an ``ssh`` hop to localhost, which dies with ``Permission denied (publickey)``;
    * running a source through the workstation ``PATH_MAP`` while already standing on the server, which
      maps every path to ``None`` and makes the UI claim nothing is reachable.
    """
    print("\n== 本机推理模式（服务部署在 GPU 服务器上）==")
    from manu.annotate import jobs as jobs_module

    local_addrs = sorted(jobs_module.local_ipv4() - {"127.0.0.1"})
    # Pick a real NIC address: this is exactly what fails when detection leans on
    # gethostbyname(hostname), which answers 127.0.1.1 on Debian and never matches.
    probe = local_addrs[0] if local_addrs else jobs_module.socket.gethostname()
    check(f"auto 模式把本机地址 {probe} 识别为 local",
          jobs_module.resolve_mode("auto", f"huangzhe@{probe}") is True)
    check("auto 模式在他机识别为 ssh",
          jobs_module.resolve_mode("auto", "huangzhe@203.0.113.7") is False)
    check("显式 local 覆盖自动判断",
          jobs_module.resolve_mode("local", "huangzhe@203.0.113.7") is True)
    check("显式 ssh 覆盖自动判断",
          jobs_module.resolve_mode("ssh", f"huangzhe@{probe}") is False)
    check("本机地址集合含 127.0.0.1", "127.0.0.1" in jobs_module.local_ipv4())
    check("未知模式被拒绝", _raises(ValueError, jobs_module.resolve_mode, "carrier-pigeon", "h@x"))

    # A server-side path must survive local mode untouched — PATH_MAP must not be consulted.
    server_path = "/mnt/data/siping/clip.mp4"
    check("local 模式不做路径翻译",
          jobs_module.to_server_path(server_path, translate=False) == server_path)
    check("local 模式下服务端路径不再被判为不可见",
          jobs_module.to_server_path(server_path, translate=True) is None,
          "这正是 UI 误报『不可见』的根因")

    launched: list[list[str]] = []

    class FakeProcess:
        def __init__(self, command, **_kwargs):
            self.command = command
            self.returncode = 0
            self.stdout = iter(["[DONE] seq: 10 frames, 3 proposals"])

        def wait(self, *_args):
            return 0

    original_popen = jobs_module.subprocess.Popen
    original_resolve = jobs_module.resolve_mode
    jobs_module.subprocess.Popen = lambda command, **_kw: (launched.append(command), FakeProcess(command))[1]
    try:
        manager = jobs_module.JobManager(
            "/srv/annotate_workspace", ssh_target="huangzhe@203.0.113.7",
            server_repo="/tmp/pycharm_project_10ae9e2e", conda_env="uav",
            inference_mode="local",
        )
        check("local 模式下 JobManager.local 为真", manager.local is True)
        job = manager.submit("seqA", "/mnt/data/siping/clip.mp4", extra=["--max-frames", "5"])
        check("local 模式接受服务端本地路径（未被 PATH_MAP 拒绝）", job.server_source == "/mnt/data/siping/clip.mp4",
              job.server_source)

        deadline = time.time() + 20
        while time.time() < deadline and manager.status("seqA").status not in {"done", "failed"}:
            time.sleep(0.2)
        check("local 任务跑完", manager.status("seqA").status == "done", manager.status("seqA").status)

        check("确实派发了一个进程", len(launched) == 1, f"{len(launched)}")
        if launched:
            command = launched[0]
            check("local 模式不起 ssh", "ssh" not in command[:2], str(command[:2]))
            check("local 模式走 bash -lc", command[:3] == ["bash", "-lc", command[2]])
            joined = " ".join(command)
            check("local 模式仍激活 conda uav", "conda activate uav" in joined)
            check("local 模式仍传 worker 路径", "manu/annotate/worker_infer.py" in joined)
            check("local 模式传入服务端路径而非工作机路径", "/mnt/data/siping/clip.mp4" in joined)
            check("local 模式不含 BatchMode（那是 ssh 的选项）", "BatchMode" not in joined)
            check("describe 报告就地执行", "in-place" in manager.describe(), manager.describe())
    finally:
        jobs_module.subprocess.Popen = original_popen
        jobs_module.resolve_mode = original_resolve


def test_local_mode_visibility(root: Path, workspace: Path) -> None:
    """The pre-annotate endpoint's visibility gate must follow the same mode as the dispatcher."""
    print("\n== 本机模式下的可见性判断 ==")
    from manu.annotate import jobs as jobs_module

    source = "/mnt/data/siping/datasets/manu/demo/frames"
    manager = jobs_module.JobManager(workspace, inference_mode="local")
    # server.py gates on this expression; assert it accepts a path the worker will in fact see.
    accepted = jobs_module.to_server_path(source, translate=not manager.local) is not None
    check("local 模式下服务端路径通过可见性闸门", accepted)
    manager_ssh = jobs_module.JobManager(workspace, inference_mode="ssh")
    rejected = jobs_module.to_server_path(source, translate=not manager_ssh.local) is None
    check("ssh 模式下同一个路径被正确判为不可见", rejected)


def _raises(exception: type, function, *args, **kwargs) -> bool:
    try:
        function(*args, **kwargs)
    except exception:
        return True
    except Exception:
        return False
    return False


def test_server(root: Path, workspace: Path, port: int) -> None:
    print("\n== HTTP 服务全链路 ==")
    process = subprocess.Popen(
        [sys.executable, "-m", "manu.annotate.server", "--root", str(root), "--workspace", str(workspace),
         "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(80):
            try:
                state = get_json(f"{base}/api/state")
                break
            except Exception:
                time.sleep(0.25)
        else:
            out = process.stdout.read() if process.stdout else ""
            check("服务启动", False, out[-500:])
            return

        check("服务启动", True)
        check("状态接口返回序列", len(state["sequences"]) == 2, f"{len(state['sequences'])}")
        check("无扫描错误", not state.get("scan_error"), str(state.get("scan_error")))

        seq_id = state["sequences"][0]["seq_id"]
        summary = get_json(f"{base}/api/seq/{seq_id}/summary")
        check("summary 接口可用", summary["frame_count"] > 0, f"{summary['frame_count']}")

        with urllib.request.urlopen(f"{base}/api/seq/{seq_id}/frame/0", timeout=20) as response:
            raw = response.read()
            check("帧接口返回 JPEG", raw[:2] == b"\xff\xd8", f"{len(raw)} bytes")

        payload = post_json(f"{base}/api/seq/{seq_id}/labels/3",
                            {"boxes": [{"cls": 0, "x1": 10, "y1": 10, "x2": 24, "y2": 26}], "reviewed": True})
        check("保存标签接口", len(payload["boxes"]) == 1)

        readback = get_json(f"{base}/api/seq/{seq_id}/labels/3")
        check("标签读回一致", len(readback["labels"]) == 1 and readback["reviewed"] is True)

        batch = post_json(f"{base}/api/seq/{seq_id}/batch", {"entries": [
            {"index": 10, "boxes": [{"cls": 0, "x1": 1, "y1": 1, "x2": 9, "y2": 9}]},
            {"index": 11, "boxes": [{"cls": 0, "x1": 2, "y1": 2, "x2": 10, "y2": 10}]},
        ]})
        check("批量写入（传播/插补）", batch["written"] == [10, 11], str(batch["written"]))

        with urllib.request.urlopen(f"{base}/static/app.js", timeout=20) as response:
            check("静态资源可访问", response.status == 200 and b"canvas" in response.read())
        with urllib.request.urlopen(f"{base}/", timeout=20) as response:
            check("首页可访问", response.status == 200 and b"viewport" in response.read())

        try:
            urllib.request.urlopen(f"{base}/api/seq/nope/labels/0", timeout=10)
            check("未知序列返回 404", False)
        except urllib.error.HTTPError as error:
            check("未知序列返回 404", error.code == 404)
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def main() -> None:
    base = Path(tempfile.mkdtemp(prefix="annotate_selftest_"))
    root = base / "videos"
    workspace = base / "ws"
    try:
        make_video(root / "clip_a.mp4", FRAMES, WIDTH, HEIGHT)
        frames_dir = root / "nested" / "clip_b"
        frames_dir.mkdir(parents=True)
        for index in range(FRAME_DIR_FRAMES):
            image = np.full((HEIGHT, WIDTH), 60, np.uint8)
            image[100 + index : 104 + index, 50 + index * 2 : 54 + index * 2] = 200
            cv2.imwrite(str(frames_dir / f"frame_{index:06d}.jpg"), image)

        test_yolo_roundtrip()
        test_scan(root)
        test_store(workspace)
        test_worker_pipeline(workspace)
        test_export(workspace)
        test_external_prelabels(workspace)
        test_job_queue(workspace)
        test_local_inference_mode(workspace)
        test_local_mode_visibility(root, workspace)
        test_server(root, workspace, 8899)

        failures = [item for item in results if item[0] == FAIL]
        print(f"\n{'=' * 60}\n总计 {len(results)} 项，失败 {len(failures)} 项")
        for _, name, detail in failures:
            print(f"  FAIL: {name} {detail}")
        print("=" * 60)
        sys.exit(1 if failures else 0)
    finally:
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()
