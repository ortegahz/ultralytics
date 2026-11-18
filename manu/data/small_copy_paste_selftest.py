"""微小目标 Copy-Paste 增强的 CPU 自检（2026-11-13）。

用法：
    python manu/data/small_copy_paste_selftest.py

覆盖：
  ① prob=0 ⇒ 图像与标签**逐位不变**（默认关闭的等价性保证）
  ② prob=1 ⇒ 真的贴出目标（框数 +1、图像改变、新框与原框不同）
  ③ 源框超过 max_src_px ⇒ 不作为源（不贴）
  ④ match_bg=True ⇒ 落点背景亮度不匹配时不贴（对应 CLI 的 --cp-allow-hot 反向）
  ⑤ 标签形状自洽（instances.bboxes 与 cls 行数一致）
  ⑥ **降尺度粘贴**：默认 scale=1.0 ⇒ 输出与**旧实现逐位一致**（黄金 sha256 比对）
  ⑦ scale=0.5 ⇒ 贴出的框显著变小（6px 源 → 3px）、标签数量正确
  ⑧ 缩放后 <2px 的目标被跳过（同图里其它目标照贴）
有 ultralytics 时用**真** Instances，没有时用等价桩对象。
"""
from __future__ import annotations

import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ultralytics.utils.instance import Instances  # noqa: PLC0415

    def _mk_inst(boxes):
        return Instances(np.asarray(boxes, dtype=np.float32), segments=None, keypoints=None,
                         bbox_format="xyxy", normalized=False)
    _REAL = True
except Exception:  # noqa: BLE001
    class _Stub:  # noqa: D401
        """无 ultralytics 时的等价桩（像素级 xyxy 容器）。"""

        def __init__(self, boxes):
            self.bboxes = np.asarray(boxes, dtype=np.float32)
            self.segments = None
            self.keypoints = None

    def _mk_inst(boxes):
        return _Stub(boxes)
    _REAL = False

from manu.data.small_copy_paste import SmallObjectCopyPaste  # noqa: E402


def _labels(with_hot: bool = False) -> dict:
    img = np.full((640, 640, 3), 40, dtype=np.uint8)
    img[300:306, 300:306] = 220          # 一个 6px 的"目标"
    boxes = [[300.0, 300.0, 306.0, 306.0]]
    if with_hot:
        img[100:106, 100:106] = 240      # 另一处更亮的"热区"
        boxes.append([100.0, 100.0, 106.0, 106.0])
    return {
        "img": img,
        "instances": _mk_inst(boxes),
        "cls": np.array([[0.0]] * len(boxes), dtype=np.float32),
    }


def _golden_labels() -> dict:
    """**黄金样本**（确定性）：噪声底 + 一个冷目标 + 一个亮目标。

    sha256 = 9155d4a7…81943 是「**降尺度功能落地之前**」的旧实现跑出来的
    ⇒ 用来证明「默认 scale=1.0 ⇒ 与旧实现逐位一致」。
    """
    rng = np.random.default_rng(1234)
    img = rng.integers(30, 60, size=(640, 640, 3), dtype=np.uint8)
    img[300:306, 300:306] = 220
    img[120:126, 500:506] = 200
    boxes = [[300.0, 300.0, 306.0, 306.0], [500.0, 120.0, 506.0, 126.0]]
    return {
        "img": img,
        "instances": _mk_inst(boxes),
        "cls": np.array([[0.0]] * len(boxes), dtype=np.float32),
    }


GOLDEN_SHA256 = "9155d4a79fdd4fab9fe68418f2f8db0af247f24d241742076886fff338481943"
GOLDEN_BOXES = [[300.0, 300.0, 306.0, 306.0], [500.0, 120.0, 506.0, 126.0],
                [580.0, 200.0, 586.0, 206.0], [551.0, 30.0, 557.0, 36.0]]

def main() -> int:
    np.random.seed(3)
    print(f"[selftest] ultralytics Instances = {'真' if _REAL else '桩'}")

    # ① 关闭 ⇒ 逐位不变
    cp = SmallObjectCopyPaste(prob=0.0)
    lb = _labels()
    before = lb["img"].copy()
    cp(lb)
    assert np.array_equal(before, lb["img"]), "prob=0 时图像被改了"
    assert len(lb["instances"].bboxes) == 1, "prob=0 时框数被改了"
    print("[selftest] ① prob=0 逐位不变 ✔")

    # ② 打开 ⇒ 贴出目标
    cp = SmallObjectCopyPaste(prob=1.0, max_src_px=12.0, max_per_img=3)
    lb = _labels()
    before = lb["img"].copy()
    cp(lb)
    n = len(lb["instances"].bboxes)
    assert n == 2, f"应贴出 1 个目标，实际框数 {n}"
    assert not np.array_equal(before, lb["img"]), "图像没变"
    assert lb["cls"].shape[0] == n, "cls 行数与框数不一致"
    ob = lb["instances"].bboxes
    assert not np.allclose(ob[0], ob[1]), "新框与原框重合"
    print(f"[selftest] ② prob=1 粘贴 ✔ {cp.stats()}")

    # ③ 源框过大 ⇒ 不作为源
    cp = SmallObjectCopyPaste(prob=1.0, max_src_px=4.0)
    lb = _labels()
    cp(lb)
    assert len(lb["instances"].bboxes) == 1, "超过 max_src_px 的框被当成源了"
    print("[selftest] ③ 尺寸门限 ✔")

    # ④ match_bg ⇒ 落点背景亮度必须与源上下文同量级
    #   构造：目标在**暗块**里（上下文≈40），整图其余是亮区（200）⇒ 落点亮度不匹配 ⇒ 不贴
    img = np.full((640, 640, 3), 200, dtype=np.uint8)
    img[100:140, 100:140] = 40
    img[110:116, 110:116] = 220
    lb = {"img": img,
          "instances": _mk_inst([[110.0, 110.0, 116.0, 116.0]]),
          "cls": np.array([[0.0]], dtype=np.float32)}
    cp = SmallObjectCopyPaste(prob=1.0, max_per_img=2, match_bg=True, max_tries=40)
    cp(lb)
    assert len(lb["instances"].bboxes) == 1, "背景亮度不匹配时不该贴出目标"
    print("[selftest] ④ match_bg 拒绝亮度不匹配的落点 ✔")

    # ⑤ match_bg=False ⇒ 允许贴到任意落点
    lb = {"img": img.copy(),
          "instances": _mk_inst([[110.0, 110.0, 116.0, 116.0]]),
          "cls": np.array([[0.0]], dtype=np.float32)}
    cp = SmallObjectCopyPaste(prob=1.0, max_per_img=2, match_bg=False, max_tries=40)
    cp(lb)
    assert len(lb["instances"].bboxes) == 2, "match_bg=False 时应贴得上"
    print("[selftest] ⑤ match_bg=False 可贴 ✔")

    # ⑥ **默认 scale=1.0 ⇒ 与旧实现逐位一致**（黄金 sha256 + 黄金框）
    for kwargs in ({}, {"scale_min": 1.0, "scale_max": 1.0}):
        np.random.seed(20261113)
        cp = SmallObjectCopyPaste(prob=1.0, max_src_px=12.0, max_per_img=3, **kwargs)
        lb = _golden_labels()
        cp(lb)
        digest = hashlib.sha256(lb["img"].tobytes()).hexdigest()
        assert digest == GOLDEN_SHA256, f"scale=1.0 时输出与旧实现不一致：{digest}"
        ob = [[float(v) for v in b] for b in np.asarray(lb["instances"].bboxes)]
        assert ob == GOLDEN_BOXES, f"scale=1.0 时框与旧实现不一致：{ob}"
    print("[selftest] ⑥ 默认 scale=1.0 与旧实现逐位一致 ✔（sha256 命中）")

    # ⑦ scale=0.5 ⇒ 贴出的框显著变小（6px 源 → 3px）、标签数量正确
    np.random.seed(20261113)
    cp = SmallObjectCopyPaste(prob=1.0, max_src_px=12.0, max_per_img=3,
                              scale_min=0.5, scale_max=0.5)
    lb = _labels()
    cp(lb)
    ob = np.asarray(lb["instances"].bboxes)
    assert len(ob) == 2, f"scale=0.5 应贴出 1 个目标，实际框数 {len(ob)}"
    assert lb["cls"].shape[0] == 2, "cls 行数与框数不一致"
    nb = ob[1]
    nw, nh = float(nb[2] - nb[0]), float(nb[3] - nb[1])
    assert (nw, nh) == (3.0, 3.0), f"6px 源 @scale0.5 应贴成 3px，实际 {nw}x{nh}"
    assert abs(float(ob[0][2] - ob[0][0]) - 6.0) < 1e-6, "原框被改了"
    print(f"[selftest] ⑦ scale=0.5 贴出框 {nw:.0f}x{nh:.0f}（源 6x6）✔")

    # ⑧ 缩放后 <2px 的目标被跳过（同图里其它目标照贴）
    img = np.full((640, 640, 3), 40, dtype=np.uint8)
    img[300:306, 300:306] = 220        # 6px 源 @scale0.2 → 1px ⇒ 必须跳过
    img[100:112, 100:112] = 200        # 12px 源 @scale0.2 → 2px ⇒ 照贴
    lb = {"img": img,
          "instances": _mk_inst([[300.0, 300.0, 306.0, 306.0],
                                  [100.0, 100.0, 112.0, 112.0]]),
          "cls": np.array([[0.0], [0.0]], dtype=np.float32)}
    np.random.seed(20261113)
    cp = SmallObjectCopyPaste(prob=1.0, max_src_px=12.0, max_per_img=3,
                              scale_min=0.2, scale_max=0.2)
    cp(lb)
    ob = np.asarray(lb["instances"].bboxes)
    assert len(ob) == 3, f"只有 12px 源该被贴出，实际框数 {len(ob)}"
    sizes = sorted((float(b[2] - b[0]), float(b[3] - b[1])) for b in ob)
    assert sizes == [(2.0, 2.0), (6.0, 6.0), (12.0, 12.0)], f"框尺寸不符：{sizes}"
    assert cp.stats()["objects_pasted"] == 1, "缩放后 <2px 的目标不该被贴出"
    print("[selftest] ⑧ 缩放后 <2px 的目标被跳过 ✔")

    print("SELFTEST PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
