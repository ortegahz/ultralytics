#!/usr/bin/env python3
"""OpenCV contrib KCF 适配层（``cv2.legacy.TrackerKCF``）。

为什么用 OpenCV 官方实现，而不是自己写一份 numpy KCF
------------------------------------------------------
* **官方实现是社区通用基线**：``cv2.legacy.TrackerKCF`` 的特征、模型更新、尺度估计
  都有公开实现与文献对应，评审时「这是不是 KCF」不需要争论；
* 自写实现一旦和论文公式有偏差，出来的阈值就无法归因到 KCF 这个家族；
* **KCF 只在 opencv_contrib 里**：主仓库 ``opencv-python`` 的
  ``modules/video/src/tracker/`` 里只有 MIL / MedianFlow / GOTURN / Nano / Vit，
  ``grep -ri kcf`` 零命中；contrib 侧的 ``modules/tracking`` 才提供 KCF/MOSSE，
  Python 绑定名是 ``cv2.legacy.TrackerKCF_create``。

安装（服务器侧、与其它环境隔离，不要污染共享 conda env）::

    pip install --no-deps --target /tmp/pylibs_cvcontrib opencv-contrib-python==4.10.0.84
    PYTHONPATH=/tmp/pylibs_cvcontrib python ...

接口约定
--------
* 输入是**灰度帧**（本项目的探针统一按灰度解码），内部转 3 通道喂给 KCF
  （contrib 版默认特征 HOG+颜色名，需要 3 通道；灰度复制后颜色分量是中性的）；
* ``init`` 以点击点为中心切 ``template_size`` 见方的模板；
* ``update`` 返回 ``(ok, (cx, cy))``，``(cx, cy)`` 是目标框中心（亚像素由 KCF 自己给）。
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np


class KcfUnavailable(RuntimeError):
    """Raised when the OpenCV contrib trackers cannot be imported."""


def require_kcf():
    """Return ``cv2.legacy`` if the contrib KCF is importable, else raise."""
    legacy = getattr(cv2, "legacy", None)
    if legacy is None or not hasattr(legacy, "TrackerKCF_create"):
        raise KcfUnavailable(
            "cv2.legacy.TrackerKCF_create is missing -- KCF lives in opencv_contrib "
            "(modules/tracking), the plain opencv-python wheel has no KCF. Install with: "
            "pip install --no-deps --target <dir> opencv-contrib-python==4.10.0.84 and "
            "run with PYTHONPATH=<dir>."
        )
    return legacy


class CvKcfTracker:
    """Thin grayscale-friendly wrapper around ``cv2.legacy.TrackerKCF``.

    ``params`` 可以覆盖 contrib 的运行期参数（``detect_thresh`` / ``resize`` /
    ``interp_factor`` / ... ）。contrib 的 ``Params::read`` 会先重置成默认值、
    再只读 YAML 里出现的键，所以「只写要改的键」即可。
    """

    def __init__(self, template_size: int = 24, params: dict | None = None):
        self.template_size = int(template_size)
        self._tracker = require_kcf().TrackerKCF_create()
        self.params = dict(params or {})
        if self.params:
            self._apply_params(self.params)

    def init(self, gray: np.ndarray, cx: float, cy: float) -> bool:
        """Initialise on the ``template_size`` square centred at ``(cx, cy)``."""
        height, width = gray.shape[:2]
        if height < self.template_size or width < self.template_size:
            return False
        x0 = max(0, min(int(round(cx)) - self.template_size // 2, width - self.template_size))
        y0 = max(0, min(int(round(cy)) - self.template_size // 2, height - self.template_size))
        rect = (x0, y0, self.template_size, self.template_size)
        return bool(self._tracker.init(self._to_color(gray), rect))

    def update(self, gray: np.ndarray) -> tuple[bool, tuple[float, float]]:
        """Advance one frame; returns ``(ok, (cx, cy))`` of the tracked target centre."""
        result = self._tracker.update(self._to_color(gray))
        if isinstance(result, tuple):
            ok, rect = result[0], result[1]
        else:
            ok, rect = bool(result), None
        if not rect:
            return bool(ok), (float("nan"), float("nan"))
        x, y, w, h = rect
        return bool(ok), (float(x) + float(w) / 2.0, float(y) + float(h) / 2.0)

    def _apply_params(self, overrides: dict) -> None:
        """Write the overrides to a YAML node and read them back into the tracker."""
        handle = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
        try:
            handle.write("%YAML:1.0\n---\n")
            for key, value in overrides.items():
                handle.write("%s: %s\n" % (key, value))
            handle.close()
            storage = cv2.FileStorage(handle.name, cv2.FILE_STORAGE_READ)
            try:
                # contrib 的 TrackerKCF::read 收的是 FileNode（不是 FileStorage）；
                # 实测只有 FileStorage.root() 这个节点能被 Params::read 正确解析。
                self._tracker.read(storage.root())
            finally:
                storage.release()
        finally:
            os.unlink(handle.name)

    def dump_params(self) -> dict:
        """Return the tracker's current parameters as a dict (writes them to a YAML)."""
        handle = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
        handle.close()
        try:
            storage = cv2.FileStorage(handle.name, cv2.FILE_STORAGE_WRITE)
            try:
                self._tracker.write(storage)
            finally:
                storage.release()
            values: dict = {}
            for line in Path(handle.name).read_text(encoding="utf-8").splitlines():
                if ":" in line:
                    key, _, value = line.partition(":")
                    values[key.strip()] = value.strip()
            return values
        finally:
            os.unlink(handle.name)

    @staticmethod
    def _to_color(gray: np.ndarray) -> np.ndarray:
        if gray.ndim == 3:
            return gray
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
