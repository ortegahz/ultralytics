#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3 维红外物理退化仿真（v2）。

为什么必须做「物理退化」而不是只用现成坏片
--------------------------------------------
坏片样本只有 4 条 ⇒ 无法把「画质掉到什么程度」和「跟踪跟不住」建立因果关系。
这里的做法是：**对清晰的真实红外窗口逐档施加单轴物理退化**，得到一条
「退化后的实测画质 X → 5 帧跟踪跟住率 Y」的衰减曲线，再从曲线上反解
**物理崩溃拐点**（膝点）。整个过程在**内存里**完成，不写任何临时视频文件。

三个退化族（任务书口径，**JPEG 族已剔除**）
---------------------------------------------
===== ===================== ==================================================
族    物理病因              档位
===== ===================== ==================================================
blur  散焦 / 过度降噪涂抹   GaussianBlur sigma ∈ {0.6, 1.0, 1.5, 2.2, 3.0, 4.0}（6 档）
motion 俯仰抖动 / 机动拖影   线性核 filter2D，核长 L ∈ {3, 5, 7, 9, 11}（5 档）
contrast 热交叉 / 未做 NUC   动态范围向全帧均值线性收缩 alpha ∈ {0.80, 0.65, 0.50, 0.35, 0.25, 0.15}（6 档）
===== ===================== ==================================================

🔴 **为什么没有 JPEG 族**（任务书 §四.2）：v1 与 v2 实测均已证实，JPEG 压缩在红外上会
**平滑微弱热噪声**，导致 ``surround_blockiness`` **不升反降（-16.8%）**，违背单调性假设；
且 DCT 振铃会让 ``target_lap_var`` 不降反升 ⇒ JPEG 轴上**不存在可用的膝点**。
⇒ 重标只保留**散焦模糊 / 运动拖影 / 对比度衰减**三项真实红外退化。

🔴 **为什么没有加性白噪声轴**：任务书同款实测结论——红外图像加性噪声指标与退化方向**相反**
（噪声越大反而越"好"），同样违背单调性假设 ⇒ **本流水线从未引入、也不引入**该轴。

运动模糊的角度按**固定种子**逐窗口随机生成（``theta ∈ [0, 180°)``），
且**窗口内 5 帧方向一致**（模拟一次连续机动，而不是逐帧乱抖）。

执行机：服务器 ``huangzhe@192.168.99.40:32222``（conda env ``uav``，依赖 cv2）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

BLUR_SIGMAS = (0.6, 1.0, 1.5, 2.2, 3.0, 4.0)
MOTION_LENGTHS = (3, 5, 7, 9, 11)
CONTRAST_ALPHAS = (0.80, 0.65, 0.50, 0.35, 0.25, 0.15)

#: 仅三族真实红外退化（**JPEG 族已按任务书 §四.2 剔除**）
FAMILIES = ("blur", "motion", "contrast")


@dataclass(frozen=True)
class Condition:
    """一个退化条件（``family == "clean"`` 表示不退化）。"""

    family: str
    level: float
    label: str

    @property
    def is_clean(self) -> bool:
        return self.family == "clean"


def build_conditions() -> list[Condition]:
    """全部退化条件（clean 放在最前）。"""
    conditions = [Condition("clean", 0.0, "clean")]
    conditions += [Condition("blur", s, "blur_sigma%.1f" % s) for s in BLUR_SIGMAS]
    conditions += [Condition("motion", float(L), "motion_len%d" % L) for L in MOTION_LENGTHS]
    conditions += [Condition("contrast", a, "contrast_a%.2f" % a) for a in CONTRAST_ALPHAS]
    return conditions


def motion_kernel(length: int, angle_deg: float) -> np.ndarray:
    """线性运动模糊核（长度 ``length`` px、方向 ``angle_deg``，行归一化）。"""
    length = max(3, int(length) | 1)          # 强制奇数
    kernel = np.zeros((length, length), dtype=np.float32)
    center = (length - 1) / 2.0
    theta = math.radians(float(angle_deg))
    dx, dy = math.cos(theta), math.sin(theta)
    for index in range(length):
        offset = index - center
        x = int(round(center + offset * dx))
        y = int(round(center + offset * dy))
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1.0
    total = float(kernel.sum())
    if total <= 0.0:
        kernel[int(center), int(center)] = 1.0
        total = 1.0
    return kernel / total


def motion_angle_for(window_key: str, seed: int = 42) -> float:
    """逐窗口**确定性**的运动角度（``theta ∈ [0, 180)``），窗口内 5 帧共用。"""
    import hashlib

    digest = hashlib.md5(("%d|%s" % (int(seed), str(window_key))).encode("utf-8")).digest()
    value = int.from_bytes(digest[:4], "big") / float(1 << 32)
    return float(value * 180.0)


def apply_degradation(frame: np.ndarray, family: str, level: float,
                      angle_deg: float = 0.0) -> np.ndarray:
    """对**单帧**施加单轴退化（不修改输入）。"""
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if family == "clean":
        return gray.copy()
    if family == "blur":
        if float(level) <= 0.0:
            return gray.copy()          # sigma=0 不是合法的 GaussianBlur 参数 ⇒ 视为不退化
        return cv2.GaussianBlur(gray, (0, 0), sigmaX=float(level), sigmaY=float(level))
    if family == "motion":
        kernel = motion_kernel(int(round(float(level))), angle_deg)
        return cv2.filter2D(gray, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    if family == "contrast":
        alpha = float(level)
        mu = float(gray.mean())
        out = alpha * gray.astype(np.float32) + (1.0 - alpha) * mu
        return np.clip(out, 0, 255).astype(np.uint8)
    raise ValueError("未知退化族：%s" % family)


def apply_window_degradation(frames: list[np.ndarray], condition: Condition,
                             angle_deg: float = 0.0) -> list[np.ndarray]:
    """对窗口内**所有帧**施加同一退化（运动模糊时 5 帧方向一致）。"""
    return [apply_degradation(frame, condition.family, condition.level, angle_deg)
            for frame in frames]


def degradation_strength(family: str, level: float) -> float:
    """把档位映射成「退化强度」标量（越大＝越糊/越灰/越压），供单调性测试使用。"""
    if family == "blur":
        return float(level)
    if family == "motion":
        return float(level)
    if family == "contrast":
        return 1.0 - float(level)          # alpha 越小＝退化越强
    raise ValueError("未知退化族：%s" % family)
